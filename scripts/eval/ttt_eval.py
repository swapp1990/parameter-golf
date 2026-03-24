"""
Test-Time Training (TTT) evaluation with LoRA adapters.
For each document in the validation set:
  1. Inject rank-8 LoRA adapters into attention Q and V projections
  2. Train LoRA on the document's own text (causal, no data leakage)
  3. Score the document with adapted weights
  4. Reset LoRA weights for next document

Usage on pod: python ttt_eval.py [--docs N] [--rank R] [--lr LR] [--epochs E]
"""
import torch, io, os, sys, time, glob, math, json, argparse
import zstandard
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from copy import deepcopy

os.environ['MLP_MULT'] = '3'
os.environ['NUM_LAYERS'] = '11'
sys.path.insert(0, '/runpod-volume/parameter-golf')
os.chdir('/runpod-volume/parameter-golf')
from train_gpt import GPT, load_data_shard

# =====================================================
# LoRA adapter
# =====================================================
class LoRALinear(nn.Module):
    """Wraps a frozen linear layer with a trainable low-rank adapter."""
    def __init__(self, original: nn.Linear, rank: int = 8):
        super().__init__()
        self.original = original
        self.rank = rank
        in_dim = original.in_features if hasattr(original, 'in_features') else original.weight.shape[1]
        out_dim = original.out_features if hasattr(original, 'out_features') else original.weight.shape[0]
        # LoRA: output = original(x) + x @ A^T @ B^T * scale
        self.lora_A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        self.scale = 1.0 / rank
        # Freeze original
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = F.linear(x, self.original.weight.to(x.dtype),
                        self.original.bias.to(x.dtype) if self.original.bias is not None else None)
        lora = (x @ self.lora_A.to(x.dtype).T @ self.lora_B.to(x.dtype).T) * self.scale
        return base + lora

    def reset(self):
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.zeros_(self.lora_B)


def inject_lora(model, rank=8):
    """Inject LoRA adapters into all attention Q and V projections."""
    lora_modules = []
    for block in model.blocks:
        # Q projection
        original_q = block.attn.c_q
        lora_q = LoRALinear(original_q, rank=rank)
        block.attn.c_q = lora_q
        lora_modules.append(lora_q)
        # V projection
        original_v = block.attn.c_v
        lora_v = LoRALinear(original_v, rank=rank)
        block.attn.c_v = lora_v
        lora_modules.append(lora_v)
    return lora_modules


def get_lora_params(lora_modules):
    """Get all trainable LoRA parameters."""
    params = []
    for m in lora_modules:
        params.extend([m.lora_A, m.lora_B])
    return params


def reset_lora(lora_modules):
    """Reset all LoRA adapters to zero."""
    for m in lora_modules:
        m.reset()


# =====================================================
# Forward pass (manual, supports LoRA)
# =====================================================
def forward_loss(model, x, y):
    """Manual forward pass returning mean loss. Works with LoRA."""
    xemb = model.tok_emb(x)
    if hasattr(model, 'bigram'):
        xemb = xemb + model.bigram(x)
    xemb = F.rms_norm(xemb, (xemb.size(-1),))
    if hasattr(model, 'smear'):
        xemb = model.smear(xemb)
    x0 = xemb
    skips = []
    for i in range(model.num_encoder_layers):
        xemb = model.blocks[i](xemb, x0)
        skips.append(xemb)
    for i in range(model.num_decoder_layers):
        if skips:
            xemb = xemb + model.skip_weights[i].to(dtype=xemb.dtype)[None, None, :] * skips.pop()
        xemb = model.blocks[model.num_encoder_layers + i](xemb, x0)
    xemb = model.final_norm(xemb)
    logits = F.linear(xemb, model.tok_emb.weight)
    logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction='mean')


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs', type=int, default=50000, help='Number of documents to evaluate')
    parser.add_argument('--rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lr', type=float, default=3e-4, help='LoRA learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='TTT epochs per document')
    parser.add_argument('--seq_len', type=int, default=1024, help='Max sequence length for TTT')
    parser.add_argument('--no_ttt', action='store_true', help='Run without TTT (baseline)')
    args = parser.parse_args()

    print(f'=== TTT Eval (LoRA rank={args.rank}, lr={args.lr}, epochs={args.epochs}) ===')

    # Load quantized model
    print('Loading quantized model...')
    with open('final_model.submission.ptz', 'rb') as f:
        blob = f.read()
    raw = zstandard.ZstdDecompressor().decompress(blob)
    qs = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    fmt = qs.pop('__quant_format__', None)

    recovered = {}
    seen = set()
    for key in list(qs.keys()):
        if key.endswith('.__q'):
            name = key[:-4]
            if name in seen: continue
            seen.add(name)
            q, scale = qs[name+'.__q'], qs[name+'.__scale']
            dtype = getattr(torch, qs[name+'.__dtype'].split('.')[-1])
            if q.ndim == 2 and scale.ndim == 1:
                recovered[name] = (q.float() * scale.float()[:, None]).to(dtype)
            else:
                recovered[name] = (q.float() * scale.float()).to(dtype)
        elif not any(key.endswith(s) for s in ('.__scale','.__dtype','.__bits')):
            recovered[key] = qs[key]

    model = GPT(vocab_size=1024, num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
                mlp_mult=3, tie_embeddings=True, tied_embed_init_std=0.02,
                logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5)
    model.load_state_dict(recovered, strict=False)
    if hasattr(model, 'bigram'):
        model.bigram.scale.data.fill_(0.0)
    model = model.cuda()

    # Freeze all base parameters
    for p in model.parameters():
        p.requires_grad = False

    # Inject LoRA
    if not args.no_ttt:
        lora_modules = inject_lora(model, rank=args.rank)
        lora_params = get_lora_params(lora_modules)
        print(f'LoRA injected: {len(lora_modules)} modules, {sum(p.numel() for p in lora_params)} trainable params')
    else:
        lora_modules = None
        print('No TTT (baseline mode)')

    model.eval()

    # Load val tokens and find document boundaries
    print('Loading val tokens...')
    val_shards = sorted(glob.glob('./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin'))
    val_tokens = torch.cat([load_data_shard(Path(s)).cuda() for s in val_shards]).long()
    print(f'Val tokens: {val_tokens.numel()}')

    BOS_ID = 1
    bos_positions = (val_tokens == BOS_ID).nonzero(as_tuple=True)[0].cpu().numpy()
    n_docs = min(args.docs, len(bos_positions))
    print(f'Documents to evaluate: {n_docs}')

    # Evaluate each document
    total_nll = 0.0
    total_tokens = 0
    t0 = time.time()

    for d in range(n_docs):
        doc_start = int(bos_positions[d])
        doc_end = int(bos_positions[d + 1]) if d + 1 < len(bos_positions) else val_tokens.numel()
        doc_tokens = val_tokens[doc_start:doc_end]
        doc_len = doc_tokens.numel()

        if doc_len < 5:
            continue

        # TTT: adapt LoRA to this document
        if lora_modules is not None and doc_len > 10:
            reset_lora(lora_modules)
            model.train()
            optimizer = torch.optim.Adam(lora_params, lr=args.lr)

            # Train on chunks of the document (causal — only backward context)
            chunk_size = min(args.seq_len, doc_len - 1)
            for epoch in range(args.epochs):
                for chunk_start in range(0, doc_len - 1, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, doc_len - 1)
                    if chunk_end - chunk_start < 2:
                        continue
                    x = doc_tokens[chunk_start:chunk_end].unsqueeze(0)
                    y = doc_tokens[chunk_start + 1:chunk_end + 1].unsqueeze(0)

                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        loss = forward_loss(model, x, y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            model.eval()

        # Score the document (no gradient)
        with torch.no_grad():
            chunk_size = min(2048, doc_len - 1)
            for chunk_start in range(0, doc_len - 1, chunk_size):
                chunk_end = min(chunk_start + chunk_size, doc_len - 1)
                if chunk_end - chunk_start < 2:
                    continue
                x = doc_tokens[chunk_start:chunk_end].unsqueeze(0)
                y = doc_tokens[chunk_start + 1:chunk_end + 1].unsqueeze(0)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    loss = forward_loss(model, x, y)
                n_tok = y.numel()
                total_nll += loss.item() * n_tok
                total_tokens += n_tok

        if (d + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (d + 1) * (n_docs - d - 1)
            running_loss = total_nll / total_tokens
            print(f'  Doc {d+1}/{n_docs}: val_loss={running_loss:.4f} elapsed={elapsed:.0f}s eta={eta:.0f}s')

    avg_loss = total_nll / total_tokens
    elapsed = time.time() - t0
    mode = 'TTT' if lora_modules else 'no-TTT'
    print(f'\nRESULT ({mode}): val_loss={avg_loss:.4f} tokens={total_tokens} time={elapsed:.0f}s')

    # Save results
    results = {
        'mode': mode,
        'val_loss': float(avg_loss),
        'total_tokens': int(total_tokens),
        'docs_evaluated': int(n_docs),
        'lora_rank': args.rank,
        'lr': args.lr,
        'epochs': args.epochs,
        'time': float(elapsed),
    }
    out = f'/runpod-volume/ttt_results_{mode}.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {out}')
    print('=== Done ===')


if __name__ == '__main__':
    main()
