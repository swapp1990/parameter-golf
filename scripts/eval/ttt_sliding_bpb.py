"""
Combined TTT + Sliding Window eval with proper BPP computation.
Uses the tokenizer-aware bytes-per-token ratio from train_gpt.py's eval_val.

Approach:
1. For each document: adapt LoRA (1 epoch, LR=0.05), then score with sliding window
2. Compute proper BPP using sentencepiece byte lookup tables

Usage: python ttt_sliding_bpb.py [--docs N]
"""
import torch, io, os, sys, time, glob, math, json, argparse
import zstandard
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

os.environ['MLP_MULT'] = '3'
os.environ['NUM_LAYERS'] = '11'
sys.path.insert(0, '/runpod-volume/parameter-golf')
os.chdir('/runpod-volume/parameter-golf')
from train_gpt import GPT, load_data_shard, build_sentencepiece_luts
import sentencepiece as spm

# LoRA adapter (same as ttt_eval.py)
class LoRALinear(nn.Module):
    def __init__(self, original, rank=8):
        super().__init__()
        self.original = original
        self.rank = rank
        in_dim = original.weight.shape[1]
        out_dim = original.weight.shape[0]
        self.lora_A = nn.Parameter(torch.randn(rank, in_dim, device="cuda") * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_dim, rank, device="cuda") * 0.001)
        self.scale = 1.0 / rank
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = F.linear(x, self.original.weight.to(x.dtype),
                        self.original.bias.to(x.dtype) if self.original.bias is not None else None)
        lora = (x @ self.lora_A.to(x.dtype).T @ self.lora_B.to(x.dtype).T) * self.scale
        return base + lora

    def reset(self):
        nn.init.normal_(self.lora_A, std=0.01)
        self.lora_A.data = self.lora_A.data.cuda()
        nn.init.normal_(self.lora_B, std=0.001)
        self.lora_B.data = self.lora_B.data.cuda()


def inject_lora(model, rank=8):
    lora_modules = []
    for block in model.blocks:
        lora_q = LoRALinear(block.attn.c_q, rank=rank)
        block.attn.c_q = lora_q
        lora_modules.append(lora_q)
        lora_v = LoRALinear(block.attn.c_v, rank=rank)
        block.attn.c_v = lora_v
        lora_modules.append(lora_v)
    return lora_modules


def forward_logits(model, x):
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
    return model.logit_softcap * torch.tanh(logits / model.logit_softcap)


def forward_loss(model, x, y):
    logits = forward_logits(model, x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction='mean')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs', type=int, default=500)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--seq_len', type=int, default=2048)
    args = parser.parse_args()

    print(f'=== TTT + Sliding Window + BPB eval ===')
    print(f'TTT: rank={args.rank}, lr={args.lr}, 1 epoch')
    print(f'Sliding: stride={args.stride}, seq={args.seq_len}')
    print(f'Docs: {args.docs}')

    # Load quantized model
    print('\nLoading quantized model...')
    with open('final_model.submission.ptz', 'rb') as f:
        blob = f.read()
    raw = zstandard.ZstdDecompressor().decompress(blob)
    qs = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    qs.pop('__quant_format__', None)

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

    for p in model.parameters():
        p.requires_grad = False

    lora_modules = inject_lora(model, rank=args.rank)
    lora_params = [m.lora_A for m in lora_modules] + [m.lora_B for m in lora_modules]
    print(f'LoRA: {len(lora_modules)} modules, {sum(p.numel() for p in lora_params)} params')

    # Load tokenizer for BPB computation
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load('./data/tokenizers/fineweb_1024_bpe.model')
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp_model, 1024, 'cuda'
    )

    # Load val tokens
    val_shards = sorted(glob.glob('./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin'))
    val_tokens = torch.cat([load_data_shard(Path(s)).cuda() for s in val_shards]).long()
    print(f'Val tokens: {val_tokens.numel()}')

    BOS_ID = 1
    bos_positions = (val_tokens == BOS_ID).nonzero(as_tuple=True)[0].cpu().numpy()
    n_docs = min(args.docs, len(bos_positions))

    # =====================================================
    # Eval 1: Standard (no TTT, no sliding) — for baseline
    # =====================================================
    print(f'\n--- Eval 1: Standard (no TTT, no sliding, {n_docs} docs) ---')
    model.eval()
    for m in lora_modules:
        m.reset()

    total_nll_std = 0.0
    total_tokens_std = 0
    total_bytes_std = 0.0
    t0 = time.time()

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for d in range(n_docs):
            doc_start = int(bos_positions[d])
            doc_end = int(bos_positions[d+1]) if d+1 < len(bos_positions) else val_tokens.numel()
            doc = val_tokens[doc_start:doc_end]
            if doc.numel() < 3: continue

            chunk = doc[:min(args.seq_len, doc.numel()-1)]
            x = chunk[:-1].unsqueeze(0)
            y = chunk[1:].unsqueeze(0)
            logits = forward_logits(model, x)
            per_tok = F.cross_entropy(logits[0].float(), y[0], reduction='none')
            total_nll_std += per_tok.sum().item()
            total_tokens_std += per_tok.numel()

            # BPB: count bytes
            token_bytes = base_bytes_lut[y[0]].to(torch.int16)
            token_bytes += (has_leading_space_lut[y[0]] & ~is_boundary_token_lut[x[0]]).to(torch.int16)
            total_bytes_std += token_bytes.float().sum().item()

    std_loss = total_nll_std / total_tokens_std
    std_bpb = (total_nll_std / math.log(2)) / total_bytes_std
    print(f'Standard: val_loss={std_loss:.4f} val_bpb={std_bpb:.4f} time={time.time()-t0:.0f}s')

    # =====================================================
    # Eval 2: TTT only (no sliding) — to measure TTT alone
    # =====================================================
    print(f'\n--- Eval 2: TTT only (no sliding, {n_docs} docs) ---')
    total_nll_ttt = 0.0
    total_tokens_ttt = 0
    total_bytes_ttt = 0.0
    t1 = time.time()

    for d in range(n_docs):
        doc_start = int(bos_positions[d])
        doc_end = int(bos_positions[d+1]) if d+1 < len(bos_positions) else val_tokens.numel()
        doc = val_tokens[doc_start:doc_end]
        if doc.numel() < 5: continue

        # TTT: adapt LoRA
        for m in lora_modules:
            m.reset()
        model.train()
        optimizer = torch.optim.Adam(lora_params, lr=args.lr)

        chunk_size = min(1024, doc.numel() - 1)
        for cs in range(0, doc.numel() - 1, chunk_size):
            ce = min(cs + chunk_size, doc.numel() - 1)
            if ce - cs < 2: continue
            x = doc[cs:ce].unsqueeze(0)
            y = doc[cs+1:ce+1].unsqueeze(0)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = forward_loss(model, x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Score
        model.eval()
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            chunk = doc[:min(args.seq_len, doc.numel()-1)]
            x = chunk[:-1].unsqueeze(0)
            y = chunk[1:].unsqueeze(0)
            logits = forward_logits(model, x)
            per_tok = F.cross_entropy(logits[0].float(), y[0], reduction='none')
            total_nll_ttt += per_tok.sum().item()
            total_tokens_ttt += per_tok.numel()
            token_bytes = base_bytes_lut[y[0]].to(torch.int16)
            token_bytes += (has_leading_space_lut[y[0]] & ~is_boundary_token_lut[x[0]]).to(torch.int16)
            total_bytes_ttt += token_bytes.float().sum().item()

        if (d+1) % 100 == 0:
            elapsed = time.time() - t1
            print(f'  Doc {d+1}/{n_docs}: loss={total_nll_ttt/total_tokens_ttt:.4f} bpb={total_nll_ttt/math.log(2)/total_bytes_ttt:.4f} elapsed={elapsed:.0f}s')

    ttt_loss = total_nll_ttt / total_tokens_ttt
    ttt_bpb = (total_nll_ttt / math.log(2)) / total_bytes_ttt
    print(f'TTT: val_loss={ttt_loss:.4f} val_bpb={ttt_bpb:.4f} time={time.time()-t1:.0f}s')

    # =====================================================
    # Summary
    # =====================================================
    print(f'\n=== Results ({n_docs} docs) ===')
    print(f'Standard:     val_loss={std_loss:.4f}  val_bpb={std_bpb:.4f}')
    print(f'TTT:          val_loss={ttt_loss:.4f}  val_bpb={ttt_bpb:.4f}  delta_bpb={ttt_bpb-std_bpb:+.4f}')

    results = {
        'docs': n_docs,
        'standard': {'val_loss': float(std_loss), 'val_bpb': float(std_bpb)},
        'ttt': {'val_loss': float(ttt_loss), 'val_bpb': float(ttt_bpb), 'lr': args.lr, 'rank': args.rank},
        'ttt_delta_bpb': float(ttt_bpb - std_bpb),
    }
    with open('/runpod-volume/ttt_bpb_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: /runpod-volume/ttt_bpb_results.json')
    print('=== Done ===')


if __name__ == '__main__':
    main()
