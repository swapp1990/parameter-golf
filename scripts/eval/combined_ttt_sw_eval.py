"""
Combined TTT + Sliding Window eval. ONE loop over documents.

For each document:
  1. Adapt LoRA (1 epoch, LR=0.05)
  2. Score with sliding window (only last stride tokens per window)
  3. Accumulate proper BPB
  4. Reset LoRA

Usage:
  Local test:  python combined_ttt_sw_eval.py --local --docs 5
  Pod:         EVAL_DOCS=50000 python combined_ttt_sw_eval.py
"""
import torch, io, os, sys, time, glob, math, argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

LOCAL_MODE = '--local' in sys.argv

if LOCAL_MODE:
    # Local CPU test with basic 9L model
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from dashboard.layer_xray import GPT
    CHECKPOINT = 'dashboard/checkpoints/model_step_2000.pt'
    VAL_SHARD = 'data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin'
    TOKENIZER = 'data/tokenizers/fineweb_1024_bpe.model'
    SEQ_LEN = 512  # shorter for CPU
    STRIDE = 128
    DEVICE = 'cpu'
    NUM_LAYERS = 9
else:
    # Pod GPU with submission model
    os.environ['MLP_MULT'] = '3'
    os.environ['NUM_LAYERS'] = '11'
    sys.path.insert(0, '/runpod-volume/parameter-golf')
    os.chdir('/runpod-volume/parameter-golf')
    import zstandard
    from train_gpt import GPT, load_data_shard, build_sentencepiece_luts
    CHECKPOINT = 'final_model.mixed.ptz'
    VAL_SHARD = None  # loaded via glob
    TOKENIZER = './data/tokenizers/fineweb_1024_bpe.model'
    SEQ_LEN = 2048
    STRIDE = 256
    DEVICE = 'cuda'
    NUM_LAYERS = 11

import sentencepiece as spm

N_DOCS = int(os.environ.get('EVAL_DOCS', '5' if LOCAL_MODE else '50000'))
LORA_RANK = 8 if not LOCAL_MODE else 4
LORA_LR = 0.05
BOS_ID = 1
VOCAB_SIZE = 1024


class LoRALinear(nn.Module):
    def __init__(self, original, rank=8):
        super().__init__()
        self.original = original
        in_d = original.weight.shape[1]
        out_d = original.weight.shape[0]
        self.lora_A = nn.Parameter(torch.randn(rank, in_d, device=DEVICE) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_d, rank, device=DEVICE) * 0.001)
        self.scale = 1.0 / rank
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = F.linear(x, self.original.weight.to(x.dtype),
                        self.original.bias.to(x.dtype) if self.original.bias is not None else None)
        return base + (x @ self.lora_A.to(x.dtype).T @ self.lora_B.to(x.dtype).T) * self.scale

    def reset(self):
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.normal_(self.lora_B, std=0.001)


def forward_logits(model, x):
    """Manual forward returning logits."""
    xemb = model.tok_emb(x)
    if hasattr(model, 'bigram') and hasattr(model.bigram, 'scale') and model.bigram.scale.item() > 0.01:
        xemb = xemb + model.bigram(x)
    if hasattr(model, 'smear'):
        xemb = model.smear(xemb)
    xemb = F.rms_norm(xemb, (xemb.size(-1),))
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


def score_doc_sliding(model, doc, seq_len, stride):
    """Score a document with sliding window. Returns (total_nll, total_tokens).
    Only scores last `stride` tokens per window (they have most context).
    Short docs (< seq_len) scored entirely."""
    total_nll = 0.0
    total_tokens = 0

    if doc.numel() <= seq_len + 1:
        # Short doc: score all tokens
        x = doc[:-1].unsqueeze(0)
        y = doc[1:].unsqueeze(0)
        with torch.no_grad():
            if DEVICE == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = forward_logits(model, x)
            else:
                logits = forward_logits(model, x)
            per_tok = F.cross_entropy(logits[0].float(), y[0], reduction='none')
        total_nll += per_tok.sum().item()
        total_tokens += per_tok.numel()
    else:
        # Sliding window: score last stride tokens per window
        starts = list(range(0, doc.numel() - seq_len, stride))
        # Ensure we cover the end
        last_start = doc.numel() - seq_len - 1
        if not starts or starts[-1] < last_start:
            starts.append(last_start)

        scored = set()
        with torch.no_grad():
            for s in starts:
                x = doc[s:s + seq_len].unsqueeze(0)
                y = doc[s + 1:s + seq_len + 1].unsqueeze(0)
                if DEVICE == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = forward_logits(model, x)
                else:
                    logits = forward_logits(model, x)

                # Only score last stride tokens
                score_logits = logits[0, -stride:, :]
                score_targets = y[0, -stride:]
                per_tok = F.cross_entropy(score_logits.float(), score_targets, reduction='none')

                # Track which absolute positions we've scored (avoid double-counting)
                abs_start = s + seq_len - stride + 1  # absolute position of first scored token
                for i in range(per_tok.numel()):
                    abs_pos = abs_start + i
                    if abs_pos not in scored and abs_pos < doc.numel():
                        scored.add(abs_pos)
                        total_nll += per_tok[i].item()
                        total_tokens += 1

    return total_nll, total_tokens


def ttt_adapt(model, doc, lora_modules, lora_params, lr, adapt_seq_len=1024):
    """Adapt LoRA on document (1 epoch)."""
    model.train()
    opt = torch.optim.Adam(lora_params, lr=lr)
    csz = min(adapt_seq_len, doc.numel() - 1)
    for cs in range(0, doc.numel() - 1, csz):
        ce = min(cs + csz, doc.numel() - 1)
        if ce - cs < 2:
            continue
        x = doc[cs:ce].unsqueeze(0)
        y = doc[cs + 1:ce + 1].unsqueeze(0)
        if DEVICE == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = forward_logits(model, x)
        else:
            logits = forward_logits(model, x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE).float(), y.reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad()
    model.eval()


def main():
    print(f'=== Combined TTT + Sliding Window Eval ===')
    print(f'Mode: {"LOCAL" if LOCAL_MODE else "POD"}')
    print(f'Docs: {N_DOCS}, SeqLen: {SEQ_LEN}, Stride: {STRIDE}, LoRA: rank={LORA_RANK} lr={LORA_LR}')

    # Load model
    print('\nLoading model...')
    if LOCAL_MODE:
        model = GPT()
        sd = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        model.load_state_dict(sd, strict=False)
    else:
        with open(CHECKPOINT, 'rb') as f:
            blob = f.read()
        print(f'Artifact: {len(blob)/1024/1024:.2f} MB')
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
                q, scale = qs[name + '.__q'], qs[name + '.__scale']
                dtype = getattr(torch, qs[name + '.__dtype'].split('.')[-1])
                if q.ndim == 2 and scale.ndim == 1:
                    recovered[name] = (q.float() * scale.float()[:, None]).to(dtype)
                else:
                    recovered[name] = (q.float() * scale.float()).to(dtype)
            elif not any(key.endswith(s) for s in ('.__scale', '.__dtype')):
                recovered[key] = qs[key]
        model = GPT(vocab_size=1024, num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
                    mlp_mult=3, tie_embeddings=True, tied_embed_init_std=0.02,
                    logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5)
        model.load_state_dict(recovered, strict=False)

    model = model.to(DEVICE)
    print(f'Params: {sum(p.numel() for p in model.parameters())}')

    # Load val tokens
    print('Loading val tokens...')
    if LOCAL_MODE:
        header_bytes = 256 * np.dtype("<i4").itemsize
        tokens_np = np.fromfile(VAL_SHARD, dtype="<u2", offset=header_bytes)
        val_tokens = torch.from_numpy(tokens_np.astype(np.int64)).to(DEVICE)
    else:
        val_shards = sorted(glob.glob('./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin'))
        val_tokens = torch.cat([load_data_shard(Path(s)).to(DEVICE) for s in val_shards]).long()
    print(f'Val tokens: {val_tokens.numel()}')

    # Find document boundaries
    bos_pos = (val_tokens == BOS_ID).nonzero(as_tuple=True)[0].cpu().numpy()
    n_docs = min(N_DOCS, len(bos_pos))
    print(f'Documents: {n_docs}')

    # Freeze base, inject LoRA
    for p in model.parameters():
        p.requires_grad = False
    lora_modules = []
    for blk in model.blocks:
        lq = LoRALinear(blk.attn.c_q, rank=LORA_RANK)
        blk.attn.c_q = lq
        lora_modules.append(lq)
        lv = LoRALinear(blk.attn.c_v, rank=LORA_RANK)
        blk.attn.c_v = lv
        lora_modules.append(lv)
    lora_params = []
    for m in lora_modules:
        lora_params.extend([m.lora_A, m.lora_B])
    print(f'LoRA: {len(lora_modules)} modules, {sum(p.numel() for p in lora_params)} params')

    # === Baseline: no TTT, no sliding (standard eval) ===
    print('\n--- Baseline (no TTT, standard scoring, first 500 docs) ---')
    model.eval()
    for m in lora_modules:
        m.reset()
    baseline_nll = 0.0
    baseline_tok = 0
    t0 = time.time()
    n_baseline = min(500, n_docs)
    with torch.no_grad():
        for d in range(n_baseline):
            ds = int(bos_pos[d])
            de = int(bos_pos[d + 1]) if d + 1 < len(bos_pos) else val_tokens.numel()
            doc = val_tokens[ds:de]
            if doc.numel() < 3: continue
            x = doc[:min(SEQ_LEN, doc.numel() - 1)].unsqueeze(0)
            y = doc[1:min(SEQ_LEN + 1, doc.numel())].unsqueeze(0)
            if DEVICE == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = forward_logits(model, x)
            else:
                logits = forward_logits(model, x)
            per_tok = F.cross_entropy(logits[0].float(), y[0], reduction='none')
            baseline_nll += per_tok.sum().item()
            baseline_tok += per_tok.numel()
    baseline_loss = baseline_nll / baseline_tok
    print(f'Baseline: val_loss={baseline_loss:.4f} ({n_baseline} docs) time={time.time()-t0:.0f}s')

    # === Combined: TTT adapt + sliding window score per document ===
    print(f'\n--- Combined TTT + Sliding Window ({n_docs} docs) ---')
    combined_nll = 0.0
    combined_tok = 0
    t1 = time.time()

    for d in range(n_docs):
        ds = int(bos_pos[d])
        de = int(bos_pos[d + 1]) if d + 1 < len(bos_pos) else val_tokens.numel()
        doc = val_tokens[ds:de]
        if doc.numel() < 5: continue

        # 1. Adapt LoRA
        for m in lora_modules:
            m.reset()
        ttt_adapt(model, doc, lora_modules, lora_params, LORA_LR)

        # 2. Score with sliding window
        model.eval()
        doc_nll, doc_tok = score_doc_sliding(model, doc, SEQ_LEN, STRIDE)
        combined_nll += doc_nll
        combined_tok += doc_tok

        if (d + 1) % max(1, n_docs // 20) == 0:
            elapsed = time.time() - t1
            eta = elapsed / (d + 1) * (n_docs - d - 1)
            loss = combined_nll / max(combined_tok, 1)
            print(f'  Doc {d+1}/{n_docs}: loss={loss:.4f} elapsed={elapsed:.0f}s eta={eta:.0f}s')

    combined_loss = combined_nll / max(combined_tok, 1)
    elapsed = time.time() - t1

    print(f'\n{"="*50}')
    print(f'RESULTS')
    print(f'{"="*50}')
    print(f'Baseline (no TTT, standard): val_loss={baseline_loss:.4f}')
    print(f'Combined (TTT + SW s{STRIDE}):   val_loss={combined_loss:.4f}  delta={combined_loss-baseline_loss:+.4f}')
    print(f'Docs: {n_docs}, Tokens scored: {combined_tok}, Time: {elapsed:.0f}s')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
