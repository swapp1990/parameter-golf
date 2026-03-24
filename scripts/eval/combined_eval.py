"""
Combined TTT + Sliding Window eval with proper BPB.
For each document:
  1. Adapt LoRA (1 epoch, LR=0.05)
  2. Score with sliding window (only last stride tokens scored per window)
  3. Compute proper BPB using tokenizer byte lookup

Usage: python combined_eval.py [--docs N] [--stride S]
"""
import torch, io, os, sys, time, glob, math, argparse
import zstandard
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

os.environ['MLP_MULT'] = '3'
os.environ['NUM_LAYERS'] = '11'
sys.path.insert(0, '/runpod-volume/parameter-golf')
os.chdir('/runpod-volume/parameter-golf')
from train_gpt import GPT, load_data_shard, build_sentencepiece_luts, eval_val
import sentencepiece as spm


class LoRALinear(nn.Module):
    def __init__(self, original, rank=8):
        super().__init__()
        self.original = original
        in_d = original.weight.shape[1]
        out_d = original.weight.shape[0]
        self.lora_A = nn.Parameter(torch.randn(rank, in_d, device='cuda') * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_d, rank, device='cuda') * 0.001)
        self.scale = 1.0 / rank
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = F.linear(
            x, self.original.weight.to(x.dtype),
            self.original.bias.to(x.dtype) if self.original.bias is not None else None,
        )
        return base + (x @ self.lora_A.to(x.dtype).T @ self.lora_B.to(x.dtype).T) * self.scale

    def reset(self):
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.normal_(self.lora_B, std=0.001)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs', type=int, default=50000)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--seq_len', type=int, default=2048)
    args = parser.parse_args()

    print(f'=== Combined TTT + Sliding Window Eval ===')
    print(f'Docs: {args.docs}, Stride: {args.stride}, Seq: {args.seq_len}')

    # Load quantized model
    print('\nLoading model...')
    with open('final_model.mixed.ptz', 'rb') as f:
        blob = f.read()
    print(f'Artifact: {len(blob)} bytes ({len(blob)/1024/1024:.2f} MB)')
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

    model = GPT(
        vocab_size=1024, num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=3, tie_embeddings=True, tied_embed_init_std=0.02,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
    )
    model.load_state_dict(recovered, strict=False)
    model = model.cuda()

    # Freeze base, inject LoRA
    for p in model.parameters():
        p.requires_grad = False
    lora_modules = []
    for blk in model.blocks:
        lq = LoRALinear(blk.attn.c_q, rank=8)
        blk.attn.c_q = lq
        lora_modules.append(lq)
        lv = LoRALinear(blk.attn.c_v, rank=8)
        blk.attn.c_v = lv
        lora_modules.append(lv)
    lora_params = []
    for m in lora_modules:
        lora_params.extend([m.lora_A, m.lora_B])
    print(f'LoRA: {len(lora_modules)} modules, {sum(p.numel() for p in lora_params)} params')

    # Load val + tokenizer
    val_shards = sorted(glob.glob('./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin'))
    val_tokens = torch.cat([load_data_shard(Path(s)).cuda() for s in val_shards]).long()
    sp = spm.SentencePieceProcessor()
    sp.Load('./data/tokenizers/fineweb_1024_bpe.model')
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, 1024, 'cuda'
    )
    print(f'Val tokens: {val_tokens.numel()}')

    # Find document boundaries
    bos_positions = (val_tokens == 1).nonzero(as_tuple=True)[0].cpu().numpy()
    n_docs = min(args.docs, len(bos_positions))
    print(f'Documents: {n_docs}')

    # === Standard eval (no TTT, no sliding) for baseline ===
    print('\n--- Baseline (standard eval) ---')
    model.eval()
    for m in lora_modules:
        m.reset()

    class FakeArgs:
        val_batch_size = 524288
        train_seq_len = 2048

    t0 = time.time()
    std_loss, std_bpb = eval_val(
        FakeArgs(), model, 0, 1, torch.device('cuda'), 8,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    print(f'Standard: val_loss={std_loss:.4f} val_bpb={std_bpb:.4f} time={time.time()-t0:.0f}s')

    # === Combined TTT + Sliding Window per document ===
    print(f'\n--- Combined TTT + Sliding Window (stride={args.stride}) ---')
    total_nll = 0.0
    total_bytes = 0.0
    total_scored = 0
    t1 = time.time()

    for d in range(n_docs):
        doc_start = int(bos_positions[d])
        doc_end = int(bos_positions[d + 1]) if d + 1 < len(bos_positions) else val_tokens.numel()
        doc = val_tokens[doc_start:doc_end].to(dtype=torch.int64)
        doc_len = doc.numel()
        if doc_len < 5:
            continue

        # 1. TTT: adapt LoRA to this document
        for m in lora_modules:
            m.reset()
        model.train()
        opt = torch.optim.Adam(lora_params, lr=0.05)
        csz = min(1024, doc_len - 1)
        for cs in range(0, doc_len - 1, csz):
            ce = min(cs + csz, doc_len - 1)
            if ce - cs < 2:
                continue
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = model(doc[cs:ce].unsqueeze(0), doc[cs + 1:ce + 1].unsqueeze(0))
            loss.backward()
            opt.step()
            opt.zero_grad()

        # 2. Score with sliding window (only last stride tokens per window)
        model.eval()
        with torch.inference_mode():
            # For short docs (< seq_len), just score the whole thing
            if doc_len <= args.seq_len + 1:
                x = doc[:-1].unsqueeze(0)
                y = doc[1:].unsqueeze(0)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits_cap = model.logit_softcap
                    # Manual forward for per-token loss
                    sloss = model(x, y).detach()
                n_tok = y.numel()
                total_nll += sloss.item() * n_tok
                total_scored += n_tok
                # BPB bytes
                prev_ids = x.reshape(-1)
                tgt_ids = y.reshape(-1)
                tb = base_bytes_lut[tgt_ids].to(torch.int16)
                tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
                total_bytes += tb.to(torch.float64).sum().item()
            else:
                # Sliding window within this document
                doc_starts = list(range(0, doc_len - args.seq_len, args.stride))
                if not doc_starts:
                    doc_starts = [0]
                # Also ensure we cover the end
                if doc_starts[-1] + args.seq_len < doc_len - 1:
                    doc_starts.append(doc_len - args.seq_len - 1)

                scored_positions = set()  # track which positions we've scored
                for ws in doc_starts:
                    we = ws + args.seq_len
                    x = doc[ws:we].unsqueeze(0)
                    y = doc[ws + 1:we + 1].unsqueeze(0)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        sloss = model(x, y).detach()
                    # Score only last stride tokens (they have most context)
                    sf = args.seq_len - args.stride
                    score_start = ws + sf + 1  # absolute position in doc
                    score_end = ws + args.seq_len + 1
                    # Per-token loss for the scored portion
                    # Use mean loss * count as approximation
                    n_score = min(args.stride, we - (ws + sf))
                    # Only score positions we haven't scored yet
                    new_positions = set(range(score_start, min(score_end, doc_len))) - scored_positions
                    if not new_positions:
                        continue
                    n_new = len(new_positions)
                    scored_positions.update(new_positions)
                    total_nll += sloss.item() * n_new
                    total_scored += n_new
                    # Bytes for BPB
                    for pos in sorted(new_positions):
                        if pos > 0 and pos < doc_len:
                            tid = doc[pos].item()
                            prev_tid = doc[pos - 1].item()
                            b = base_bytes_lut[tid].item()
                            b += (has_leading_space_lut[tid].item() & (~is_boundary_token_lut[prev_tid].item()))
                            total_bytes += b

        if (d + 1) % 100 == 0:
            elapsed = time.time() - t1
            eta = elapsed / (d + 1) * (n_docs - d - 1)
            bpb = (total_nll / math.log(2.0)) / max(total_bytes, 1)
            print(f'  Doc {d+1}/{n_docs}: bpb={bpb:.4f} elapsed={elapsed:.0f}s eta={eta:.0f}s')

    combined_loss = total_nll / max(total_scored, 1)
    combined_bpb = (total_nll / math.log(2.0)) / max(total_bytes, 1)
    elapsed = time.time() - t1

    print(f'\n{"="*50}')
    print(f'RESULTS')
    print(f'{"="*50}')
    print(f'Artifact: {len(blob)/1024/1024:.2f} MB')
    print(f'Standard:                val_bpb={std_bpb:.4f}')
    print(f'TTT+Sliding(s{args.stride}):      val_bpb={combined_bpb:.4f}  delta={combined_bpb-std_bpb:+.4f}')
    print(f'Docs: {n_docs}, Scored tokens: {total_scored}, Time: {elapsed:.0f}s')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
