"""
Vectorized TTT + N-gram eval. Inner scoring loop is numpy-vectorized per chunk.
Only the cache update is sequential (but cheap: just array increments).
"""
import torch, os, sys, glob, time, math, io
import numpy as np
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, "/runpod-volume/parameter-golf")
os.chdir("/runpod-volume/parameter-golf")
os.environ["MLP_MULT"] = "3"
os.environ["NUM_LAYERS"] = "11"
from train_gpt import (GPT, load_data_shard, dequantize_state_dict_mixed)
import sentencepiece as spm

VOCAB_SIZE = 1024
SEQ_LEN = 2048
MIXED_MODEL_PATH = "/runpod-volume/parameter-golf/final_model.mixed.ptz"

TTT_RANK = 8
TTT_LR = 0.05
TTT_CHUNK_SIZE = 256
TTT_MIN_DOC_LEN = 32

N_ORDERS = 6  # orders 2-7
NGRAM_BUCKETS = 256 * 1024  # 256K — fits in memory with full target arrays
NGRAM_MASK = NGRAM_BUCKETS - 1
NGRAM_PRIMES = np.array([36313, 27191, 51647, 81929, 131071, 65537], dtype=np.int64)
NGRAM_MIN_COUNT = 2
NGRAM_ALPHA = 0.40

device = torch.device("cuda")


class LoRALinear(torch.nn.Module):
    def __init__(self, original, rank=8):
        super().__init__()
        self.original = original
        in_d = original.weight.shape[1]
        out_d = original.weight.shape[0]
        self.lora_A = torch.nn.Parameter(torch.randn(rank, in_d, device=device) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.randn(out_d, rank, device=device) * 0.001)
        self.scale = 1.0 / rank
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = F.linear(x, self.original.weight.to(x.dtype),
                        self.original.bias.to(x.dtype) if self.original.bias is not None else None)
        return base + (x @ self.lora_A.to(x.dtype).T @ self.lora_B.to(x.dtype).T) * self.scale

    def reset(self):
        torch.nn.init.normal_(self.lora_A, std=0.01)
        torch.nn.init.normal_(self.lora_B, std=0.001)


def get_logits(model, x):
    xe = model.tok_emb(x)
    if model.ngram is not None:
        xe = xe + model.ngram(x)
    xe = F.rms_norm(xe, (xe.size(-1),))
    xe = model.smear(xe)
    x0 = xe
    skips = []
    for i in range(model.num_encoder_layers):
        xe = model.blocks[i](xe, x0)
        skips.append(xe)
    for i in range(model.num_decoder_layers):
        if skips:
            xe = xe + model.skip_weights[i].to(dtype=xe.dtype)[None, None, :] * skips.pop()
        xe = model.blocks[model.num_encoder_layers + i](xe, x0)
    xe = model.final_norm(xe)
    lp = F.linear(xe, model.tok_emb.weight)
    return model.logit_softcap * torch.tanh(lp / model.logit_softcap)


def precompute_doc_hashes(doc_np):
    """Compute hashes for all positions and all orders. Returns [doc_len, N_ORDERS] int64 array."""
    n = len(doc_np)
    doc_i64 = doc_np.astype(np.int64)
    hashes = np.full((n, N_ORDERS), -1, dtype=np.int64)
    for oi in range(N_ORDERS):
        order = oi + 2
        if n < order:
            continue
        h = np.zeros(n, dtype=np.int64)
        for k in range(order - 1):
            # tokens at positions [pos - order + 1 + k] for each pos
            # = shifted right by (order - 1 - k) positions
            shift = order - 1 - k
            h[shift:] ^= doc_i64[:n - shift] * int(NGRAM_PRIMES[k])
        h &= NGRAM_MASK
        h[:order - 1] = -1
        hashes[:, oi] = h
    return hashes


def score_chunk_vectorized(log_probs_np, targets_np, doc_hashes, chunk_start,
                           ctx_count, ctx_target):
    """
    Vectorized scoring of a chunk.
    log_probs_np: [chunk_len, VOCAB_SIZE] float32
    targets_np: [chunk_len] int32
    doc_hashes: [doc_len, N_ORDERS] int64
    Returns: total_nll (float)
    """
    chunk_len = len(targets_np)
    # Positions in document for target tokens
    doc_positions = np.arange(chunk_start + 1, chunk_start + 1 + chunk_len, dtype=np.int64)

    # Get model probs for targets
    model_lp_targets = log_probs_np[np.arange(chunk_len), targets_np]  # [chunk_len]
    model_p_targets = np.exp(model_lp_targets)

    # N-gram lookup: try highest order first, per token
    ng_probs = np.full(chunk_len, -1.0, dtype=np.float64)

    for oi in range(N_ORDERS - 1, -1, -1):
        # Which tokens still need an n-gram prediction?
        need = ng_probs < 0
        if not need.any():
            break

        # Get hashes for this order at target positions
        h = doc_hashes[doc_positions, oi]  # [chunk_len]
        valid = need & (h >= 0)
        if not valid.any():
            continue

        idx = np.where(valid)[0]
        buckets = h[idx].astype(np.intp)
        totals = ctx_count[oi][buckets]  # [len(idx)]
        enough = totals >= NGRAM_MIN_COUNT
        idx2 = idx[enough]
        if len(idx2) == 0:
            continue

        buckets2 = buckets[enough]
        tgts2 = targets_np[idx2]
        counts = ctx_target[oi][buckets2, tgts2].astype(np.float64)
        ng_probs[idx2] = counts / totals[enough].astype(np.float64)

    # Interpolate where we have n-gram predictions
    has_ngram = ng_probs >= 0
    mixed_p = model_p_targets.copy().astype(np.float64)
    mixed_p[has_ngram] = (1 - NGRAM_ALPHA) * model_p_targets[has_ngram] + NGRAM_ALPHA * ng_probs[has_ngram]
    mixed_p = np.maximum(mixed_p, 1e-10)

    total_nll = -np.sum(np.log(mixed_p))

    # Update cache (sequential but fast — just array increments)
    for t in range(chunk_len):
        tgt = int(targets_np[t])
        dp = int(doc_positions[t])
        for oi in range(N_ORDERS):
            h = int(doc_hashes[dp, oi])
            if h < 0:
                continue
            ctx_count[oi][h] += 1
            if ctx_target[oi][h, tgt] < 65535:
                ctx_target[oi][h, tgt] += 1

    return float(total_nll)


def main():
    print("=== Vectorized TTT + N-gram Eval ===")
    print(f"Alpha: {NGRAM_ALPHA}, Orders: 2-7, Buckets: {NGRAM_BUCKETS}")

    sp = spm.SentencePieceProcessor()
    sp.Load("./data/tokenizers/fineweb_1024_bpe.model")

    base_bytes_lut = torch.zeros(VOCAB_SIZE, dtype=torch.int64, device=device)
    has_leading_space_lut = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
    is_boundary_token_lut = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
    for tid in range(VOCAB_SIZE):
        piece = sp.IdToPiece(tid) if tid > 0 else ""
        raw = piece.replace("\u2581", " ")
        base_bytes_lut[tid] = len(raw.encode("utf-8"))
        if piece.startswith("\u2581"):
            has_leading_space_lut[tid] = True
            base_bytes_lut[tid] -= 1
        if tid <= 2:
            is_boundary_token_lut[tid] = True

    print("Loading model...")
    model = GPT(
        vocab_size=VOCAB_SIZE, num_layers=11, model_dim=512, num_heads=8,
        num_kv_heads=4, mlp_mult=3, tie_embeddings=True, tied_embed_init_std=0.02,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
    )
    import zstandard
    blob = open(MIXED_MODEL_PATH, "rb").read()
    raw = zstandard.ZstdDecompressor().decompress(blob)
    mixed_state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    model.load_state_dict(dequantize_state_dict_mixed(mixed_state), strict=False)
    model = model.to(device).eval().bfloat16()
    print(f"  Params: {sum(p.numel() for p in model.parameters())}")

    print("Loading val tokens...")
    val_shards = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))
    val_tokens = torch.cat([load_data_shard(Path(s)).to(device) for s in val_shards]).long()
    val_np = val_tokens.cpu().numpy().astype(np.int32)
    print(f"  Val tokens: {len(val_np)}")

    bos_positions = np.where(val_np == 1)[0]
    n_docs = len(bos_positions)
    print(f"  Documents: {n_docs}")

    doc_list = []
    for d in range(n_docs):
        ds = int(bos_positions[d])
        de = int(bos_positions[d + 1]) if d + 1 < n_docs else len(val_np)
        if de - ds >= 5:
            doc_list.append((ds, de - ds))
    short_docs = [(ds, dl) for ds, dl in doc_list if dl < TTT_MIN_DOC_LEN]
    long_docs = [(ds, dl) for ds, dl in doc_list if dl >= TTT_MIN_DOC_LEN]
    print(f"  Short: {len(short_docs)}, Long: {len(long_docs)}")

    # Inject LoRA
    for p in model.parameters():
        p.requires_grad = False
    lora_modules = []
    for blk in model.blocks:
        lq = LoRALinear(blk.attn.c_q, rank=TTT_RANK); blk.attn.c_q = lq; lora_modules.append(lq)
        lv = LoRALinear(blk.attn.c_v, rank=TTT_RANK); blk.attn.c_v = lv; lora_modules.append(lv)
    lora_params = [p for m in lora_modules for p in [m.lora_A, m.lora_B]]
    print(f"  LoRA: {len(lora_modules)} modules")

    # N-gram cache
    mem_gb = N_ORDERS * NGRAM_BUCKETS * VOCAB_SIZE * 2 / 1e9
    print(f"  N-gram cache memory: {mem_gb:.1f} GB")
    ctx_count = [np.zeros(NGRAM_BUCKETS, dtype=np.int32) for _ in range(N_ORDERS)]
    ctx_target = [np.zeros((NGRAM_BUCKETS, VOCAB_SIZE), dtype=np.uint16) for _ in range(N_ORDERS)]

    ttt_nll = 0.0
    ttt_bytes = 0.0
    ttt_tokens = 0
    t0 = time.perf_counter()

    # Short docs
    print("\nScoring short docs...")
    model.eval()
    with torch.no_grad():
        for ds, dl in short_docs:
            x = val_tokens[ds:ds+dl-1].unsqueeze(0)
            y = val_tokens[ds+1:ds+dl].unsqueeze(0)
            doc_np = val_np[ds:ds+dl]
            doc_hashes = precompute_doc_hashes(doc_np)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = get_logits(model, x)
            lp_np = F.log_softmax(logits[0].float(), dim=-1).cpu().numpy()
            tgt_np = y[0].cpu().numpy().astype(np.int32)
            ttt_nll += score_chunk_vectorized(lp_np, tgt_np, doc_hashes, 0, ctx_count, ctx_target)
            n = dl - 1
            ttt_tokens += n
            prev_ids = x.reshape(-1); tgt_ids = y.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(torch.float64)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
            ttt_bytes += tb.sum().item()

    # Long docs
    print("Scoring long docs with TTT + n-gram...")
    for di, (ds, dl) in enumerate(long_docs):
        for m in lora_modules:
            m.reset()
        optimizer = torch.optim.Adam(lora_params, lr=TTT_LR, betas=(0.9, 0.95))
        doc_np = val_np[ds:ds+dl]
        doc_hashes = precompute_doc_hashes(doc_np)
        pred_len = dl - 1

        for chunk_start in range(0, pred_len, TTT_CHUNK_SIZE):
            chunk_end = min(chunk_start + TTT_CHUNK_SIZE, pred_len)
            chunk_len = chunk_end - chunk_start
            if chunk_len < 2:
                continue
            x = val_tokens[ds+chunk_start:ds+chunk_end].unsqueeze(0)
            y = val_tokens[ds+chunk_start+1:ds+chunk_end+1].unsqueeze(0)
            is_last = (chunk_end >= pred_len)

            # Score with vectorized n-gram
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = get_logits(model, x)
            lp_np = F.log_softmax(logits[0].float(), dim=-1).cpu().numpy()
            tgt_np = y[0].cpu().numpy().astype(np.int32)
            chunk_nll = score_chunk_vectorized(lp_np, tgt_np, doc_hashes, chunk_start,
                                               ctx_count, ctx_target)

            ttt_nll += chunk_nll
            ttt_tokens += chunk_len
            prev_ids = x.reshape(-1); tgt_ids = y.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(torch.float64)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
            ttt_bytes += tb.sum().item()

            # Train LoRA
            if not is_last:
                model.train()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if (di + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            bpb = (ttt_nll / math.log(2.0)) / max(ttt_bytes, 1.0)
            print(f"  {di+1}/{len(long_docs)} docs, bpb={bpb:.4f}, {elapsed:.0f}s", flush=True)

    final_bpb = (ttt_nll / math.log(2.0)) / max(ttt_bytes, 1.0)
    final_loss = ttt_nll / max(ttt_tokens, 1)
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"FINAL: TTT + N-gram (vectorized, alpha={NGRAM_ALPHA})")
    print(f"{'='*60}")
    print(f"  val_loss: {final_loss:.4f}")
    print(f"  val_bpb:  {final_bpb:.4f}")
    print(f"  tokens:   {ttt_tokens}")
    print(f"  docs:     {len(doc_list)}")
    print(f"  time:     {elapsed:.0f}s")
    print(f"\n  Comparison:")
    print(f"    Mixed quant only:    1.1930 BPB")
    print(f"    + LoRA TTT:          1.1573 BPB")
    print(f"    + TTT + N-gram:      {final_bpb:.4f} BPB")
    print("Done.")


if __name__ == "__main__":
    main()
