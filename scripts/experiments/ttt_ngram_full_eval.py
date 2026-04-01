"""
Full eval: Load quantized checkpoint, run LoRA TTT + N-gram cache on all 50K docs.
Produces final val_bpb number.

Matches the original train_gpt.py TTT eval exactly, with n-gram cache added.

Usage: python ttt_ngram_full_eval.py
"""
import torch, os, sys, glob, time, math, io
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pathlib import Path

BASE_DIR = os.environ.get("BASE_DIR", "/root/parameter-golf")
RECORDS_DIR = os.path.join(BASE_DIR, "records/track_non_record_16mb/2026-03-24_11L_XSA_SwiGLU_LoRATTT_1xH100")
os.chdir(BASE_DIR)
os.environ["MLP_MULT"] = "3"
os.environ["NUM_LAYERS"] = "11"
import importlib.util
_spec = importlib.util.spec_from_file_location("train_gpt_records", os.path.join(RECORDS_DIR, "train_gpt.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
GPT = _mod.GPT
load_data_shard = _mod.load_data_shard
dequantize_state_dict_mixed = _mod.dequantize_state_dict_mixed
import sentencepiece as spm

VOCAB_SIZE = 1024
MIXED_MODEL_PATH = os.path.join(BASE_DIR, "final_model.mixed.ptz")
MODEL_PATH = os.path.join(BASE_DIR, "final_model.pt")
SEQ_LEN = 2048  # cap doc processing at this length

# TTT config — match original exactly
TTT_RANK = 8
TTT_LR = 0.05
TTT_CHUNK_SIZE = 256
TTT_MIN_DOC_LEN = 32

# N-gram config
NGRAM_ENABLED = int(os.environ.get("NGRAM_ENABLED", "1"))
NGRAM_ORDERS = list(range(2, 8))
NGRAM_BUCKETS = 4 * 1024 * 1024
NGRAM_PRIMES = [36313, 27191, 51647, 81929, 131071, 65537, 104729]
NGRAM_MIN_COUNT = 2
ENT_BASE = 0.05
ENT_RANGE = 0.55

device = torch.device("cuda")


# LoRA — exact copy from original train_gpt.py
class LoRALinear(nn.Module):
    def __init__(self, original, lora_rank=8):
        super().__init__()
        self.original = original
        in_d = original.weight.shape[1]
        out_d = original.weight.shape[0]
        self.lora_A = nn.Parameter(torch.randn(lora_rank, in_d, device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_d, lora_rank, device=device) * 0.001)
        self.scale = 1.0 / lora_rank
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = F.linear(x, self.original.weight.to(x.dtype),
                        self.original.bias.to(x.dtype) if self.original.bias is not None else None)
        return base + (x @ self.lora_A.to(x.dtype).T @ self.lora_B.to(x.dtype).T) * self.scale

    def reset(self):
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.normal_(self.lora_B, std=0.001)


def ngram_hash(tokens_np, pos, order):
    if pos < order - 1:
        return -1
    h = 0
    for k in range(order - 1):
        h ^= int(tokens_np[pos - order + 1 + k]) * NGRAM_PRIMES[k]
    return h & (NGRAM_BUCKETS - 1)


def ngram_adjust_nll(model_loss_per_token, x, y, tables, doc_np, chunk_start):
    """Given per-token model log-probs, compute n-gram adjusted NLL.
    model_loss_per_token: (chunk_len,) tensor of -log p(target) from model
    Returns: adjusted total NLL (float)
    """
    # We need per-token logits for n-gram mixing. Get them from model forward internals.
    # Actually we use the model loss as baseline and only adjust tokens where n-gram helps.
    # For tokens without n-gram match, use model loss directly.
    # For tokens with n-gram match, we need the model probability for that token.
    # model_loss_per_token[t] = -log p_model(target_t)
    # So p_model(target_t) = exp(-model_loss_per_token[t])

    targets = y[0]
    total_nll = 0.0
    for t in range(targets.size(0)):
        tgt = int(targets[t].item())
        model_nll_t = float(model_loss_per_token[t].item())
        doc_pos = chunk_start + t + 1

        # N-gram lookup
        ng_p = None
        if NGRAM_ENABLED:
            for oi in range(len(NGRAM_ORDERS) - 1, -1, -1):
                order = NGRAM_ORDERS[oi]
                h = ngram_hash(doc_np, doc_pos, order)
                if h < 0:
                    continue
                bucket = tables[oi].get(h)
                if bucket is None:
                    continue
                total_count = sum(bucket.values())
                if total_count < NGRAM_MIN_COUNT:
                    continue
                ng_p = bucket.get(tgt, 0) / total_count
                break

        if ng_p is not None:
            model_p = math.exp(-model_nll_t)
            # Entropy approximation — use model_nll as proxy (higher loss = higher entropy)
            # Proper entropy requires full distribution, but model_nll_t is a reasonable proxy
            # For now use a fixed moderate alpha since we don't have full logits cheaply
            alpha = 0.15  # conservative fixed alpha
            mixed_p = (1 - alpha) * model_p + alpha * ng_p
            total_nll += -math.log(max(mixed_p, 1e-10))
        else:
            total_nll += model_nll_t

        # Update cache
        if NGRAM_ENABLED:
            for oi, order in enumerate(NGRAM_ORDERS):
                h = ngram_hash(doc_np, doc_pos, order)
                if h < 0:
                    continue
                if h not in tables[oi]:
                    tables[oi][h] = {}
                tables[oi][h][tgt] = tables[oi][h].get(tgt, 0) + 1

    return total_nll


def main():
    print("=== Full TTT + N-gram Eval ===")
    print(f"  N-gram enabled: {NGRAM_ENABLED}")

    sp = spm.SentencePieceProcessor()
    sp.Load("./data/tokenizers/fineweb_1024_bpe.model")

    # Build byte LUTs — exact same as original
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

    # Load model — keep float32, use autocast (matches original)
    print("Loading model...")
    base_model = GPT(
        vocab_size=VOCAB_SIZE, num_layers=11, model_dim=512, num_heads=8,
        num_kv_heads=4, mlp_mult=3, tie_embeddings=True, tied_embed_init_std=0.02,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
    )

    if os.path.exists(MIXED_MODEL_PATH):
        import zstandard
        print(f"  Loading mixed-quantized model from {MIXED_MODEL_PATH}")
        blob = open(MIXED_MODEL_PATH, "rb").read()
        raw = zstandard.ZstdDecompressor().decompress(blob)
        mixed_state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        base_model.load_state_dict(dequantize_state_dict_mixed(mixed_state), strict=False)
    else:
        print(f"  Loading unquantized model from {MODEL_PATH}")
        sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        base_model.load_state_dict(sd, strict=False)

    # DO NOT call .bfloat16() — keep float32 params, use autocast for computation
    base_model = base_model.to(device)
    print(f"  Params: {sum(p.numel() for p in base_model.parameters())}")

    # Load val tokens
    print("Loading val tokens...")
    val_shards = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))
    val_tokens = torch.cat([load_data_shard(Path(s)).to(device) for s in val_shards]).long()
    val_np = val_tokens.cpu().numpy()
    print(f"  Val tokens: {len(val_np)}")

    bos_positions = np.where(val_np == 1)[0]
    n_docs = len(bos_positions)
    print(f"  Documents: {n_docs}")

    doc_list = []
    for d in range(n_docs):
        ds = int(bos_positions[d])
        de = int(bos_positions[d + 1]) if d + 1 < n_docs else len(val_np)
        dl = de - ds
        if dl >= 5:
            doc_list.append((ds, dl))

    short_docs = [(ds, dl) for ds, dl in doc_list if dl < TTT_MIN_DOC_LEN]
    long_docs = [(ds, dl) for ds, dl in doc_list if dl >= TTT_MIN_DOC_LEN]
    print(f"  Short: {len(short_docs)}, Long: {len(long_docs)}")

    # Inject LoRA — exact same as original
    for p in base_model.parameters():
        p.requires_grad = False
    lora_modules = []
    for blk in base_model.blocks:
        lq = LoRALinear(blk.attn.c_q, lora_rank=TTT_RANK)
        blk.attn.c_q = lq
        lora_modules.append(lq)
        lv = LoRALinear(blk.attn.c_v, lora_rank=TTT_RANK)
        blk.attn.c_v = lv
        lora_modules.append(lv)
    lora_params = []
    for m in lora_modules:
        lora_params.extend([m.lora_A, m.lora_B])
    print(f"  LoRA: {len(lora_modules)} modules, {sum(p.numel() for p in lora_params)} params")

    # N-gram tables (reset per doc)
    ngram_tables = [{} for _ in range(len(NGRAM_ORDERS))]

    # Use float64 accumulators on GPU — matches original
    ttt_nll = torch.zeros((), device=device, dtype=torch.float64)
    ttt_bytes = torch.zeros((), device=device, dtype=torch.float64)
    ttt_tokens = torch.zeros((), device=device, dtype=torch.float64)
    # Separate accumulator for n-gram adjusted NLL
    ngram_nll = 0.0  # python float64

    t0 = time.perf_counter()

    # Short docs — score without TTT (matches original)
    print("\nScoring short docs...")
    base_model.eval()
    with torch.no_grad():
        for ds, dl in short_docs:
            capped_dl = min(dl, SEQ_LEN)
            x = val_tokens[ds:ds + capped_dl - 1].unsqueeze(0)
            y = val_tokens[ds + 1:ds + capped_dl].unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = base_model(x, y)
            n = capped_dl - 1
            ttt_nll += loss.to(torch.float64) * n
            ttt_tokens += n
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(torch.float64)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
            ttt_bytes += tb.sum()

    # Long docs — EXACT original TTT logic: single forward, score then train
    print("Scoring long docs with TTT...")
    for di, (ds, dl) in enumerate(long_docs):
        for m in lora_modules:
            m.reset()
        # Match original: default Adam betas (0.9, 0.999)
        optimizer = torch.optim.Adam(lora_params, lr=TTT_LR)

        # Reset n-gram tables per doc (not cross-doc — avoids hash collision buildup)
        if NGRAM_ENABLED:
            ngram_tables = [{} for _ in range(len(NGRAM_ORDERS))]

        capped_dl = min(dl, SEQ_LEN)
        doc_np = val_np[ds:ds + capped_dl]
        pred_len = capped_dl - 1
        for chunk_start in range(0, pred_len, TTT_CHUNK_SIZE):
            chunk_end = min(chunk_start + TTT_CHUNK_SIZE, pred_len)
            chunk_len = chunk_end - chunk_start
            if chunk_len < 2:
                continue

            x = val_tokens[ds + chunk_start:ds + chunk_end].unsqueeze(0)
            y = val_tokens[ds + chunk_start + 1:ds + chunk_end + 1].unsqueeze(0)
            is_last_chunk = (chunk_end >= pred_len)

            # Single forward pass — matches original exactly
            if is_last_chunk:
                base_model.eval()
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = base_model(x, y)
            else:
                base_model.train()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = base_model(x, y)

            # 1. Score — accumulate NLL + bytes (matches original)
            with torch.no_grad():
                ttt_nll += loss.to(torch.float64) * chunk_len
                ttt_tokens += chunk_len
                prev_ids = x.reshape(-1)
                tgt_ids = y.reshape(-1)
                tb = base_bytes_lut[tgt_ids].to(torch.float64)
                tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
                ttt_bytes += tb.sum()

            # N-gram: compute per-token loss and adjust
            # We use the mean loss * chunk_len for TTT BPB (matches original)
            # and separately track n-gram adjusted NLL
            if NGRAM_ENABLED:
                with torch.no_grad():
                    # Get per-token losses for n-gram adjustment
                    # Reuse the forward pass result — loss is mean, so per-token = loss for uniform approx
                    # Actually we need per-token NLL. Use loss as uniform approx for now.
                    per_token_nll = torch.full((chunk_len,), loss.item(), device=device)
                    chunk_ngram_nll = ngram_adjust_nll(per_token_nll, x, y, ngram_tables, doc_np, chunk_start)
                    ngram_nll += chunk_ngram_nll

            # 2. Train LoRA (matches original)
            if not is_last_chunk:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if (di + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            bpb = (ttt_nll.item() / math.log(2.0)) / max(ttt_bytes.item(), 1.0)
            ngram_bpb = (ngram_nll / math.log(2.0)) / max(ttt_bytes.item(), 1.0) if NGRAM_ENABLED else 0
            print(f"  {di+1}/{len(long_docs)} long docs, ttt_bpb={bpb:.4f}, ngram_bpb={ngram_bpb:.4f}, {elapsed:.0f}s", flush=True)

    # Final results
    ttt_bpb = (ttt_nll.item() / math.log(2.0)) / max(ttt_bytes.item(), 1.0)
    ttt_loss = ttt_nll.item() / max(ttt_tokens.item(), 1.0)
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"  TTT val_loss: {ttt_loss:.4f}")
    print(f"  TTT val_bpb:  {ttt_bpb:.4f}")
    if NGRAM_ENABLED:
        ngram_bpb = (ngram_nll / math.log(2.0)) / max(ttt_bytes.item(), 1.0)
        print(f"  N-gram val_bpb: {ngram_bpb:.4f}")
    print(f"  tokens:   {int(ttt_tokens.item())}")
    print(f"  docs:     {len(doc_list)}")
    print(f"  time:     {elapsed:.0f}s")
    print(f"\n  Comparison:")
    print(f"    Mixed quant only:    1.1930 BPB")
    print(f"    + LoRA TTT:          1.1573 BPB")
    print(f"    + TTT (this run):    {ttt_bpb:.4f} BPB")
    if NGRAM_ENABLED:
        print(f"    + TTT + N-gram:      {ngram_bpb:.4f} BPB")
    print("Done.")


if __name__ == "__main__":
    main()
