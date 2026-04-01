"""
Standalone TTT + N-gram eval that exactly replicates the train_gpt.py post-training eval.
Loads mixed-quantized checkpoint, runs LoRA TTT with optional n-gram cache.

Usage:
  NGRAM_EVAL=1 python ttt_ngram_standalone.py
  NGRAM_EVAL=0 python ttt_ngram_standalone.py   # TTT only, should reproduce 1.1573
"""
import torch, os, sys, glob, time, math, io, zlib
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# --- Setup: match original train_gpt.py exactly ---
BASE_DIR = os.environ.get("BASE_DIR", "/root/parameter-golf")
RECORDS_DIR = os.path.join(BASE_DIR, "records/track_non_record_16mb/2026-03-24_11L_XSA_SwiGLU_LoRATTT_1xH100")

os.chdir(BASE_DIR)

# Set env vars BEFORE importing train_gpt (it reads them at import time)
os.environ.setdefault("MLP_MULT", "3")
os.environ.setdefault("NUM_LAYERS", "11")
os.environ.setdefault("VOCAB_SIZE", "1024")
os.environ.setdefault("TRAIN_SEQ_LEN", "2048")

# Import from records/ version
import importlib.util
_spec = importlib.util.spec_from_file_location("train_gpt_records", os.path.join(RECORDS_DIR, "train_gpt.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

GPT = _mod.GPT
load_data_shard = _mod.load_data_shard
dequantize_state_dict_mixed = _mod.dequantize_state_dict_mixed
eval_val = _mod.eval_val
Hyperparameters = _mod.Hyperparameters
build_sentencepiece_luts = _mod.build_sentencepiece_luts

import sentencepiece as spm
import zstandard

device = torch.device("cuda")

# Config
MIXED_MODEL_PATH = os.path.join(BASE_DIR, "final_model.mixed.ptz")
TTT_ENABLED = int(os.environ.get("TTT_ENABLED", "1"))
TTT_RANK = int(os.environ.get("TTT_RANK", "8"))
TTT_LR = float(os.environ.get("TTT_LR", "0.05"))
TTT_CHUNK_SIZE = int(os.environ.get("TTT_CHUNK_SIZE", "256"))
TTT_MIN_DOC_LEN = int(os.environ.get("TTT_MIN_DOC_LEN", "32"))
TTT_MAX_DOCS = int(os.environ.get("TTT_MAX_DOCS", "0"))

# N-gram config (from original train_gpt.py)
NGRAM_ENABLED = int(os.environ.get("NGRAM_EVAL", "0"))
NGRAM_ORDERS = list(range(2, 8))
NGRAM_BUCKETS = 4 * 1024 * 1024
NGRAM_PRIMES = [36313, 27191, 51647, 81929, 131071, 65537, 104729]
NGRAM_MIN_COUNT = 2
NGRAM_ENT_BASE = 0.05
NGRAM_ENT_RANGE = 0.55

print(f"=== Standalone TTT Eval ===")
print(f"  TTT: enabled={TTT_ENABLED} rank={TTT_RANK} lr={TTT_LR} chunk={TTT_CHUNK_SIZE}")
print(f"  N-gram: enabled={NGRAM_ENABLED}")

# --- Tokenizer + LUTs ---
args = Hyperparameters()
sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
    sp, args.vocab_size, device
)

# --- Model: create and load exactly like original ---
# Keep float32 — dequantized int5/int6 weights lose precision in bfloat16
base_model = GPT(
    vocab_size=args.vocab_size,
    num_layers=args.num_layers,
    model_dim=args.model_dim,
    num_heads=args.num_heads,
    num_kv_heads=args.num_kv_heads,
    mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings,
    tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap,
    rope_base=args.rope_base,
    qk_gain_init=args.qk_gain_init,
).to(device)

# Load mixed-quantized checkpoint (same as original roundtrip validation)
print(f"Loading mixed-quantized model from {MIXED_MODEL_PATH}...")
blob = open(MIXED_MODEL_PATH, "rb").read()
raw = zstandard.ZstdDecompressor().decompress(blob)
mixed_state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
# Original uses strict=True
base_model.load_state_dict(dequantize_state_dict_mixed(mixed_state), strict=True)
print(f"  Artifact: {len(blob) / 1e6:.2f} MB")
print(f"  Params: {sum(p.numel() for p in base_model.parameters())}")

# --- Load val tokens ---
print("Loading val tokens...")
val_shards = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))
val_tokens = torch.cat([load_data_shard(Path(s)).to(device) for s in val_shards]).long()
print(f"  Val tokens: {val_tokens.numel()}")

# Baseline already confirmed: val_bpb=1.1914
base_bpb = 1.1914
print(f"Baseline (cached): val_bpb={base_bpb:.4f}")

if not TTT_ENABLED:
    print("TTT disabled, exiting.")
    sys.exit(0)

# --- TTT eval: exact copy from train_gpt.py lines 1337-1620 ---
print("\n--- LoRA TTT ---")

# N-gram helpers (exact copy from original)
def _get_logits(mdl, x_ids):
    xe = mdl.tok_emb(x_ids)
    if getattr(mdl, 'ngram', None) is not None:
        xe = xe + mdl.ngram(x_ids)
    xe = F.rms_norm(xe, (xe.size(-1),))
    xe = mdl.smear(xe)
    x0 = xe
    skips = []
    for ii in range(mdl.num_encoder_layers):
        xe = mdl.blocks[ii](xe, x0)
        skips.append(xe)
    for ii in range(mdl.num_decoder_layers):
        if skips:
            xe = xe + mdl.skip_weights[ii].to(dtype=xe.dtype)[None, None, :] * skips.pop()
        xe = mdl.blocks[mdl.num_encoder_layers + ii](xe, x0)
    xe = mdl.final_norm(xe)
    lp = F.linear(xe, mdl.tok_emb.weight)
    return mdl.logit_softcap * torch.tanh(lp / mdl.logit_softcap)

def _ngram_hash(tokens_np, pos, order):
    if pos < order - 1:
        return -1
    h = 0
    for k in range(order - 1):
        h ^= int(tokens_np[pos - order + 1 + k]) * NGRAM_PRIMES[k]
    return h & (NGRAM_BUCKETS - 1)

def _score_chunk_with_ngram(mdl, x, y, ngram_tables, doc_tokens_np, chunk_start_in_doc):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = _get_logits(mdl, x)
    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    targets = y[0]
    chunk_len = targets.size(0)
    total_nll = 0.0
    for t in range(chunk_len):
        tgt = int(targets[t].item())
        model_lp = log_probs[t]
        doc_pos = chunk_start_in_doc + t + 1
        ng_p = None
        for oi in range(len(NGRAM_ORDERS) - 1, -1, -1):
            order = NGRAM_ORDERS[oi]
            h = _ngram_hash(doc_tokens_np, doc_pos, order)
            if h < 0:
                continue
            bucket = ngram_tables[oi].get(h)
            if bucket is None:
                continue
            total_count = sum(bucket.values())
            if total_count < NGRAM_MIN_COUNT:
                continue
            ng_p = bucket.get(tgt, 0) / total_count
            break
        if ng_p is not None:
            entropy = -float((torch.exp(model_lp) * model_lp).sum().item())
            entropy = max(entropy, 0.0)
            alpha = NGRAM_ENT_BASE + NGRAM_ENT_RANGE / (1.0 + math.exp(-2.0 * (entropy - 4.0)))
            mixed_p = (1 - alpha) * math.exp(float(model_lp[tgt].item())) + alpha * ng_p
            total_nll += -math.log(max(mixed_p, 1e-10))
        else:
            total_nll += -float(model_lp[tgt].item())
        for oi, order in enumerate(NGRAM_ORDERS):
            h = _ngram_hash(doc_tokens_np, doc_pos, order)
            if h < 0:
                continue
            if h not in ngram_tables[oi]:
                ngram_tables[oi][h] = {}
            ngram_tables[oi][h][tgt] = ngram_tables[oi][h].get(tgt, 0) + 1
    return total_nll

# LoRA adapter (exact copy from original)
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

# Inject LoRA
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
print(f"LoRA: {len(lora_modules)} modules, {sum(p.numel() for p in lora_params)} params")

# Find document boundaries
bos_positions = (val_tokens == 1).nonzero(as_tuple=True)[0].cpu().numpy()
n_all_docs = len(bos_positions)
if TTT_MAX_DOCS > 0:
    n_all_docs = min(n_all_docs, TTT_MAX_DOCS)

doc_list = []
for d in range(n_all_docs):
    ds = int(bos_positions[d])
    de = int(bos_positions[d + 1]) if d + 1 < len(bos_positions) else val_tokens.numel()
    dl = de - ds
    if dl >= 5:
        doc_list.append((ds, dl))

short_docs = [(ds, dl) for ds, dl in doc_list if dl < TTT_MIN_DOC_LEN]
long_docs = [(ds, dl) for ds, dl in doc_list if dl >= TTT_MIN_DOC_LEN]
print(f"Documents: {len(doc_list)}")
print(f"  Short: {len(short_docs)}, Long: {len(long_docs)}")

# Accumulators (float64 on GPU, matches original)
ttt_nll = torch.zeros((), device=device, dtype=torch.float64)
ttt_bytes = torch.zeros((), device=device, dtype=torch.float64)
ttt_tokens = torch.zeros((), device=device, dtype=torch.float64)
t_ttt = time.perf_counter()

# Init n-gram tables (shared across all docs, matches original)
ngram_tables = [{} for _ in range(len(NGRAM_ORDERS))] if NGRAM_ENABLED else None

# Short docs: score without TTT
base_model.eval()
with torch.no_grad():
    for ds, dl in short_docs:
        x = val_tokens[ds:ds + dl - 1].to(device=device, dtype=torch.int64).unsqueeze(0)
        y = val_tokens[ds + 1:ds + dl].to(device=device, dtype=torch.int64).unsqueeze(0)
        n = dl - 1
        if NGRAM_ENABLED:
            doc_np = val_tokens[ds:ds + dl].cpu().numpy()
            chunk_nll = _score_chunk_with_ngram(base_model, x, y, ngram_tables, doc_np, 0)
            ttt_nll += chunk_nll
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = base_model(x, y)
            ttt_nll += loss.to(torch.float64) * n
        ttt_tokens += n
        prev_ids = x.reshape(-1)
        tgt_ids = y.reshape(-1)
        tb = base_bytes_lut[tgt_ids].to(torch.float64)
        tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
        ttt_bytes += tb.sum()

# Long docs: train-then-score (matches original train_gpt_submission.py)
# 1. Train LoRA on all chunks (1024-token chunks, 1 epoch)
# 2. Score full doc in one shot, optionally with n-gram interpolation
TTT_TRAIN_CHUNK = 1024

for di, (ds, dl) in enumerate(long_docs):
    doc = val_tokens[ds:ds + dl].to(device=device, dtype=torch.int64)
    doc_len = doc.numel()
    doc_np = doc.cpu().numpy() if NGRAM_ENABLED else None

    # --- Step 1: Train LoRA (1 epoch over doc) ---
    for m in lora_modules:
        m.reset()
    base_model.train()
    optimizer = torch.optim.Adam(lora_params, lr=TTT_LR)
    train_chunk_sz = min(TTT_TRAIN_CHUNK, doc_len - 1)
    for cs in range(0, doc_len - 1, train_chunk_sz):
        ce = min(cs + train_chunk_sz, doc_len - 1)
        if ce - cs < 2:
            continue
        tx = doc[cs:ce].unsqueeze(0)
        ty = doc[cs + 1:ce + 1].unsqueeze(0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tloss = base_model(tx, ty)
        tloss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # --- Step 2: Score with full doc context ---
    base_model.eval()
    score_len = min(args.train_seq_len, doc_len - 1)  # cap at 2048
    sx = doc[:score_len].unsqueeze(0)
    sy = doc[1:score_len + 1].unsqueeze(0)

    with torch.no_grad():
        if NGRAM_ENABLED:
            # N-gram scoring: get per-token logits, apply n-gram interpolation
            chunk_nll = _score_chunk_with_ngram(base_model, sx, sy, ngram_tables, doc_np, 0)
            ttt_nll += chunk_nll
        else:
            # Standard scoring
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                sloss = base_model(sx, sy).detach()
            ttt_nll += sloss.to(torch.float64) * score_len

        ttt_tokens += score_len
        prev_ids = sx.reshape(-1)
        tgt_ids = sy.reshape(-1)
        tb = base_bytes_lut[tgt_ids].to(torch.float64)
        tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
        ttt_bytes += tb.sum()

    if (di + 1) % 1000 == 0:
        elapsed = time.perf_counter() - t_ttt
        running_bpb = (ttt_nll.item() / math.log(2.0)) / max(ttt_bytes.item(), 1.0)
        eta = elapsed / (di + 1) * (len(long_docs) - di - 1)
        print(f"  Doc {di+1}/{len(long_docs)}: bpb={running_bpb:.4f} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

ttt_bpb = (ttt_nll.item() / math.log(2.0)) / max(ttt_bytes.item(), 1.0)
ttt_loss = ttt_nll.item() / max(ttt_tokens.item(), 1.0)
ngram_tag = "+ngram" if NGRAM_ENABLED else ""

print(f"\n{'='*50}")
print(f"RESULTS")
print(f"{'='*50}")
print(f"Standard:     val_bpb={base_bpb:.4f}")
print(f"LoRA TTT{ngram_tag}:     val_bpb={ttt_bpb:.4f}  delta={ttt_bpb - base_bpb:.4f}")
print(f"Docs: {len(doc_list)}, Tokens: {int(ttt_tokens.item())}, Time: {time.perf_counter()-t_ttt:.0f}s")
print(f"{'='*50}")
