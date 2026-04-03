"""Exp 28: Full 50K-doc TTT eval with SGD all-weights unfrozen.
Based on quick test result: bpb=1.1652 on 500 docs (vs 1.1792 baseline).
"""
import torch, os, sys, glob, time, math, io, copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

BASE_DIR = os.environ.get("BASE_DIR", "/runpod-volume/parameter-golf")
RECORDS_DIR = os.path.join(BASE_DIR, "records/track_non_record_16mb/2026-03-24_11L_XSA_SwiGLU_LoRATTT_1xH100")
MIXED_MODEL_PATH = os.path.join(BASE_DIR, "final_model.mixed.ptz")
TTT_MAX_DOCS = int(os.environ.get("TTT_MAX_DOCS", "0"))  # 0 = all
TTT_MIN_DOC_LEN = 32
TTT_TRAIN_CHUNK = 1024
SCORE_CAP = 2048
SGD_LR = float(os.environ.get("SGD_LR", "0.002"))
SGD_MOMENTUM = float(os.environ.get("SGD_MOMENTUM", "0.9"))

os.chdir(BASE_DIR)

XSA_LAST_N = os.environ.get("XSA_LAST_N", "11")
os.environ["XSA_LAST_N"] = XSA_LAST_N
os.environ.setdefault("MLP_MULT", "3")
os.environ.setdefault("NUM_LAYERS", "11")
os.environ.setdefault("VOCAB_SIZE", "1024")
os.environ.setdefault("TRAIN_SEQ_LEN", "2048")

import importlib.util
_spec = importlib.util.spec_from_file_location("train_gpt_records", os.path.join(RECORDS_DIR, "train_gpt.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

GPT = _mod.GPT
load_data_shard = _mod.load_data_shard
dequantize_state_dict_mixed = _mod.dequantize_state_dict_mixed
Hyperparameters = _mod.Hyperparameters
build_sentencepiece_luts = _mod.build_sentencepiece_luts

import sentencepiece as spm
import zstandard

device = torch.device("cuda")

args = Hyperparameters()
sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
    sp, args.vocab_size, device
)

# Load model
print(f"Loading mixed-quantized model from {MIXED_MODEL_PATH}...")
blob = open(MIXED_MODEL_PATH, "rb").read()
raw = zstandard.ZstdDecompressor().decompress(blob)
mixed_state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
base_state_dict = dequantize_state_dict_mixed(mixed_state)
print(f"  Artifact: {len(blob) / 1e6:.2f} MB")
del blob, raw, mixed_state

# Create model
model = GPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap, rope_base=args.rope_base,
    qk_gain_init=args.qk_gain_init, xsa_last_n=args.xsa_last_n,
).to(device)
model.load_state_dict(base_state_dict, strict=True)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Params: {total_params}")

# Save base state for restoration
base_model_state = {k: v.clone() for k, v in model.state_dict().items()}

# Load val tokens
print("Loading val tokens...")
val_shards = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))
val_tokens = torch.cat([load_data_shard(Path(s)).to(device) for s in val_shards]).long()
print(f"  Val tokens: {val_tokens.numel()}")

# Find document boundaries
bos_positions = (val_tokens == 1).nonzero(as_tuple=True)[0].cpu().numpy()
n_all_docs = min(len(bos_positions), TTT_MAX_DOCS) if TTT_MAX_DOCS > 0 else len(bos_positions)

doc_list = []
for d in range(n_all_docs):
    ds = int(bos_positions[d])
    de = int(bos_positions[d + 1]) if d + 1 < len(bos_positions) else val_tokens.numel()
    dl = de - ds
    if dl >= 5:
        doc_list.append((ds, dl))

short_docs = [(ds, dl) for ds, dl in doc_list if dl < TTT_MIN_DOC_LEN]
long_docs = [(ds, dl) for ds, dl in doc_list if dl >= TTT_MIN_DOC_LEN]

print(f"\n{'='*60}")
print(f"Exp 28: SGD All-Weights TTT ({len(doc_list)} docs)")
print(f"SGD lr={SGD_LR}, momentum={SGD_MOMENTUM}")
print(f"Train chunk={TTT_TRAIN_CHUNK}, Score cap={SCORE_CAP}")
print(f"Short docs: {len(short_docs)}, Long docs: {len(long_docs)}")
print(f"{'='*60}\n")

ttt_nll = torch.zeros((), device=device, dtype=torch.float64)
ttt_bytes = torch.zeros((), device=device, dtype=torch.float64)
ttt_tokens = torch.zeros((), device=device, dtype=torch.float64)
t0 = time.perf_counter()

# Short docs: score without TTT
model.eval()
with torch.no_grad():
    for ds, dl in short_docs:
        x = val_tokens[ds:ds + dl - 1].unsqueeze(0)
        y = val_tokens[ds + 1:ds + dl].unsqueeze(0)
        n = dl - 1
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        ttt_nll += loss.to(torch.float64) * n
        ttt_tokens += n
        prev_ids = x.reshape(-1)[:n]
        tgt_ids = y.reshape(-1)[:n]
        tb = base_bytes_lut[tgt_ids].to(torch.float64)
        tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
        ttt_bytes += tb.sum()

# Long docs: train-then-score with SGD all-weights
for di, (ds, dl) in enumerate(long_docs):
    doc = val_tokens[ds:ds + dl]
    doc_len = doc.numel()

    # Restore base weights
    model.load_state_dict(base_model_state, strict=True)

    # Unfreeze all params
    for p in model.parameters():
        p.requires_grad = True

    # Train (1 epoch, SGD)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=SGD_LR, momentum=SGD_MOMENTUM)
    train_chunk_sz = min(TTT_TRAIN_CHUNK, doc_len - 1)
    for cs in range(0, doc_len - 1, train_chunk_sz):
        ce = min(cs + train_chunk_sz, doc_len - 1)
        if ce - cs < 2:
            continue
        tx = doc[cs:ce].unsqueeze(0)
        ty = doc[cs + 1:ce + 1].unsqueeze(0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tloss = model(tx, ty)
        tloss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Score
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    score_len = min(SCORE_CAP, doc_len - 1)
    sx = doc[:score_len].unsqueeze(0)
    sy = doc[1:score_len + 1].unsqueeze(0)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sloss = model(sx, sy).detach()
        ttt_nll += sloss.to(torch.float64) * score_len
        ttt_tokens += score_len
        prev_ids = sx.reshape(-1)[:score_len]
        tgt_ids = sy.reshape(-1)[:score_len]
        tb = base_bytes_lut[tgt_ids].to(torch.float64)
        tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
        ttt_bytes += tb.sum()

    if (di + 1) % 1000 == 0:
        running_bpb = (ttt_nll.item() / math.log(2.0)) / max(ttt_bytes.item(), 1.0)
        elapsed = time.perf_counter() - t0
        eta = elapsed / (di + 1) * (len(long_docs) - di - 1)
        print(f"  Doc {di+1}/{len(long_docs)}: bpb={running_bpb:.4f} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

final_bpb = (ttt_nll.item() / math.log(2.0)) / max(ttt_bytes.item(), 1.0)
elapsed = time.perf_counter() - t0

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Pre-TTT (mixed quant):  val_bpb=1.1877")
print(f"LoRA TTT (baseline):    val_bpb=1.1540  delta=-0.0337")
print(f"SGD all-weights TTT:    val_bpb={final_bpb:.4f}  delta={final_bpb - 1.1877:+.4f}")
print(f"Improvement vs LoRA:    {final_bpb - 1.1540:+.4f}")
print(f"Docs: {len(doc_list)}, Tokens: {ttt_tokens.item():.0f}, Time: {elapsed:.0f}s")
print(f"{'='*60}")
