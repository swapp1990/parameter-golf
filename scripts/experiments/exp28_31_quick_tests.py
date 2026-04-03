"""
Quick 500-doc TTT eval tests for Exp 28 (SGD TTT) and Exp 31 (Higher LoRA Rank).

Variants:
  A: SGD(lr=0.002, momentum=0.9), all weights unfrozen, 1 epoch, 1024 chunks
  B: SGD(lr=0.002, momentum=0.9), LoRA rank=8 on Q/V, 1 epoch, 1024 chunks
  C: Adam(lr=0.05), LoRA rank=8 on Q/V (current baseline)
  D: Adam(lr=0.05), LoRA rank=16 on Q/V
  E: Adam(lr=0.05), LoRA rank=32 on Q/V

Usage:
  python exp28_31_quick_tests.py
  TTT_MAX_DOCS=100 python exp28_31_quick_tests.py  # faster test
"""
import torch, os, sys, glob, time, math, io, copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# --- Setup ---
BASE_DIR = os.environ.get("BASE_DIR", "/runpod-volume/parameter-golf")
RECORDS_DIR = os.path.join(BASE_DIR, "records/track_non_record_16mb/2026-03-24_11L_XSA_SwiGLU_LoRATTT_1xH100")
MIXED_MODEL_PATH = os.path.join(BASE_DIR, "final_model.mixed.ptz")
TTT_MAX_DOCS = int(os.environ.get("TTT_MAX_DOCS", "500"))
TTT_MIN_DOC_LEN = int(os.environ.get("TTT_MIN_DOC_LEN", "32"))
TTT_TRAIN_CHUNK = 1024
SCORE_CAP = 2048  # max tokens for scoring

os.chdir(BASE_DIR)

# Set env vars BEFORE importing train_gpt (it reads them at import time)
XSA_LAST_N = os.environ.get("XSA_LAST_N", "11")
os.environ["XSA_LAST_N"] = XSA_LAST_N
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
Hyperparameters = _mod.Hyperparameters
build_sentencepiece_luts = _mod.build_sentencepiece_luts

import sentencepiece as spm
import zstandard

device = torch.device("cuda")

# --- Tokenizer + LUTs ---
args = Hyperparameters()
sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
    sp, args.vocab_size, device
)

# --- Load base model state dict (keep on CPU for cloning) ---
print(f"Loading mixed-quantized model from {MIXED_MODEL_PATH}...")
blob = open(MIXED_MODEL_PATH, "rb").read()
raw = zstandard.ZstdDecompressor().decompress(blob)
mixed_state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
base_state_dict = dequantize_state_dict_mixed(mixed_state)
print(f"  Artifact: {len(blob) / 1e6:.2f} MB")
del blob, raw, mixed_state  # free memory

# --- Load val tokens ---
print("Loading val tokens...")
val_shards = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))
val_tokens = torch.cat([load_data_shard(Path(s)).to(device) for s in val_shards]).long()
print(f"  Val tokens: {val_tokens.numel()}")

# --- Find document boundaries ---
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
print(f"Documents: {len(doc_list)} (short={len(short_docs)}, long={len(long_docs)})")

# =========================================================================
# LoRA adapter
# =========================================================================
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


# =========================================================================
# Helper: create fresh model and load base weights
# =========================================================================
def create_base_model():
    model = GPT(
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
        xsa_last_n=args.xsa_last_n,
    ).to(device)
    model.load_state_dict(base_state_dict, strict=True)
    return model


# =========================================================================
# Helper: inject LoRA into model, return (lora_modules, lora_params)
# =========================================================================
def inject_lora(model, rank=8):
    for p in model.parameters():
        p.requires_grad = False
    lora_modules = []
    for blk in model.blocks:
        lq = LoRALinear(blk.attn.c_q, lora_rank=rank)
        blk.attn.c_q = lq
        lora_modules.append(lq)
        lv = LoRALinear(blk.attn.c_v, lora_rank=rank)
        blk.attn.c_v = lv
        lora_modules.append(lv)
    lora_params = []
    for m in lora_modules:
        lora_params.extend([m.lora_A, m.lora_B])
    return lora_modules, lora_params


# =========================================================================
# Helper: compute byte-level BPB accumulators for a scored chunk
# =========================================================================
def accumulate_bytes(sx, sy, score_len, ttt_bytes_acc):
    prev_ids = sx.reshape(-1)[:score_len]
    tgt_ids = sy.reshape(-1)[:score_len]
    tb = base_bytes_lut[tgt_ids].to(torch.float64)
    tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
    ttt_bytes_acc += tb.sum()


# =========================================================================
# Variant runners
# =========================================================================

def run_variant_lora(variant_name, lora_rank, optimizer_factory):
    """Run LoRA-based TTT variant (variants B, C, D, E)."""
    print(f"\n{'='*60}")
    print(f"Variant {variant_name}")
    print(f"{'='*60}")

    model = create_base_model()
    lora_modules, lora_params = inject_lora(model, rank=lora_rank)
    print(f"  LoRA rank={lora_rank}, params={sum(p.numel() for p in lora_params)}")

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
            accumulate_bytes(x, y, n, ttt_bytes)

    # Long docs: train-then-score
    for di, (ds, dl) in enumerate(long_docs):
        doc = val_tokens[ds:ds + dl]
        doc_len = doc.numel()

        # Reset LoRA weights
        for m in lora_modules:
            m.reset()

        # Train LoRA (1 epoch)
        model.train()
        optimizer = optimizer_factory(lora_params)
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
        score_len = min(SCORE_CAP, doc_len - 1)
        sx = doc[:score_len].unsqueeze(0)
        sy = doc[1:score_len + 1].unsqueeze(0)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                sloss = model(sx, sy).detach()
            ttt_nll += sloss.to(torch.float64) * score_len
            ttt_tokens += score_len
            accumulate_bytes(sx, sy, score_len, ttt_bytes)

        if (di + 1) % 100 == 0:
            running_bpb = (ttt_nll.item() / math.log(2.0)) / max(ttt_bytes.item(), 1.0)
            elapsed = time.perf_counter() - t0
            print(f"  [{variant_name}] Doc {di+1}/{len(long_docs)}: bpb={running_bpb:.4f} elapsed={elapsed:.0f}s", flush=True)

    final_bpb = (ttt_nll.item() / math.log(2.0)) / max(ttt_bytes.item(), 1.0)
    elapsed = time.perf_counter() - t0
    print(f"  [{variant_name}] DONE: bpb={final_bpb:.4f} time={elapsed:.0f}s")

    # Free model
    del model, lora_modules, lora_params
    torch.cuda.empty_cache()

    return final_bpb, elapsed


def run_variant_sgd_all_weights(variant_name, lr, momentum):
    """Run SGD with all weights unfrozen (variant A). Clones state per doc."""
    print(f"\n{'='*60}")
    print(f"Variant {variant_name}: SGD all-weights lr={lr} momentum={momentum}")
    print(f"{'='*60}")

    model = create_base_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  All weights unfrozen, params={total_params}")

    # Save base state for restoration after each doc
    base_model_state = {k: v.clone() for k, v in model.state_dict().items()}

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
            accumulate_bytes(x, y, n, ttt_bytes)

    # Long docs: train-then-score, restore weights after each doc
    for di, (ds, dl) in enumerate(long_docs):
        doc = val_tokens[ds:ds + dl]
        doc_len = doc.numel()

        # Restore base weights before each doc
        model.load_state_dict(base_model_state, strict=True)

        # Unfreeze all params
        for p in model.parameters():
            p.requires_grad = True

        # Train all weights (1 epoch, SGD)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
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
            accumulate_bytes(sx, sy, score_len, ttt_bytes)

        if (di + 1) % 100 == 0:
            running_bpb = (ttt_nll.item() / math.log(2.0)) / max(ttt_bytes.item(), 1.0)
            elapsed = time.perf_counter() - t0
            print(f"  [{variant_name}] Doc {di+1}/{len(long_docs)}: bpb={running_bpb:.4f} elapsed={elapsed:.0f}s", flush=True)

    final_bpb = (ttt_nll.item() / math.log(2.0)) / max(ttt_bytes.item(), 1.0)
    elapsed = time.perf_counter() - t0
    print(f"  [{variant_name}] DONE: bpb={final_bpb:.4f} time={elapsed:.0f}s")

    del model, base_model_state
    torch.cuda.empty_cache()

    return final_bpb, elapsed


# =========================================================================
# Main: run all variants sequentially
# =========================================================================
if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"# Exp 28/31 Quick TTT Tests ({TTT_MAX_DOCS} docs)")
    print(f"# XSA_LAST_N={XSA_LAST_N}, train_chunk={TTT_TRAIN_CHUNK}, score_cap={SCORE_CAP}")
    print(f"{'#'*60}")

    results = {}

    # --- Exp 28: SGD TTT ---

    # Variant A: SGD all weights
    bpb_a, time_a = run_variant_sgd_all_weights("A: SGD-all-weights", lr=0.002, momentum=0.9)
    results["A: SGD-all-weights (lr=0.002,m=0.9)"] = (bpb_a, time_a)

    # Variant B: SGD LoRA rank=8
    bpb_b, time_b = run_variant_lora(
        "B: SGD-LoRA-r8",
        lora_rank=8,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.002, momentum=0.9),
    )
    results["B: SGD-LoRA-r8 (lr=0.002,m=0.9)"] = (bpb_b, time_b)

    # Variant C: Adam LoRA rank=8 (baseline)
    bpb_c, time_c = run_variant_lora(
        "C: Adam-LoRA-r8 (baseline)",
        lora_rank=8,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=0.05),
    )
    results["C: Adam-LoRA-r8 (lr=0.05) BASELINE"] = (bpb_c, time_c)

    # --- Exp 31: Higher LoRA Rank ---

    # Variant D: Adam LoRA rank=16
    bpb_d, time_d = run_variant_lora(
        "D: Adam-LoRA-r16",
        lora_rank=16,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=0.05),
    )
    results["D: Adam-LoRA-r16 (lr=0.05)"] = (bpb_d, time_d)

    # Variant E: Adam LoRA rank=32
    bpb_e, time_e = run_variant_lora(
        "E: Adam-LoRA-r32",
        lora_rank=32,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=0.05),
    )
    results["E: Adam-LoRA-r32 (lr=0.05)"] = (bpb_e, time_e)

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY ({TTT_MAX_DOCS} docs)")
    print(f"{'='*70}")
    print(f"{'Variant':<45} {'BPB':>8} {'Time':>8}")
    print(f"{'-'*45} {'-'*8} {'-'*8}")
    for name, (bpb, elapsed) in results.items():
        print(f"{name:<45} {bpb:>8.4f} {elapsed:>7.0f}s")
    print(f"{'='*70}")

    # Find best
    best_name = min(results, key=lambda k: results[k][0])
    best_bpb = results[best_name][0]
    baseline_bpb = results["C: Adam-LoRA-r8 (lr=0.05) BASELINE"][0]
    print(f"\nBest: {best_name} (bpb={best_bpb:.4f}, delta vs baseline={best_bpb - baseline_bpb:+.4f})")
