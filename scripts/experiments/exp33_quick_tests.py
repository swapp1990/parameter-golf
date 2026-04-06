"""
Quick tests for base model improvements:
  A: GPTQ-lite quantization (optimal clip search)
  B: Logit softcap sweep (15, 20, 25 vs 30)
  D: SGD learning rate sweep (0.001, 0.002, 0.003, 0.005)
  E: Multiple TTT epochs (1, 2, 3)

All no-retrain — eval-only on existing exp26 checkpoint.
500 docs each, ~35 min total on H100.
"""
import torch, os, sys, glob, time, math, io, copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

BASE_DIR = os.environ.get("BASE_DIR", "/runpod-volume/parameter-golf")
RECORDS_DIR = os.path.join(BASE_DIR, "records/track_non_record_16mb/2026-03-24_11L_XSA_SwiGLU_LoRATTT_1xH100")
MIXED_MODEL_PATH = os.path.join(BASE_DIR, "final_model.mixed.ptz")
UNQUANT_MODEL_PATH = os.path.join(BASE_DIR, "final_model.pt")
TTT_MAX_DOCS = int(os.environ.get("TTT_MAX_DOCS", "500"))
TTT_MIN_DOC_LEN = 32
TTT_TRAIN_CHUNK = 1024
SCORE_CAP = 2048

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

# Load mixed-quantized state dict
print("Loading mixed-quantized model...")
blob = open(MIXED_MODEL_PATH, "rb").read()
raw = zstandard.ZstdDecompressor().decompress(blob)
mixed_state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
base_state_dict = dequantize_state_dict_mixed(mixed_state)
print(f"  Artifact: {len(blob) / 1e6:.2f} MB")
del blob, raw, mixed_state

# Load val tokens
print("Loading val tokens...")
val_shards = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))
val_tokens = torch.cat([load_data_shard(Path(s)).to(device) for s in val_shards]).long()

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
print(f"Documents: {len(doc_list)} (short={len(short_docs)}, long={len(long_docs)})")


def create_model(state_dict, softcap=30.0):
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init, xsa_last_n=args.xsa_last_n,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    return model


def run_sgd_ttt(name, model, lr, momentum, n_epochs):
    """SGD all-weights TTT with configurable lr and epochs."""
    base_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    ttt_nll = torch.zeros((), device=device, dtype=torch.float64)
    ttt_bytes = torch.zeros((), device=device, dtype=torch.float64)
    ttt_tokens = torch.zeros((), device=device, dtype=torch.float64)
    t0 = time.perf_counter()

    # Short docs
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
            prev_ids = x.reshape(-1)[:n]; tgt_ids = y.reshape(-1)[:n]
            tb = base_bytes_lut[tgt_ids].to(torch.float64)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
            ttt_bytes += tb.sum()

    # Long docs
    for di, (ds, dl) in enumerate(long_docs):
        doc = val_tokens[ds:ds + dl]
        doc_len = doc.numel()

        model.load_state_dict(base_model_state, strict=True)
        for p in model.parameters():
            p.requires_grad = True

        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        train_chunk_sz = min(TTT_TRAIN_CHUNK, doc_len - 1)

        for epoch in range(n_epochs):
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
            prev_ids = sx.reshape(-1)[:score_len]; tgt_ids = sy.reshape(-1)[:score_len]
            tb = base_bytes_lut[tgt_ids].to(torch.float64)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
            ttt_bytes += tb.sum()

        if (di + 1) % 100 == 0:
            bpb = (ttt_nll.item() / math.log(2.0)) / max(ttt_bytes.item(), 1.0)
            print(f"  [{name}] Doc {di+1}/{len(long_docs)}: bpb={bpb:.4f} elapsed={time.perf_counter()-t0:.0f}s", flush=True)

    final_bpb = (ttt_nll.item() / math.log(2.0)) / max(ttt_bytes.item(), 1.0)
    elapsed = time.perf_counter() - t0
    print(f"  [{name}] DONE: bpb={final_bpb:.4f} time={elapsed:.0f}s", flush=True)

    del base_model_state
    torch.cuda.empty_cache()
    return final_bpb, elapsed


def eval_no_ttt(name, model):
    """Eval without TTT (just forward pass) — for testing quantization/softcap."""
    ttt_nll = torch.zeros((), device=device, dtype=torch.float64)
    ttt_bytes = torch.zeros((), device=device, dtype=torch.float64)
    ttt_tokens = torch.zeros((), device=device, dtype=torch.float64)
    t0 = time.perf_counter()

    model.eval()
    with torch.no_grad():
        for di, (ds, dl) in enumerate(doc_list):
            doc = val_tokens[ds:ds + dl]
            score_len = min(SCORE_CAP, dl - 1)
            sx = doc[:score_len].unsqueeze(0)
            sy = doc[1:score_len + 1].unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(sx, sy).detach()
            ttt_nll += loss.to(torch.float64) * score_len
            ttt_tokens += score_len
            prev_ids = sx.reshape(-1)[:score_len]; tgt_ids = sy.reshape(-1)[:score_len]
            tb = base_bytes_lut[tgt_ids].to(torch.float64)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
            ttt_bytes += tb.sum()

    final_bpb = (ttt_nll.item() / math.log(2.0)) / max(ttt_bytes.item(), 1.0)
    elapsed = time.perf_counter() - t0
    print(f"  [{name}] DONE: bpb={final_bpb:.4f} time={elapsed:.0f}s", flush=True)
    return final_bpb, elapsed


# =========================================================================
# GPTQ-lite quantization
# =========================================================================
def quantize_gptq_lite(state_dict, bits, percentiles=[0.999, 0.9995, 0.9999, 0.99999, 1.0]):
    """Quantize with optimal clip search per row."""
    quantized = {}
    for name, tensor in state_dict.items():
        if tensor.ndim != 2 or not tensor.is_floating_point():
            quantized[name] = tensor
            continue

        max_val = (2 ** (bits - 1)) - 1  # e.g., 31 for int6, 15 for int5
        tensor = tensor.float()  # quantile requires float32
        best_q = None
        best_mse = float('inf')
        best_scale = None

        for pct in percentiles:
            # Clip per row
            row_abs = tensor.abs()
            if pct < 1.0:
                clip_val = torch.quantile(row_abs, pct, dim=1, keepdim=True)
            else:
                clip_val = row_abs.max(dim=1, keepdim=True).values
            clip_val = clip_val.clamp_min(1e-8)

            scale = clip_val / max_val
            q = torch.clamp(torch.round(tensor / scale), -max_val, max_val).to(torch.int8)
            reconstructed = q.float() * scale
            mse = ((tensor - reconstructed) ** 2).sum(dim=1)

            if best_q is None:
                best_q = q.clone()
                best_scale = scale.squeeze(1).clone()
                best_mse = mse.clone()
            else:
                # Per-row best
                improved = mse < best_mse
                if improved.any():
                    best_q[improved] = q[improved]
                    best_scale[improved] = scale.squeeze(1)[improved]
                    best_mse[improved] = mse[improved]

        quantized[name] = (best_q.float() * best_scale[:, None]).to(tensor.dtype)

    return quantized


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    results = {}

    print(f"\n{'#'*60}")
    print(f"# Quick Tests: Quantization, Softcap, SGD LR, Epochs")
    print(f"# {TTT_MAX_DOCS} docs")
    print(f"{'#'*60}")

    # =================================================================
    # Test A: GPTQ-lite quantization (no TTT, just pre-TTT BPB)
    # =================================================================
    print(f"\n{'='*60}")
    print("Test A: GPTQ-lite Quantization (pre-TTT eval)")
    print(f"{'='*60}")

    # Baseline: current mixed quant
    model = create_model(base_state_dict)
    bpb_baseline_noTTT, _ = eval_no_ttt("A0: Mixed quant baseline", model)
    results["A0: Mixed quant (baseline, no TTT)"] = bpb_baseline_noTTT
    del model; torch.cuda.empty_cache()

    # GPTQ-lite: re-quantize with optimal clipping
    # We need the unquantized model for this
    if os.path.exists(UNQUANT_MODEL_PATH):
        print("  Loading unquantized model for GPTQ-lite...")
        unquant_state = torch.load(UNQUANT_MODEL_PATH, map_location="cpu", weights_only=False)

        # Unquantized baseline
        model = create_model(unquant_state)
        bpb_unquant, _ = eval_no_ttt("A1: Unquantized", model)
        results["A1: Unquantized (no TTT)"] = bpb_unquant
        del model; torch.cuda.empty_cache()

        # GPTQ-lite int6
        print("  Applying GPTQ-lite int6...")
        gptq6_state = quantize_gptq_lite(unquant_state, bits=6)
        model = create_model(gptq6_state)
        bpb_gptq6, _ = eval_no_ttt("A2: GPTQ-lite int6", model)
        results["A2: GPTQ-lite int6 (no TTT)"] = bpb_gptq6
        del model, gptq6_state; torch.cuda.empty_cache()

        # GPTQ-lite int8
        print("  Applying GPTQ-lite int8...")
        gptq8_state = quantize_gptq_lite(unquant_state, bits=8)
        model = create_model(gptq8_state)
        bpb_gptq8, _ = eval_no_ttt("A3: GPTQ-lite int8", model)
        results["A3: GPTQ-lite int8 (no TTT)"] = bpb_gptq8
        del model, gptq8_state, unquant_state; torch.cuda.empty_cache()
    else:
        print(f"  WARNING: {UNQUANT_MODEL_PATH} not found, skipping GPTQ-lite tests")

    # =================================================================
    # Test B: Softcap sweep (no TTT, just pre-TTT BPB)
    # =================================================================
    print(f"\n{'='*60}")
    print("Test B: Softcap Sweep (pre-TTT eval)")
    print(f"{'='*60}")

    for softcap in [15.0, 20.0, 25.0, 30.0]:
        model = create_model(base_state_dict, softcap=softcap)
        bpb, _ = eval_no_ttt(f"B: softcap={softcap}", model)
        results[f"B: softcap={softcap} (no TTT)"] = bpb
        del model; torch.cuda.empty_cache()

    # =================================================================
    # Test D: SGD LR sweep (with TTT)
    # =================================================================
    print(f"\n{'='*60}")
    print("Test D: SGD Learning Rate Sweep")
    print(f"{'='*60}")

    for lr in [0.001, 0.002, 0.003, 0.005]:
        model = create_model(base_state_dict)
        bpb, elapsed = run_sgd_ttt(f"D: SGD lr={lr}", model, lr=lr, momentum=0.9, n_epochs=1)
        results[f"D: SGD lr={lr} (1 epoch)"] = bpb
        del model; torch.cuda.empty_cache()

    # =================================================================
    # Test E: Multiple epochs (with best LR from D, or default 0.002)
    # =================================================================
    print(f"\n{'='*60}")
    print("Test E: Multiple TTT Epochs")
    print(f"{'='*60}")

    for epochs in [2, 3]:
        model = create_model(base_state_dict)
        bpb, elapsed = run_sgd_ttt(f"E: {epochs} epochs", model, lr=0.002, momentum=0.9, n_epochs=epochs)
        results[f"E: SGD lr=0.002 ({epochs} epochs)"] = bpb
        del model; torch.cuda.empty_cache()

    # =================================================================
    # Summary
    # =================================================================
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY ({TTT_MAX_DOCS} docs)")
    print(f"{'='*70}")
    print(f"{'Test':<45} {'BPB':>8}")
    print(f"{'-'*45} {'-'*8}")
    for name, bpb in results.items():
        print(f"{name:<45} {bpb:>8.4f}")
    print(f"{'='*70}")
