"""
Exp 48: Per-Block GPTQ Bit Allocation

B6 does 63% of loss reduction. Currently all attention blocks get int6.
Test: give B6 int8 (or even float16) while keeping everything else at int5/int6.

Variants:
  A: Baseline (current: int5 MLP, int6 attn, int8 embed) — for comparison
  B: B6 at int8 (all other attn at int6)
  C: B5+B6 at int8 (decoder blocks get int8)

For each variant:
  1. Apply GPTQ with the specified bit allocation
  2. Quantize → compress → measure artifact size
  3. Decompress → dequantize → SGD TTT eval (500 docs)
"""

import torch, os, sys, glob, time, math, io, json, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

BASE_DIR = os.environ.get("BASE_DIR", "/runpod-volume/parameter-golf")
RECORDS_DIR = os.path.join(BASE_DIR, "records/track_non_record_16mb/2026-03-24_11L_XSA_SwiGLU_LoRATTT_1xH100")
FLOAT_MODEL_PATH = os.environ.get("FLOAT_MODEL_PATH", os.path.join(BASE_DIR, "exp40d_model.pt"))
N_CALIB = int(os.environ.get("N_CALIB", "64"))
TTT_MAX_DOCS = int(os.environ.get("TTT_MAX_DOCS", "500"))
TTT_MIN_DOC_LEN = 32
TTT_TRAIN_CHUNK = 2048
SCORE_CAP = 2048
SGD_LR = 0.005
SGD_MOMENTUM = 0.9
SEED = 42

os.chdir(BASE_DIR)
os.environ.setdefault("XSA_LAST_N", "7")
os.environ.setdefault("MLP_MULT", "3")
os.environ.setdefault("NUM_LAYERS", "7")
os.environ.setdefault("MODEL_DIM", "624")
os.environ.setdefault("VOCAB_SIZE", "1024")
os.environ.setdefault("TRAIN_SEQ_LEN", "2048")
os.environ.setdefault("LAYER_SCHEDULE", "0,1,2,3,4,3,4,3,4,5,6,5,6")

import importlib.util
_spec = importlib.util.spec_from_file_location("tgp", os.path.join(RECORDS_DIR, "train_gpt.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

GPT = _mod.GPT
load_data_shard = _mod.load_data_shard
dequantize_state_dict_mixed = _mod.dequantize_state_dict_mixed
Hyperparameters = _mod.Hyperparameters
build_sentencepiece_luts = _mod.build_sentencepiece_luts
MLP_QUANT_PATTERNS = _mod.MLP_QUANT_PATTERNS
EMBED_QUANT_PATTERNS = _mod.EMBED_QUANT_PATTERNS
CONTROL_TENSOR_NAME_PATTERNS = _mod.CONTROL_TENSOR_NAME_PATTERNS

import sentencepiece as spm
import zstandard

device = torch.device("cuda")


def compute_bpb(total_nll, total_bytes):
    return (total_nll / math.log(2.0)) / max(total_bytes, 1.0)


def collect_layer_inputs(model, calib_data, module_name):
    inputs_list = []
    hooks = []
    def hook_fn(module, input, output):
        inp = input[0] if isinstance(input, tuple) else input
        inputs_list.append(inp.detach().reshape(-1, inp.shape[-1]).float())
    for name, module in model.named_modules():
        if name == module_name:
            hooks.append(module.register_forward_hook(hook_fn))
            break
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for x in calib_data:
            _ = model(x, torch.roll(x, -1, dims=1))
    for h in hooks:
        h.remove()
    return torch.cat(inputs_list, dim=0) if inputs_list else None


def gptq_quantize_weight(W, H, bits, block_size=128, dampening=0.01):
    rows, cols = W.shape
    max_val = 2 ** (bits - 1) - 1
    scale = W.abs().amax(dim=1) / max_val
    scale = scale.clamp_min(1e-12)
    damp = dampening * H.diag().mean()
    H_damped = H + damp * torch.eye(cols, device=H.device, dtype=H.dtype)
    try:
        L = torch.linalg.cholesky(H_damped)
        H_inv = torch.cholesky_inverse(L)
    except Exception:
        H_inv = torch.diag(1.0 / H_damped.diag().clamp_min(1e-12))
    Q = torch.zeros_like(W)
    W_work = W.clone()
    for col_start in range(0, cols, block_size):
        col_end = min(col_start + block_size, cols)
        for j in range(col_start, col_end):
            w_col = W_work[:, j]
            q_col = torch.clamp(torch.round(w_col / scale), -max_val - 1, max_val)
            Q[:, j] = q_col * scale
            err = w_col - Q[:, j]
            if j + 1 < cols:
                h_jj = H_inv[j, j].clamp_min(1e-12)
                W_work[:, j+1:cols] -= err.unsqueeze(1) * H_inv[j, j+1:cols].unsqueeze(0) / h_jj
    return Q


def quantize_int_per_row(t, bits):
    """Quantize a tensor to N-bit integer with per-row scaling."""
    t32 = t.float()
    max_val = 2 ** (bits - 1) - 1
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / max_val).clamp_min(1e-12).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -max_val - 1, max_val).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(max(amax / max_val, 1e-12), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -max_val - 1, max_val).to(torch.int8)
    return q, scale


def quantize_state_dict_mixed_custom(state_dict, block_bits_override=None):
    """Mixed quantization with optional per-block bit overrides.

    block_bits_override: dict mapping block index -> bits for attention weights.
    e.g., {5: 8, 6: 8} means blocks 5 and 6 get int8 for attention.
    """
    if block_bits_override is None:
        block_bits_override = {}

    result = {}
    for name, t in state_dict.items():
        t_cpu = t.detach().cpu()
        if t_cpu.numel() <= 896 or not t_cpu.is_floating_point():
            result[name] = t_cpu.to(torch.float16) if t_cpu.is_floating_point() else t_cpu
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS) or any(p in name for p in ("smear",)):
            result[name] = t_cpu.to(torch.float16).contiguous()
            continue

        # Determine bits for this weight
        if any(p in name for p in EMBED_QUANT_PATTERNS):
            bits = 8
        elif any(p in name for p in MLP_QUANT_PATTERNS):
            bits = 5
        else:
            # Attention weight — check for block-specific override
            bits = 6  # default
            for block_idx, override_bits in block_bits_override.items():
                if f"blocks.{block_idx}." in name:
                    bits = override_bits
                    break

        q, scale = quantize_int_per_row(t_cpu, bits)
        result[name + ".__q"] = q
        result[name + ".__scale"] = scale
        result[name + ".__dtype"] = str(t_cpu.dtype)

    result["__quant_format__"] = "mixed_v1"
    return result


def apply_gptq_custom(model, calib_data, block_bits_override=None):
    """Apply GPTQ with per-block bit allocation."""
    if block_bits_override is None:
        block_bits_override = {}

    gptq_state = {k: v.cpu() for k, v in model.state_dict().items()}
    n_calib = min(32, len(calib_data))

    for name, param in model.named_parameters():
        if param.ndim != 2 or param.numel() < 4096:
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS) or "smear" in name:
            continue
        if any(p in name for p in EMBED_QUANT_PATTERNS):
            continue

        # Determine bits
        if any(p in name for p in MLP_QUANT_PATTERNS):
            bits = 5
        else:
            bits = 6
            for block_idx, override_bits in block_bits_override.items():
                if f"blocks.{block_idx}." in name:
                    bits = override_bits
                    break

        module_name = ".".join(name.split(".")[:-1])
        print(f"    GPTQ {name} (int{bits})...", end="", flush=True)
        X = collect_layer_inputs(model, calib_data[:n_calib], module_name)
        if X is not None and X.shape[0] > 0:
            H = (X.float().T @ X.float()) / X.shape[0]
            Q = gptq_quantize_weight(param.data.float(), H, bits)
            gptq_state[name] = Q.cpu()
            print(" done")
        else:
            print(" skipped")

    return gptq_state


def eval_ttt(model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, max_docs):
    """SGD all-weights TTT eval."""
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    bos_positions = (val_tokens == 1).nonzero(as_tuple=True)[0].cpu().numpy()
    doc_list = []
    for d in range(len(bos_positions)):
        ds = int(bos_positions[d])
        de = int(bos_positions[d + 1]) if d + 1 < len(bos_positions) else val_tokens.numel()
        dl = de - ds
        if dl >= 5:
            doc_list.append((ds, dl))
    if max_docs > 0:
        doc_list = doc_list[:max_docs]

    short_docs = [(ds, dl) for ds, dl in doc_list if dl < TTT_MIN_DOC_LEN]
    long_docs = [(ds, dl) for ds, dl in doc_list if dl >= TTT_MIN_DOC_LEN]

    ttt_nll = 0.0
    ttt_bytes = 0.0
    ttt_tokens = 0.0
    t0 = time.perf_counter()

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        for ds, dl in short_docs:
            x = val_tokens[ds:ds + dl - 1].unsqueeze(0)
            y = val_tokens[ds + 1:ds + dl].unsqueeze(0)
            n = dl - 1
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            ttt_nll += loss.to(torch.float64).item() * n
            ttt_tokens += n
            prev_ids = x.reshape(-1)[:n]
            tgt_ids = y.reshape(-1)[:n]
            tb = base_bytes_lut[tgt_ids].to(torch.float64)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
            ttt_bytes += tb.sum().item()

    for di, (ds, dl) in enumerate(long_docs):
        doc = val_tokens[ds:ds + dl]
        model.load_state_dict(base_state, strict=True)
        for p in model.parameters():
            p.requires_grad = True
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=SGD_LR, momentum=SGD_MOMENTUM)
        for cs in range(0, dl - 1, TTT_TRAIN_CHUNK):
            ce = min(cs + TTT_TRAIN_CHUNK, dl - 1)
            if ce - cs < 2:
                continue
            tx = doc[cs:ce].unsqueeze(0)
            ty = doc[cs + 1:ce + 1].unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(tx, ty)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        score_len = min(SCORE_CAP, dl - 1)
        sx = doc[:score_len].unsqueeze(0)
        sy = doc[1:score_len + 1].unsqueeze(0)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sloss = model(sx, sy).detach()
        ttt_nll += sloss.to(torch.float64).item() * score_len
        ttt_tokens += score_len
        prev_ids = sx.reshape(-1)[:score_len]
        tgt_ids = sy.reshape(-1)[:score_len]
        tb = base_bytes_lut[tgt_ids].to(torch.float64)
        tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
        ttt_bytes += tb.sum().item()

        if (di + 1) % 100 == 0:
            bpb = compute_bpb(ttt_nll, ttt_bytes)
            elapsed = time.perf_counter() - t0
            print(f"      Doc {di+1}/{len(long_docs)}: bpb={bpb:.4f} elapsed={elapsed:.0f}s")

    model.load_state_dict(base_state, strict=True)
    return compute_bpb(ttt_nll, ttt_bytes), ttt_nll / max(ttt_tokens, 1.0), time.perf_counter() - t0


VARIANTS = {
    "A_baseline": {},              # all attn int6 (current)
    "B_b6_int8": {6: 8},          # B6 at int8, rest int6
    "C_b5b6_int8": {5: 8, 6: 8},  # B5+B6 at int8
}


if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"# Exp 48: Per-Block GPTQ Bit Allocation")
    print(f"{'#'*60}")

    args = Hyperparameters()
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    print(f"\nLoading float model...")
    float_state = torch.load(FLOAT_MODEL_PATH, map_location="cpu", weights_only=False)
    if "model" in float_state:
        float_state = float_state["model"]

    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init, xsa_last_n=args.xsa_last_n,
        layer_schedule=args.layer_schedule if args.layer_schedule else None,
    ).to(device)
    model.load_state_dict(float_state, strict=False)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Load data
    print("Loading tokens...")
    val_shards = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))
    val_tokens = torch.cat([load_data_shard(Path(s)).to(device) for s in val_shards]).long()
    train_shards = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"))
    train_tokens = torch.cat([load_data_shard(Path(s)).to(device) for s in train_shards[:2]]).long()

    # Calibration data
    random.seed(SEED)
    calib_data = []
    for _ in range(N_CALIB):
        start = random.randint(0, train_tokens.numel() - 2049)
        calib_data.append(train_tokens[start:start + 2048].unsqueeze(0))
    del train_tokens

    results = {}

    for variant_name, block_override in VARIANTS.items():
        print(f"\n{'='*60}")
        print(f"Variant: {variant_name} (override: {block_override or 'none'})")
        print(f"{'='*60}")

        # Reload float weights
        model.load_state_dict({k: v.to(device) for k, v in float_state.items()}, strict=False)

        # GPTQ
        print(f"  Applying GPTQ...")
        gptq_state = apply_gptq_custom(model, calib_data, block_override)

        # Quantize with custom bit allocation
        print(f"  Quantizing (mixed, custom bits)...")
        quant_mixed = quantize_state_dict_mixed_custom(gptq_state, block_override)
        buf = io.BytesIO()
        torch.save(quant_mixed, buf)
        compressed = zstandard.ZstdCompressor(level=22).compress(buf.getvalue())
        artifact_size = len(compressed)
        fits = artifact_size < 16_000_000
        print(f"  Artifact: {artifact_size:,} bytes ({artifact_size/1e6:.2f} MB) {'FITS' if fits else 'OVER 16MB!'}")

        if not fits:
            print(f"  SKIPPING eval — artifact too large")
            results[variant_name] = {"artifact_size": artifact_size, "fits": False}
            continue

        # Decompress and load
        decompressed = zstandard.ZstdDecompressor().decompress(compressed)
        loaded = torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=False)
        dequant = dequantize_state_dict_mixed(loaded)
        model.load_state_dict({k: v.to(device) for k, v in dequant.items()}, strict=True)
        del compressed, decompressed, loaded, dequant, quant_mixed, gptq_state

        # TTT eval
        print(f"  Running SGD TTT eval ({TTT_MAX_DOCS} docs)...")
        ttt_bpb, ttt_loss, elapsed = eval_ttt(
            model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, TTT_MAX_DOCS
        )
        print(f"  TTT val_bpb={ttt_bpb:.4f} val_loss={ttt_loss:.4f} time={elapsed:.0f}s")

        results[variant_name] = {
            "block_override": block_override,
            "artifact_size": artifact_size,
            "fits": fits,
            "ttt_val_bpb": ttt_bpb,
            "ttt_val_loss": ttt_loss,
            "ttt_time_s": elapsed,
        }

    # Summary
    print(f"\n{'#'*60}")
    print(f"# SUMMARY")
    print(f"{'#'*60}")
    print(f"{'Variant':<20} {'Artifact':>12} {'TTT BPB':>10} {'Delta vs A':>12}")
    baseline_bpb = results.get("A_baseline", {}).get("ttt_val_bpb", 0)
    for name, r in results.items():
        size_str = f"{r['artifact_size']/1e6:.2f} MB"
        if r.get("ttt_val_bpb"):
            delta = r["ttt_val_bpb"] - baseline_bpb
            sign = "+" if delta >= 0 else ""
            print(f"{name:<20} {size_str:>12} {r['ttt_val_bpb']:>10.4f} {sign}{delta:>11.4f}")
        else:
            print(f"{name:<20} {size_str:>12} {'SKIPPED':>10} {'N/A':>12}")

    out_path = "/runpod-volume/exp48_per_block_gptq_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
