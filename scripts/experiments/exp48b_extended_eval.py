"""
Exp 48b: Extended eval for per-block GPTQ bit allocation.

Runs on the already-saved GPTQ artifacts from exp48:
  - Pre-TTT eval (full val set, fast) to isolate quantization effect
  - TTT eval on 5000 docs (10x more than exp48) to reduce noise

We re-run GPTQ + quantize inline since we didn't save the artifacts.
"""

import torch, os, sys, glob, time, math, io, json, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

BASE_DIR = os.environ.get("BASE_DIR", "/runpod-volume/parameter-golf")
RECORDS_DIR = os.path.join(BASE_DIR, "records/track_non_record_16mb/2026-03-24_11L_XSA_SwiGLU_LoRATTT_1xH100")
FLOAT_MODEL_PATH = os.environ.get("FLOAT_MODEL_PATH", os.path.join(BASE_DIR, "exp40d_model.pt"))
N_CALIB = 64
TTT_MAX_DOCS = int(os.environ.get("TTT_MAX_DOCS", "5000"))
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
quantize_state_dict_mixed = _mod.quantize_state_dict_mixed
Hyperparameters = _mod.Hyperparameters
build_sentencepiece_luts = _mod.build_sentencepiece_luts
MLP_QUANT_PATTERNS = _mod.MLP_QUANT_PATTERNS
EMBED_QUANT_PATTERNS = _mod.EMBED_QUANT_PATTERNS
CONTROL_TENSOR_NAME_PATTERNS = _mod.CONTROL_TENSOR_NAME_PATTERNS
eval_val = _mod.eval_val

import sentencepiece as spm
import zstandard
import torch.distributed as dist

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
        if any(p in name for p in EMBED_QUANT_PATTERNS):
            bits = 8
        elif any(p in name for p in MLP_QUANT_PATTERNS):
            bits = 5
        else:
            bits = 6
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


def eval_no_ttt(model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, args):
    """Simple no-TTT eval over full val set."""
    seq_len = args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    loss_sum = 0.0
    token_count = 0.0
    byte_count = 0.0
    model.eval()
    with torch.no_grad():
        for si in range(total_seqs):
            start = si * seq_len
            x = val_tokens[start:start + seq_len].unsqueeze(0)
            y = val_tokens[start + 1:start + seq_len + 1].unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y).detach()
            n = seq_len
            loss_sum += loss.to(torch.float64).item() * n
            token_count += n
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(torch.float64)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
            byte_count += tb.sum().item()
    return loss_sum / token_count, compute_bpb(loss_sum, byte_count)


def eval_ttt(model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, max_docs):
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

        if (di + 1) % 1000 == 0:
            bpb = compute_bpb(ttt_nll, ttt_bytes)
            elapsed = time.perf_counter() - t0
            eta = elapsed / (di + 1) * (len(long_docs) - di - 1)
            print(f"      Doc {di+1}/{len(long_docs)}: bpb={bpb:.4f} elapsed={elapsed:.0f}s eta={eta:.0f}s")

    model.load_state_dict(base_state, strict=True)
    return compute_bpb(ttt_nll, ttt_bytes), ttt_nll / max(ttt_tokens, 1.0), time.perf_counter() - t0


VARIANTS = {
    "A_baseline": {},
    "B_b6_int8": {6: 8},
}


if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"# Exp 48b: Extended Eval — Per-Block GPTQ")
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

    # Load data
    print("Loading tokens...")
    val_shards = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))
    val_tokens = torch.cat([load_data_shard(Path(s)).to(device) for s in val_shards]).long()
    train_shards = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"))
    train_tokens = torch.cat([load_data_shard(Path(s)).to(device) for s in train_shards[:2]]).long()

    random.seed(SEED)
    calib_data = []
    for _ in range(N_CALIB):
        start = random.randint(0, train_tokens.numel() - 2049)
        calib_data.append(train_tokens[start:start + 2048].unsqueeze(0))
    del train_tokens

    results = {}

    for variant_name, block_override in VARIANTS.items():
        print(f"\n{'='*60}")
        print(f"Variant: {variant_name}")
        print(f"{'='*60}")

        model.load_state_dict({k: v.to(device) for k, v in float_state.items()}, strict=False)

        # GPTQ
        print(f"  GPTQ...")
        gptq_state = apply_gptq_custom(model, calib_data, block_override)

        # Quantize + compress
        quant_mixed = quantize_state_dict_mixed_custom(gptq_state, block_override)
        buf = io.BytesIO()
        torch.save(quant_mixed, buf)
        compressed = zstandard.ZstdCompressor(level=22).compress(buf.getvalue())
        artifact_size = len(compressed)
        print(f"  Artifact: {artifact_size:,} bytes ({artifact_size/1e6:.2f} MB)")

        # Decompress + load
        decompressed = zstandard.ZstdDecompressor().decompress(compressed)
        loaded = torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=False)
        dequant = dequantize_state_dict_mixed(loaded)
        model.load_state_dict({k: v.to(device) for k, v in dequant.items()}, strict=True)
        del compressed, decompressed, loaded, dequant, quant_mixed, gptq_state

        # Pre-TTT eval (full val set)
        print(f"  Pre-TTT eval (full val)...")
        t0 = time.perf_counter()
        no_ttt_loss, no_ttt_bpb = eval_no_ttt(
            model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, args
        )
        print(f"  Pre-TTT: val_loss={no_ttt_loss:.4f} val_bpb={no_ttt_bpb:.4f} time={time.perf_counter()-t0:.0f}s")

        # TTT eval (5000 docs)
        print(f"  TTT eval ({TTT_MAX_DOCS} docs)...")
        ttt_bpb, ttt_loss, elapsed = eval_ttt(
            model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, TTT_MAX_DOCS
        )
        print(f"  TTT: val_loss={ttt_loss:.4f} val_bpb={ttt_bpb:.4f} time={elapsed:.0f}s")

        results[variant_name] = {
            "artifact_size": artifact_size,
            "no_ttt_val_loss": no_ttt_loss,
            "no_ttt_val_bpb": no_ttt_bpb,
            "ttt_val_loss": ttt_loss,
            "ttt_val_bpb": ttt_bpb,
            "ttt_docs": TTT_MAX_DOCS,
            "ttt_time_s": elapsed,
        }

    # Summary
    print(f"\n{'#'*60}")
    print(f"# SUMMARY")
    print(f"{'#'*60}")
    baseline = results.get("A_baseline", {})
    print(f"{'Variant':<20} {'Artifact':>10} {'Pre-TTT BPB':>12} {'TTT BPB':>10} {'Pre-TTT Δ':>10} {'TTT Δ':>10}")
    for name, r in results.items():
        pre_delta = r["no_ttt_val_bpb"] - baseline.get("no_ttt_val_bpb", 0)
        ttt_delta = r["ttt_val_bpb"] - baseline.get("ttt_val_bpb", 0)
        print(f"{name:<20} {r['artifact_size']/1e6:>9.2f}M {r['no_ttt_val_bpb']:>12.4f} {r['ttt_val_bpb']:>10.4f} {pre_delta:>+10.4f} {ttt_delta:>+10.4f}")

    out_path = "/runpod-volume/exp48b_extended_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
