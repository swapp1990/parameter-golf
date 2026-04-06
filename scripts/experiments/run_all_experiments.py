"""
Run all 5 experiments (Exp 21-25) sequentially on a single GPU (e.g. Vast.ai RTX 4060 Ti 16GB).

Usage:
    python scripts/experiments/run_all_experiments.py [--base-dir /root/parameter-golf] [--experiments 21,22,23,24,25]

Each experiment reloads the model fresh and prints clear results with BPB/loss comparisons.
A final summary table is printed at the end.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import io
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BASE_DIR = os.environ.get("PARAM_GOLF_DIR", "/root/parameter-golf")

TRAIN_SCRIPT_REL = "records/track_non_record_16mb/2026-03-24_11L_XSA_SwiGLU_LoRATTT_1xH100/train_gpt.py"
MIXED_CKPT_REL = "dashboard/checkpoints/submission/final_model.mixed.ptz"
UNQUANT_CKPT_REL = "dashboard/checkpoints/exp17_xsa/final_model.pt"
TOKENIZER_REL = "data/tokenizers/fineweb_1024_bpe.model"
VAL_DATA_REL = "data/datasets/fineweb10B_sp1024"
TRAIN_DATA_REL = "data/datasets/fineweb10B_sp1024"

# Baselines from prior runs
BASELINE_MIXED_BPB = 1.1914       # mixed quant without TTT
BASELINE_TTT_BPB = 1.1573         # mixed quant + LoRA TTT (50K docs)
BASELINE_TRAIN_VAL_LOSS_S1000 = 2.2538  # val_loss at step ~1000 during original training

# Model hyperparameters (must match the saved checkpoint)
MODEL_CONFIG = dict(
    vocab_size=1024,
    num_layers=11,
    model_dim=512,
    num_heads=8,
    num_kv_heads=4,
    mlp_mult=3,
    tie_embeddings=True,
    tied_embed_init_std=0.005,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.5,
    ngram_n=0,
    ngram_buckets=8192,
    ngram_dim=64,
)

SEQ_LEN = 2048

# ---------------------------------------------------------------------------
# Import helpers - load the training module dynamically
# ---------------------------------------------------------------------------

def load_train_module(base_dir: str):
    """Import train_gpt.py as a module so we can reuse its classes and functions."""
    script_path = os.path.join(base_dir, TRAIN_SCRIPT_REL)
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Training script not found at {script_path}")
    spec = importlib.util.spec_from_file_location("train_gpt", script_path)
    mod = importlib.util.module_from_spec(spec)
    # Prevent it from running main() on import
    sys.modules["train_gpt"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("CUDA is required for these experiments")


def load_tokenizer(base_dir: str):
    import sentencepiece as spm
    tok_path = os.path.join(base_dir, TOKENIZER_REL)
    sp = spm.SentencePieceProcessor(model_file=tok_path)
    assert int(sp.vocab_size()) == MODEL_CONFIG["vocab_size"], \
        f"Tokenizer vocab size mismatch: {sp.vocab_size()} vs {MODEL_CONFIG['vocab_size']}"
    return sp


def load_val_tokens(tgpt, base_dir: str):
    """Load the full validation token set."""
    val_pattern = os.path.join(base_dir, VAL_DATA_REL, "fineweb_val_*.bin")
    return tgpt.load_validation_tokens(val_pattern, SEQ_LEN)


def build_luts(tgpt, sp, device):
    """Build byte-counting lookup tables for BPB calculation."""
    return tgpt.build_sentencepiece_luts(sp, MODEL_CONFIG["vocab_size"], device)


def build_model(tgpt, device):
    """Create a fresh model instance with the correct config."""
    model = tgpt.GPT(**MODEL_CONFIG).to(device)
    # Keep weights in float32 (critical for dequantized int5/int6 precision)
    return model


def load_mixed_checkpoint(tgpt, model, base_dir: str):
    """Load the mixed-quantized checkpoint, dequantize, and load into model (float32)."""
    import zstandard
    ckpt_path = os.path.join(base_dir, MIXED_CKPT_REL)
    with open(ckpt_path, "rb") as f:
        blob = f.read()
    raw = zstandard.ZstdDecompressor().decompress(blob)
    quant_dict = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    state = tgpt.dequantize_state_dict_mixed(quant_dict)
    # Ensure float32 -- bfloat16 destroys int5/int6 precision
    for k in state:
        if state[k].is_floating_point():
            state[k] = state[k].float()
    model.load_state_dict(state, strict=True)
    return model


def load_unquantized_checkpoint(model, base_dir: str):
    """Load the unquantized checkpoint (float32)."""
    ckpt_path = os.path.join(base_dir, UNQUANT_CKPT_REL)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Ensure float32
    for k in state:
        if isinstance(state[k], Tensor) and state[k].is_floating_point():
            state[k] = state[k].float()
    model.load_state_dict(state, strict=False)  # strict=False: checkpoint has bigram.* keys
    return model


def eval_val_simple(tgpt, model, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    """Evaluate model on full validation set. Returns (val_loss, val_bpb)."""
    args = tgpt.Hyperparameters()
    # Override for single-GPU eval — use small batches to avoid OOM
    args.val_batch_size = 524_288
    args.train_seq_len = SEQ_LEN
    return tgpt.eval_val(
        args, model, rank=0, world_size=1, device=device,
        grad_accum_steps=32,  # 524288 / (1 * 32) / 2048 = 8 seqs per batch
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )


def ensure_train_data(base_dir: str):
    """Download 1 train shard if not present."""
    import subprocess
    train_dir = os.path.join(base_dir, TRAIN_DATA_REL)
    import glob
    shards = glob.glob(os.path.join(train_dir, "fineweb_train_*.bin"))
    if len(shards) >= 1:
        print(f"  Train data found: {len(shards)} shard(s)")
        return
    print("  Downloading 1 train shard...")
    script = os.path.join(base_dir, "data", "cached_challenge_fineweb.py")
    subprocess.run(
        [sys.executable, script, "--variant", "sp1024", "--train-shards", "1"],
        cwd=base_dir, check=True,
    )
    print("  Train shard downloaded.")


# ---------------------------------------------------------------------------
# Experiment 21: GPTQ-lite - Optimal clip percentile search
# ---------------------------------------------------------------------------

def run_exp21(tgpt, base_dir: str, device):
    """
    Load unquantized checkpoint, search for optimal clip percentile per weight
    matrix row, requantize with best percentiles, measure val_bpb.
    """
    print("\n" + "=" * 80)
    print("EXP 21: GPTQ-lite - Optimal Clip Percentile Search")
    print("=" * 80)

    sp = load_tokenizer(base_dir)
    val_tokens = load_val_tokens(tgpt, base_dir)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_luts(tgpt, sp, device)

    # First, evaluate the current mixed quant baseline
    print("\n--- Baseline: Current mixed quantization ---")
    model_baseline = build_model(tgpt, device)
    load_mixed_checkpoint(tgpt, model_baseline, base_dir)
    bl_loss, bl_bpb = eval_val_simple(
        tgpt, model_baseline, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    print(f"  Baseline mixed quant: val_loss={bl_loss:.4f}, val_bpb={bl_bpb:.4f}")
    del model_baseline
    torch.cuda.empty_cache()

    # Load unquantized checkpoint
    print("\n--- Loading unquantized checkpoint ---")
    model = build_model(tgpt, device)
    load_unquantized_checkpoint(model, base_dir)
    state_dict = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}

    # Evaluate unquantized model
    uq_loss, uq_bpb = eval_val_simple(
        tgpt, model, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    print(f"  Unquantized: val_loss={uq_loss:.4f}, val_bpb={uq_bpb:.4f}")

    # GPTQ-lite: per-row optimal clip percentile search
    CLIP_PERCENTILES = [0.999, 0.9995, 0.9999, 0.99999, 1.0]
    print(f"\n--- Searching clip percentiles: {CLIP_PERCENTILES} ---")

    optimized_state = {}
    total_improved = 0
    total_searched = 0

    for name, t in state_dict.items():
        if not t.is_floating_point() or t.numel() <= 896:
            optimized_state[name] = t
            continue
        if any(p in name for p in tgpt.CONTROL_TENSOR_NAME_PATTERNS) or "smear" in name:
            optimized_state[name] = t
            continue

        t32 = t.float()

        # Determine quantization type based on name (same logic as quantize_state_dict_mixed)
        is_embed = any(p in name for p in tgpt.EMBED_QUANT_PATTERNS)
        is_mlp = any(p in name for p in tgpt.MLP_QUANT_PATTERNS)

        if t32.ndim != 2:
            # Only optimize 2D weight matrices
            optimized_state[name] = t
            continue

        total_searched += 1
        best_q = None
        best_scale = None
        best_mse = float("inf")
        best_pct = None

        for pct in CLIP_PERCENTILES:
            if is_embed:
                # int8 for embeddings
                if pct < 1.0:
                    clip_abs = torch.quantile(t32.abs(), pct, dim=1)
                else:
                    clip_abs = t32.abs().amax(dim=1)
                scale = (clip_abs / 127.0).clamp_min(1e-12).to(torch.float16)
                clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
                q = torch.clamp(torch.round(clipped / scale.float()[:, None]), -128, 127).to(torch.int8)
                recon = q.float() * scale.float()[:, None]
            elif is_mlp:
                # int5 for MLP
                if pct < 1.0:
                    clip_abs = torch.quantile(t32.abs(), pct, dim=1)
                else:
                    clip_abs = t32.abs().amax(dim=1)
                scale = (clip_abs / 15.0).clamp_min(1e-12).to(torch.float16)
                clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
                q = torch.clamp(torch.round(clipped / scale.float()[:, None]), -16, 15).to(torch.int8)
                recon = q.float() * scale.float()[:, None]
            else:
                # int6 for attention
                if pct < 1.0:
                    clip_abs = torch.quantile(t32.abs(), pct, dim=1)
                else:
                    clip_abs = t32.abs().amax(dim=1)
                scale = (clip_abs / 31.0).clamp_min(1e-12).to(torch.float16)
                clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
                q = torch.clamp(torch.round(clipped / scale.float()[:, None]), -32, 31).to(torch.int8)
                recon = q.float() * scale.float()[:, None]

            mse = (t32 - recon).pow(2).mean().item()
            if mse < best_mse:
                best_mse = mse
                best_q = q
                best_scale = scale
                best_pct = pct

        # Store the best quantization as dequantized float32 for model loading
        optimized_state[name] = (best_q.float() * best_scale.float()[:, None]).to(t.dtype)
        if best_pct != 1.0:
            total_improved += 1
        if total_searched <= 5 or total_searched % 10 == 0:
            print(f"  {name}: best_pct={best_pct}, mse={best_mse:.2e}")

    print(f"  Searched {total_searched} weight matrices, {total_improved} improved by clipping")

    # Load optimized state into model and evaluate
    model.load_state_dict(optimized_state, strict=True)
    opt_loss, opt_bpb = eval_val_simple(
        tgpt, model, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    print(f"\n  GPTQ-lite optimized: val_loss={opt_loss:.4f}, val_bpb={opt_bpb:.4f}")

    # Also test: requantize with mixed quant and evaluate the compressed artifact
    print("\n--- Requantizing with mixed quant (to measure compressed BPB) ---")
    import zstandard
    quant_mixed = tgpt.quantize_state_dict_mixed(model.state_dict())
    mixed_buf = io.BytesIO()
    torch.save(quant_mixed, mixed_buf)
    mixed_raw = mixed_buf.getvalue()
    mixed_blob = zstandard.ZstdCompressor(level=22).compress(mixed_raw)
    mixed_size = len(mixed_blob)
    print(f"  Requantized mixed size: {mixed_size} bytes ({mixed_size / 1024 / 1024:.2f} MB)")

    # Reload requantized model and eval
    mixed_state = torch.load(
        io.BytesIO(zstandard.ZstdDecompressor().decompress(mixed_blob)),
        map_location="cpu", weights_only=False,
    )
    deq_state = tgpt.dequantize_state_dict_mixed(mixed_state)
    for k in deq_state:
        if deq_state[k].is_floating_point():
            deq_state[k] = deq_state[k].float()
    model.load_state_dict(deq_state, strict=True)
    rq_loss, rq_bpb = eval_val_simple(
        tgpt, model, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    print(f"  Requantized mixed: val_loss={rq_loss:.4f}, val_bpb={rq_bpb:.4f}")

    del model
    torch.cuda.empty_cache()

    result = {
        "unquantized_bpb": uq_bpb,
        "baseline_mixed_bpb": bl_bpb,
        "gptq_lite_bpb": opt_bpb,
        "requantized_mixed_bpb": rq_bpb,
        "requantized_size_mb": mixed_size / 1024 / 1024,
        "delta_vs_baseline": rq_bpb - bl_bpb,
    }
    print(f"\n  RESULT: GPTQ-lite requantized BPB={rq_bpb:.4f} vs baseline mixed BPB={bl_bpb:.4f} "
          f"(delta={rq_bpb - bl_bpb:+.4f})")
    return result


# ---------------------------------------------------------------------------
# Experiment 22: LeakyReLU^2 activation
# ---------------------------------------------------------------------------

def run_exp22(tgpt, base_dir: str, device):
    """
    Replace SwiGLU activation with LeakyReLU(0.5)^2 and train for 300s.
    Compare val_loss at step ~500 with baseline.
    """
    print("\n" + "=" * 80)
    print("EXP 22: LeakyReLU^2 Activation (replacing SwiGLU)")
    print("=" * 80)

    ensure_train_data(base_dir)

    sp = load_tokenizer(base_dir)
    val_tokens = load_val_tokens(tgpt, base_dir)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_luts(tgpt, sp, device)

    # Build model with modified MLP
    class LeakyReLU2MLP(nn.Module):
        """LeakyReLU(0.5)^2 gated MLP, replacing SwiGLU."""
        def __init__(self, dim: int, mlp_mult: int):
            super().__init__()
            hidden = int(2 * mlp_mult * dim / 3)
            hidden = ((hidden + 63) // 64) * 64
            self.gate = tgpt.CastedLinear(dim, hidden, bias=False)
            self.up = tgpt.CastedLinear(dim, hidden, bias=False)
            self.proj = tgpt.CastedLinear(hidden, dim, bias=False)
            self.proj._zero_init = True

        def forward(self, x: Tensor) -> Tensor:
            return self.proj(F.leaky_relu(self.gate(x), 0.5).square() * self.up(x))

    # Create model and replace MLP modules
    model = build_model(tgpt, device)
    for block in model.blocks:
        old_mlp = block.mlp
        new_mlp = LeakyReLU2MLP(MODEL_CONFIG["model_dim"], MODEL_CONFIG["mlp_mult"]).to(device)
        block.mlp = new_mlp
    model._init_weights()

    # Training setup - simplified single GPU training loop
    import glob as glob_mod
    train_pattern = os.path.join(base_dir, TRAIN_DATA_REL, "fineweb_train_*.bin")
    train_stream = tgpt.TokenStream(train_pattern)

    # Hyperparameters for short training
    train_batch_tokens = 16384  # 8 seqs of 2048, fits in 24GB RTX 4090
    grad_accum_steps = 1
    max_wallclock_s = 300.0
    warmdown_iters = 100  # ~30% of expected ~300 steps
    val_every = 100

    # Optimizer setup (simplified: Adam for everything)
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8)

    model.train()
    t_start = time.perf_counter()
    step = 0
    results_log = []

    print(f"  Training with LeakyReLU^2 MLP for {max_wallclock_s}s, batch_tokens={train_batch_tokens}")

    while True:
        elapsed_s = time.perf_counter() - t_start
        if elapsed_s >= max_wallclock_s:
            break

        # LR warmdown
        estimated_total_steps = max(int(max_wallclock_s / max(elapsed_s / max(step, 1), 0.01)), step + 1)
        warmdown_start = max(estimated_total_steps - warmdown_iters, 0)
        if step >= warmdown_start:
            remaining = max(estimated_total_steps - step, 0)
            lr_scale = remaining / max(warmdown_iters, 1)
        else:
            lr_scale = 1.0
        for pg in optimizer.param_groups:
            pg["lr"] = lr * lr_scale

        # Get batch
        local_tokens = train_batch_tokens + 1
        chunk = train_stream.take(local_tokens)
        local = chunk.to(dtype=torch.int64, device=device)
        x = local[:-1].reshape(-1, SEQ_LEN)
        y = local[1:].reshape(-1, SEQ_LEN)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1

        # Validate periodically
        if step % val_every == 0 or step == 1:
            val_loss, val_bpb = eval_val_simple(
                tgpt, model, device, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            elapsed_ms = 1000.0 * (time.perf_counter() - t_start)
            results_log.append((step, val_loss, val_bpb, elapsed_ms))
            print(f"  step:{step} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                  f"train_loss:{loss.item():.4f} time:{elapsed_ms:.0f}ms")
            model.train()

    # Final eval
    val_loss, val_bpb = eval_val_simple(
        tgpt, model, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    elapsed_ms = 1000.0 * (time.perf_counter() - t_start)
    results_log.append((step, val_loss, val_bpb, elapsed_ms))
    print(f"  FINAL step:{step} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    del model, optimizer
    torch.cuda.empty_cache()

    # Compare with baseline
    # Baseline trained with SwiGLU had val_loss=2.2538 at step 1000
    # We compare at similar step counts
    result = {
        "final_step": step,
        "final_val_loss": val_loss,
        "final_val_bpb": val_bpb,
        "baseline_val_loss_s1000": BASELINE_TRAIN_VAL_LOSS_S1000,
        "log": results_log,
    }
    print(f"\n  RESULT: LeakyReLU^2 final val_loss={val_loss:.4f} at step {step}")
    print(f"  Baseline SwiGLU val_loss={BASELINE_TRAIN_VAL_LOSS_S1000:.4f} at step 1000")
    return result


# ---------------------------------------------------------------------------
# Experiment 23: Sliding Window TTT (full-weight SGD)
# ---------------------------------------------------------------------------

def run_exp23(tgpt, base_dir: str, device):
    """
    Implement the top submission's TTT recipe on our mixed-quant checkpoint:
    - Split val tokens into 32K-token chunks
    - Score each chunk with sliding window (stride=64, seq_len=2048)
    - Train ALL weights with SGD(lr=0.002, momentum=0.9) for 3 epochs
    - Cosine LR decay, grad clip 1.0
    - Score-first: chunk N scored before training on chunk N
    - Run on first 2000 docs worth of tokens
    """
    print("\n" + "=" * 80)
    print("EXP 23: Sliding Window TTT (Full-Weight SGD, Top Submission Recipe)")
    print("=" * 80)

    sp = load_tokenizer(base_dir)
    val_tokens = load_val_tokens(tgpt, base_dir)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_luts(tgpt, sp, device)

    # Load mixed-quant checkpoint
    model = build_model(tgpt, device)
    load_mixed_checkpoint(tgpt, model, base_dir)

    # TTT parameters
    TTT_CHUNK_SIZE = 32768  # 32K tokens per chunk
    TTT_STRIDE = 64
    TTT_EPOCHS = 3
    TTT_LR = 0.002
    TTT_MOMENTUM = 0.9
    TTT_GRAD_CLIP = 1.0
    TTT_MAX_DOCS = 2000

    # Find document boundaries (BOS=1)
    bos_positions = (val_tokens == 1).nonzero(as_tuple=True)[0].cpu().numpy()
    n_docs = min(len(bos_positions), TTT_MAX_DOCS)

    # Determine token range for first N docs
    if n_docs < len(bos_positions):
        token_end = int(bos_positions[n_docs])
    else:
        token_end = val_tokens.numel()
    token_start = int(bos_positions[0])
    total_tokens = token_end - token_start
    print(f"  Processing {n_docs} docs, {total_tokens} tokens")
    print(f"  Chunk size: {TTT_CHUNK_SIZE}, stride: {TTT_STRIDE}, epochs: {TTT_EPOCHS}")
    print(f"  SGD(lr={TTT_LR}, momentum={TTT_MOMENTUM}), grad_clip={TTT_GRAD_CLIP}")

    # Save initial state for reference
    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Split into 32K chunks
    doc_tokens = val_tokens[token_start:token_end]
    n_chunks = (doc_tokens.numel() + TTT_CHUNK_SIZE - 1) // TTT_CHUNK_SIZE
    print(f"  Number of chunks: {n_chunks}")

    total_nll = 0.0
    total_byte_count = 0.0
    total_token_count = 0

    t_start = time.perf_counter()

    for ci in range(n_chunks):
        chunk_start = ci * TTT_CHUNK_SIZE
        chunk_end = min((ci + 1) * TTT_CHUNK_SIZE, doc_tokens.numel())
        chunk = doc_tokens[chunk_start:chunk_end].to(device=device, dtype=torch.int64)

        if chunk.numel() < 2:
            continue

        # --- SCORE phase: sliding window scoring ---
        model.eval()
        chunk_nll = 0.0
        chunk_tokens = 0
        chunk_bytes = 0.0

        with torch.no_grad():
            # Score with sliding windows (stride=64, window=2048)
            pred_len = chunk.numel() - 1
            scored = torch.zeros(pred_len, dtype=torch.bool, device=device)
            nll_per_pos = torch.zeros(pred_len, dtype=torch.float64, device=device)

            pos = 0
            while pos < pred_len:
                win_end = min(pos + SEQ_LEN, chunk.numel())
                win_start = max(win_end - SEQ_LEN, 0)
                x = chunk[win_start:win_end - 1].unsqueeze(0)
                y = chunk[win_start + 1:win_end].unsqueeze(0)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    # Get per-token loss
                    xe = model.tok_emb(x)
                    if model.ngram is not None:
                        xe = xe + model.ngram(x)
                    xe = F.rms_norm(xe, (xe.size(-1),))
                    xe = model.smear(xe)
                    x0 = xe
                    skips = []
                    for ii in range(model.num_encoder_layers):
                        xe = model.blocks[ii](xe, x0)
                        skips.append(xe)
                    for ii in range(model.num_decoder_layers):
                        if skips:
                            xe = xe + model.skip_weights[ii].to(dtype=xe.dtype)[None, None, :] * skips.pop()
                        xe = model.blocks[model.num_encoder_layers + ii](xe, x0)
                    xe = model.final_norm(xe)
                    if model.tie_embeddings:
                        logits = F.linear(xe, model.tok_emb.weight)
                    else:
                        logits = model.lm_head(xe)
                    logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)

                log_probs = F.log_softmax(logits[0].float(), dim=-1)
                targets = y[0]

                # Only score tokens that haven't been scored yet
                # Map window positions to chunk positions
                for t in range(targets.size(0)):
                    chunk_pos = win_start + t  # position of the target in chunk (0-indexed pred)
                    if chunk_pos < pred_len and not scored[chunk_pos]:
                        nll_per_pos[chunk_pos] = -log_probs[t, targets[t]].to(torch.float64)
                        scored[chunk_pos] = True

                pos += TTT_STRIDE
                if win_end >= chunk.numel():
                    break

            # Compute metrics for scored positions
            valid_mask = scored
            chunk_nll = nll_per_pos[valid_mask].sum().item()
            chunk_tokens = int(valid_mask.sum().item())

            # Byte counting
            if chunk_tokens > 0:
                # prev_ids and tgt_ids for the chunk
                prev_ids = chunk[:-1][valid_mask]
                tgt_ids = chunk[1:][valid_mask]
                tb = base_bytes_lut[tgt_ids].to(torch.float64)
                tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
                chunk_bytes = tb.sum().item()

        total_nll += chunk_nll
        total_token_count += chunk_tokens
        total_byte_count += chunk_bytes

        # --- TRAIN phase: fine-tune ALL weights on this chunk ---
        if ci < n_chunks - 1:  # Don't train on the last chunk
            model.train()
            for p in model.parameters():
                p.requires_grad = True

            optimizer = torch.optim.SGD(model.parameters(), lr=TTT_LR, momentum=TTT_MOMENTUM)

            for epoch in range(TTT_EPOCHS):
                # Cosine LR decay across epochs
                epoch_lr = TTT_LR * 0.5 * (1 + math.cos(math.pi * epoch / TTT_EPOCHS))
                for pg in optimizer.param_groups:
                    pg["lr"] = epoch_lr

                # Train on chunk with sliding windows
                train_pos = 0
                while train_pos < pred_len:
                    win_end = min(train_pos + SEQ_LEN, chunk.numel())
                    win_start = max(win_end - SEQ_LEN, 0)
                    x = chunk[win_start:win_end - 1].unsqueeze(0)
                    y = chunk[win_start + 1:win_end].unsqueeze(0)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        loss = model(x, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), TTT_GRAD_CLIP)
                    optimizer.step()

                    train_pos += TTT_STRIDE
                    if win_end >= chunk.numel():
                        break

            del optimizer

        # Progress
        if (ci + 1) % 5 == 0 or ci == 0:
            elapsed = time.perf_counter() - t_start
            running_bpb = (total_nll / math.log(2.0)) / max(total_byte_count, 1.0)
            print(f"  Chunk {ci + 1}/{n_chunks}: running_bpb={running_bpb:.4f}, "
                  f"tokens={total_token_count}, elapsed={elapsed:.0f}s")

    elapsed = time.perf_counter() - t_start
    final_bpb = (total_nll / math.log(2.0)) / max(total_byte_count, 1.0)
    final_loss = total_nll / max(total_token_count, 1)

    del model
    torch.cuda.empty_cache()

    result = {
        "final_bpb": final_bpb,
        "final_loss": final_loss,
        "n_docs": n_docs,
        "total_tokens": total_token_count,
        "elapsed_s": elapsed,
        "baseline_ttt_bpb_50k": BASELINE_TTT_BPB,
        "baseline_mixed_bpb": BASELINE_MIXED_BPB,
    }
    print(f"\n  RESULT: Sliding Window TTT BPB={final_bpb:.4f} on {n_docs} docs "
          f"({total_token_count} tokens, {elapsed:.0f}s)")
    print(f"  Baseline LoRA TTT: {BASELINE_TTT_BPB:.4f} (50K docs)")
    print(f"  Baseline no-TTT:   {BASELINE_MIXED_BPB:.4f}")
    return result


# ---------------------------------------------------------------------------
# Experiment 24: EMA alongside SWA during training
# ---------------------------------------------------------------------------

def run_exp24(tgpt, base_dir: str, device):
    """
    Train from scratch for 300s with EMA(decay=0.997) applied every step,
    alongside SWA. Compare final val_loss with baseline.
    """
    print("\n" + "=" * 80)
    print("EXP 24: EMA (decay=0.997) During Training")
    print("=" * 80)

    ensure_train_data(base_dir)

    sp = load_tokenizer(base_dir)
    val_tokens = load_val_tokens(tgpt, base_dir)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_luts(tgpt, sp, device)

    import glob as glob_mod
    train_pattern = os.path.join(base_dir, TRAIN_DATA_REL, "fineweb_train_*.bin")
    train_stream = tgpt.TokenStream(train_pattern)

    # Build fresh model
    model = build_model(tgpt, device)
    model._init_weights()

    # Training hyperparameters
    train_batch_tokens = 16384  # 8 seqs of 2048, fits in 24GB RTX 4090
    max_wallclock_s = 300.0
    warmdown_iters = 100
    val_every = 100
    lr = 0.001
    ema_decay = 0.997

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8)

    # Initialize EMA shadow weights
    ema_state = {k: v.detach().cpu().clone().float() for k, v in model.state_dict().items()}

    # Also track SWA
    swa_state = None
    swa_count = 0

    model.train()
    t_start = time.perf_counter()
    step = 0
    results_log = []

    print(f"  Training with EMA(decay={ema_decay}) for {max_wallclock_s}s, batch_tokens={train_batch_tokens}")

    while True:
        elapsed_s = time.perf_counter() - t_start
        if elapsed_s >= max_wallclock_s:
            break

        # LR warmdown
        estimated_total_steps = max(int(max_wallclock_s / max(elapsed_s / max(step, 1), 0.01)), step + 1)
        warmdown_start = max(estimated_total_steps - warmdown_iters, 0)
        if step >= warmdown_start:
            remaining = max(estimated_total_steps - step, 0)
            lr_scale = remaining / max(warmdown_iters, 1)
        else:
            lr_scale = 1.0
        for pg in optimizer.param_groups:
            pg["lr"] = lr * lr_scale

        # Get batch
        local_tokens = train_batch_tokens + 1
        chunk = train_stream.take(local_tokens)
        local = chunk.to(dtype=torch.int64, device=device)
        x = local[:-1].reshape(-1, SEQ_LEN)
        y = local[1:].reshape(-1, SEQ_LEN)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1

        # Update EMA every step
        with torch.no_grad():
            for k, v in model.state_dict().items():
                ema_state[k].mul_(ema_decay).add_(v.detach().cpu().float(), alpha=1 - ema_decay)

        # SWA: collect every 200 steps during warmdown
        if step >= warmdown_start and step % 200 == 0 and step > 100:
            if swa_state is None:
                swa_state = {k: v.detach().cpu().clone().float() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                for k, v in model.state_dict().items():
                    swa_state[k] += v.detach().cpu().float()
                swa_count += 1

        # Validate periodically
        if step % val_every == 0 or step == 1:
            val_loss, val_bpb = eval_val_simple(
                tgpt, model, device, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            elapsed_ms = 1000.0 * (time.perf_counter() - t_start)
            results_log.append(("online", step, val_loss, val_bpb, elapsed_ms))
            print(f"  step:{step} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                  f"train_loss:{loss.item():.4f} time:{elapsed_ms:.0f}ms")
            model.train()

    # Evaluate online weights (final)
    val_loss_online, val_bpb_online = eval_val_simple(
        tgpt, model, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    print(f"\n  Online weights: val_loss={val_loss_online:.4f}, val_bpb={val_bpb_online:.4f}")

    # Evaluate EMA weights
    saved_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ema_load = {k: v.to(dtype=saved_state[k].dtype) for k, v in ema_state.items()}
    model.load_state_dict(ema_load, strict=True)
    val_loss_ema, val_bpb_ema = eval_val_simple(
        tgpt, model, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    print(f"  EMA weights:    val_loss={val_loss_ema:.4f}, val_bpb={val_bpb_ema:.4f}")

    # Evaluate SWA weights
    if swa_state is not None and swa_count > 1:
        swa_load = {}
        for k, v in swa_state.items():
            avg = (v / swa_count).to(dtype=saved_state[k].dtype)
            swa_load[k] = avg
        model.load_state_dict(swa_load, strict=True)
        val_loss_swa, val_bpb_swa = eval_val_simple(
            tgpt, model, device, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        print(f"  SWA weights:    val_loss={val_loss_swa:.4f}, val_bpb={val_bpb_swa:.4f} ({swa_count} checkpoints)")
    else:
        val_loss_swa, val_bpb_swa = float("nan"), float("nan")
        print(f"  SWA: not enough checkpoints collected ({swa_count})")

    del model, optimizer
    torch.cuda.empty_cache()

    result = {
        "final_step": step,
        "val_loss_online": val_loss_online,
        "val_loss_ema": val_loss_ema,
        "val_loss_swa": val_loss_swa,
        "val_bpb_online": val_bpb_online,
        "val_bpb_ema": val_bpb_ema,
        "val_bpb_swa": val_bpb_swa,
        "baseline_val_loss_s1000": BASELINE_TRAIN_VAL_LOSS_S1000,
        "log": results_log,
    }
    best_loss = min(val_loss_online, val_loss_ema, val_loss_swa if not math.isnan(val_loss_swa) else float("inf"))
    print(f"\n  RESULT: Best val_loss={best_loss:.4f} at step {step}")
    print(f"  Baseline SwiGLU val_loss={BASELINE_TRAIN_VAL_LOSS_S1000:.4f} at step 1000")
    return result


# ---------------------------------------------------------------------------
# Experiment 25: XSA on all layers
# ---------------------------------------------------------------------------

def run_exp25(tgpt, base_dir: str, device):
    """
    Change use_xsa to True for all 11 layers (instead of last 4).
    Train for 300s and compare val_loss.
    """
    print("\n" + "=" * 80)
    print("EXP 25: XSA on All Layers (11/11 instead of 4/11)")
    print("=" * 80)

    ensure_train_data(base_dir)

    sp = load_tokenizer(base_dir)
    val_tokens = load_val_tokens(tgpt, base_dir)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_luts(tgpt, sp, device)

    # Build model with XSA on ALL layers
    config = dict(MODEL_CONFIG)
    model = tgpt.GPT(**config).to(device)

    # Override: set use_xsa=True for all blocks
    for block in model.blocks:
        block.attn.use_xsa = True
    model._init_weights()
    print(f"  XSA enabled on all {len(model.blocks)} layers")

    import glob as glob_mod
    train_pattern = os.path.join(base_dir, TRAIN_DATA_REL, "fineweb_train_*.bin")
    train_stream = tgpt.TokenStream(train_pattern)

    # Training hyperparameters
    train_batch_tokens = 16384  # 8 seqs of 2048, fits in 24GB RTX 4090
    max_wallclock_s = 300.0
    warmdown_iters = 100
    val_every = 100
    lr = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8)

    model.train()
    t_start = time.perf_counter()
    step = 0
    results_log = []

    print(f"  Training XSA-all for {max_wallclock_s}s, batch_tokens={train_batch_tokens}")

    while True:
        elapsed_s = time.perf_counter() - t_start
        if elapsed_s >= max_wallclock_s:
            break

        # LR warmdown
        estimated_total_steps = max(int(max_wallclock_s / max(elapsed_s / max(step, 1), 0.01)), step + 1)
        warmdown_start = max(estimated_total_steps - warmdown_iters, 0)
        if step >= warmdown_start:
            remaining = max(estimated_total_steps - step, 0)
            lr_scale = remaining / max(warmdown_iters, 1)
        else:
            lr_scale = 1.0
        for pg in optimizer.param_groups:
            pg["lr"] = lr * lr_scale

        # Get batch
        local_tokens = train_batch_tokens + 1
        chunk = train_stream.take(local_tokens)
        local = chunk.to(dtype=torch.int64, device=device)
        x = local[:-1].reshape(-1, SEQ_LEN)
        y = local[1:].reshape(-1, SEQ_LEN)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1

        # Validate periodically
        if step % val_every == 0 or step == 1:
            val_loss, val_bpb = eval_val_simple(
                tgpt, model, device, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            elapsed_ms = 1000.0 * (time.perf_counter() - t_start)
            results_log.append((step, val_loss, val_bpb, elapsed_ms))
            print(f"  step:{step} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                  f"train_loss:{loss.item():.4f} time:{elapsed_ms:.0f}ms")
            model.train()

    # Final eval
    val_loss, val_bpb = eval_val_simple(
        tgpt, model, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    elapsed_ms = 1000.0 * (time.perf_counter() - t_start)
    results_log.append((step, val_loss, val_bpb, elapsed_ms))
    print(f"  FINAL step:{step} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    del model, optimizer
    torch.cuda.empty_cache()

    # Also train a baseline SwiGLU model (XSA on last 4 only) for fair comparison
    print("\n  --- Training baseline (XSA last-4 only) for same duration ---")
    model_bl = build_model(tgpt, device)
    model_bl._init_weights()
    train_stream_bl = tgpt.TokenStream(train_pattern)
    optimizer_bl = torch.optim.Adam(model_bl.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8)

    model_bl.train()
    t_start_bl = time.perf_counter()
    step_bl = 0

    while True:
        elapsed_s = time.perf_counter() - t_start_bl
        if elapsed_s >= max_wallclock_s:
            break

        estimated_total_steps = max(int(max_wallclock_s / max(elapsed_s / max(step_bl, 1), 0.01)), step_bl + 1)
        warmdown_start = max(estimated_total_steps - warmdown_iters, 0)
        if step_bl >= warmdown_start:
            remaining = max(estimated_total_steps - step_bl, 0)
            lr_scale = remaining / max(warmdown_iters, 1)
        else:
            lr_scale = 1.0
        for pg in optimizer_bl.param_groups:
            pg["lr"] = lr * lr_scale

        local_tokens = train_batch_tokens + 1
        chunk = train_stream_bl.take(local_tokens)
        local = chunk.to(dtype=torch.int64, device=device)
        x = local[:-1].reshape(-1, SEQ_LEN)
        y = local[1:].reshape(-1, SEQ_LEN)

        optimizer_bl.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss_bl = model_bl(x, y)
        loss_bl.backward()
        torch.nn.utils.clip_grad_norm_(model_bl.parameters(), 1.0)
        optimizer_bl.step()
        step_bl += 1

    bl_val_loss, bl_val_bpb = eval_val_simple(
        tgpt, model_bl, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    print(f"  Baseline (XSA last-4): val_loss={bl_val_loss:.4f}, val_bpb={bl_val_bpb:.4f} at step {step_bl}")

    del model_bl, optimizer_bl
    torch.cuda.empty_cache()

    result = {
        "xsa_all_final_step": step,
        "xsa_all_val_loss": val_loss,
        "xsa_all_val_bpb": val_bpb,
        "baseline_final_step": step_bl,
        "baseline_val_loss": bl_val_loss,
        "baseline_val_bpb": bl_val_bpb,
        "delta_val_loss": val_loss - bl_val_loss,
        "log": results_log,
    }
    print(f"\n  RESULT: XSA-all val_loss={val_loss:.4f} vs baseline val_loss={bl_val_loss:.4f} "
          f"(delta={val_loss - bl_val_loss:+.4f})")
    return result


# ---------------------------------------------------------------------------
# Main: run all experiments and print summary
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run parameter-golf experiments 21-25")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR, help="Path to parameter-golf repo root")
    parser.add_argument("--experiments", default="21,22,23,24,25",
                        help="Comma-separated list of experiments to run (default: all)")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    exps_to_run = [int(x.strip()) for x in args.experiments.split(",")]

    print(f"Parameter Golf Experiment Runner")
    print(f"Base dir: {base_dir}")
    print(f"Experiments: {exps_to_run}")
    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU (will fail)'}")
    print()

    # Load training module
    tgpt = load_train_module(base_dir)

    device = get_device()
    results = {}

    # Run each experiment
    if 21 in exps_to_run:
        try:
            results[21] = run_exp21(tgpt, base_dir, device)
        except Exception as e:
            print(f"\n  EXP 21 FAILED: {e}")
            import traceback; traceback.print_exc()
            results[21] = {"error": str(e)}

    if 22 in exps_to_run:
        try:
            results[22] = run_exp22(tgpt, base_dir, device)
        except Exception as e:
            print(f"\n  EXP 22 FAILED: {e}")
            import traceback; traceback.print_exc()
            results[22] = {"error": str(e)}

    if 23 in exps_to_run:
        try:
            results[23] = run_exp23(tgpt, base_dir, device)
        except Exception as e:
            print(f"\n  EXP 23 FAILED: {e}")
            import traceback; traceback.print_exc()
            results[23] = {"error": str(e)}

    if 24 in exps_to_run:
        try:
            results[24] = run_exp24(tgpt, base_dir, device)
        except Exception as e:
            print(f"\n  EXP 24 FAILED: {e}")
            import traceback; traceback.print_exc()
            results[24] = {"error": str(e)}

    if 25 in exps_to_run:
        try:
            results[25] = run_exp25(tgpt, base_dir, device)
        except Exception as e:
            print(f"\n  EXP 25 FAILED: {e}")
            import traceback; traceback.print_exc()
            results[25] = {"error": str(e)}

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"{'Exp':>5} | {'Name':<35} | {'Key Metric':<20} | {'Baseline':<12} | {'Delta':<10} | {'Worth Full Run?'}")
    print("-" * 110)

    def fmt(v, prec=4):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        return f"{v:.{prec}f}"

    for exp_id in sorted(results.keys()):
        r = results[exp_id]
        if "error" in r:
            print(f"  {exp_id:>3} | {'FAILED':<35} | {r['error'][:40]}")
            continue

        if exp_id == 21:
            name = "GPTQ-lite clip search"
            metric = f"BPB={fmt(r.get('requantized_mixed_bpb'))}"
            baseline = f"BPB={fmt(r.get('baseline_mixed_bpb'))}"
            delta = fmt(r.get('delta_vs_baseline'), 4)
            worth = "YES" if r.get("delta_vs_baseline", 0) < -0.001 else "MAYBE" if r.get("delta_vs_baseline", 0) < 0 else "NO"

        elif exp_id == 22:
            name = "LeakyReLU^2 (vs SwiGLU)"
            metric = f"loss={fmt(r.get('final_val_loss'))}"
            baseline = f"loss={fmt(r.get('baseline_val_loss_s1000'))}"
            delta_v = r.get("final_val_loss", 0) - r.get("baseline_val_loss_s1000", 0)
            delta = fmt(delta_v, 4)
            worth = "YES" if delta_v < -0.01 else "MAYBE" if delta_v < 0 else "NO"

        elif exp_id == 23:
            name = "Sliding Window TTT (SGD)"
            metric = f"BPB={fmt(r.get('final_bpb'))}"
            baseline = f"BPB={fmt(r.get('baseline_ttt_bpb_50k'))}"
            delta_v = r.get("final_bpb", 0) - r.get("baseline_ttt_bpb_50k", 0)
            delta = fmt(delta_v, 4)
            # Note: comparing 2K docs vs 50K docs baseline
            worth = "YES" if r.get("final_bpb", 99) < r.get("baseline_mixed_bpb", 0) else "MAYBE"

        elif exp_id == 24:
            name = "EMA (decay=0.997)"
            best = min(r.get("val_loss_online", 99), r.get("val_loss_ema", 99),
                       r.get("val_loss_swa", 99) if not math.isnan(r.get("val_loss_swa", float("nan"))) else 99)
            metric = f"loss={fmt(best)}"
            baseline = f"loss={fmt(r.get('baseline_val_loss_s1000'))}"
            ema_vs_online = r.get("val_loss_ema", 99) - r.get("val_loss_online", 99)
            delta = fmt(ema_vs_online, 4) + " (EMA-online)"
            worth = "YES" if ema_vs_online < -0.01 else "MAYBE" if ema_vs_online < 0 else "NO"

        elif exp_id == 25:
            name = "XSA all layers"
            metric = f"loss={fmt(r.get('xsa_all_val_loss'))}"
            baseline = f"loss={fmt(r.get('baseline_val_loss'))}"
            delta = fmt(r.get("delta_val_loss"), 4)
            worth = "YES" if r.get("delta_val_loss", 0) < -0.01 else "MAYBE" if r.get("delta_val_loss", 0) < 0 else "NO"

        else:
            continue

        print(f"  {exp_id:>3} | {name:<35} | {metric:<20} | {baseline:<12} | {delta:<10} | {worth}")

    print("-" * 110)
    print("\nNotes:")
    print("  - Exp 21: Compares requantized BPB (no TTT). Lower is better.")
    print("  - Exp 22: Compares val_loss at comparable step count. Different step counts noted.")
    print("  - Exp 23: Compares BPB on 2K docs (vs baseline 50K docs). Extrapolation needed.")
    print("  - Exp 24: EMA vs online weights at same training budget. Negative delta = EMA wins.")
    print("  - Exp 25: XSA-all vs XSA-last-4, same training config. Negative delta = XSA-all wins.")
    print("  - 'Worth Full Run?' is YES (clear win), MAYBE (marginal/needs more data), NO (no benefit).")


if __name__ == "__main__":
    main()
