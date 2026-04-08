"""
Exp 53: QK-Gain=4.0 Analysis Script

Compares exp40-D (QK-Gain=1.5) vs exp53 (QK-Gain=4.0) checkpoints to understand
WHY QK-Gain=4.0 helps. Tests three hypotheses:

1. ATTENTION SHARPNESS: QK=4.0 produces sharper attention (lower entropy)
2. HEAD SPECIALIZATION: QK=4.0 heads are more diverse (different entropy profiles)
3. QUANTIZATION FRIENDLINESS: QK=4.0 weights have better structure for int5/int6

Usage:
  python exp53_qkgain_analysis.py
"""

import torch, os, sys, glob, math, json, random
import torch.nn.functional as F
import numpy as np
from pathlib import Path

BASE_DIR = os.environ.get("BASE_DIR", "/runpod-volume/parameter-golf")
RECORDS_DIR = os.path.join(BASE_DIR, "records/track_non_record_16mb/2026-03-24_11L_XSA_SwiGLU_LoRATTT_1xH100")
EXP40D_PATH = os.environ.get("EXP40D_PATH", os.path.join(BASE_DIR, "exp40d_model.pt"))
EXP53_PATH = os.environ.get("EXP53_PATH", os.path.join(BASE_DIR, "final_model.pt"))
SEED = 42
N_EVAL_SEQS = 32

os.chdir(BASE_DIR)

# Set model config env vars BEFORE loading model code
for k, v in {"XSA_LAST_N": "7", "MLP_MULT": "3", "NUM_LAYERS": "7", "MODEL_DIM": "624",
             "VOCAB_SIZE": "1024", "TRAIN_SEQ_LEN": "2048",
             "LAYER_SCHEDULE": "0,1,2,3,4,3,4,3,4,5,6,5,6",
             "QK_GAIN_INIT": "1.5"}.items():
    os.environ[k] = v

import importlib.util
_spec = importlib.util.spec_from_file_location("tgp", os.path.join(RECORDS_DIR, "train_gpt.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

GPT = _mod.GPT
load_data_shard = _mod.load_data_shard
Hyperparameters = _mod.Hyperparameters
apply_rotary_emb = _mod.apply_rotary_emb
MLP_QUANT_PATTERNS = _mod.MLP_QUANT_PATTERNS
EMBED_QUANT_PATTERNS = _mod.EMBED_QUANT_PATTERNS
CONTROL_TENSOR_NAME_PATTERNS = _mod.CONTROL_TENSOR_NAME_PATTERNS

device = torch.device("cuda")
LAYER_SCHEDULE = [0, 1, 2, 3, 4, 3, 4, 3, 4, 5, 6, 5, 6]


def load_model(path, qk_gain_init):
    """Load a model checkpoint with specified QK gain."""
    os.environ["QK_GAIN_INIT"] = str(qk_gain_init)
    os.environ["MODEL_DIM"] = "624"
    os.environ["NUM_LAYERS"] = "7"
    os.environ["VOCAB_SIZE"] = "1024"
    os.environ["MLP_MULT"] = "3"
    os.environ["XSA_LAST_N"] = "7"
    os.environ["LAYER_SCHEDULE"] = "0,1,2,3,4,3,4,3,4,5,6,5,6"
    args = Hyperparameters()
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=qk_gain_init, xsa_last_n=args.xsa_last_n,
        layer_schedule=args.layer_schedule if args.layer_schedule else None,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, args


def compute_attention_entropy(model, sequences):
    """
    Compute attention entropy at each layer position by hooking into attention.
    Uses manual Q/K computation to extract attention weights (SDPA doesn't expose them).

    Returns dict: (block_idx, pos_idx) -> {entropy_per_head, max_attn_per_head, mean_entropy}
    """
    all_entropies = {}   # (block_idx, pos_idx) -> list of [num_heads] arrays
    all_max_attn = {}

    num_encoder = model.num_encoder_layers
    num_decoder = model.num_decoder_layers

    for seq in sequences:
        x_input = seq.unsqueeze(0).to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Match GPT.forward exactly
            xe = model.tok_emb(x_input)
            if model.ngram is not None:
                xe = xe + model.ngram(x_input)
            xe = F.rms_norm(xe, (xe.size(-1),))
            xe = model.smear(xe)
            x0 = xe.clone()
            skips = []

            for i, block_idx in enumerate(LAYER_SCHEDULE):
                block = model.blocks[block_idx]
                attn = block.attn

                # Apply resid_mix (same as Block.forward)
                mix = block.resid_mix.to(dtype=xe.dtype)
                x_mixed = mix[0][None, None, :] * xe + mix[1][None, None, :] * x0

                # Pre-attention norm
                x_normed = block.attn_norm(x_mixed)

                # Compute Q, K manually (matching CausalSelfAttention.forward exactly)
                bsz, seqlen, dim = x_normed.shape
                q = attn.c_q(x_normed).reshape(bsz, seqlen, attn.num_heads, attn.head_dim).transpose(1, 2)
                k = attn.c_k(x_normed).reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim).transpose(1, 2)

                # QK norm
                q = F.rms_norm(q, (q.size(-1),))
                k = F.rms_norm(k, (k.size(-1),))

                # RoPE
                cos, sin = attn.rotary(seqlen, x_normed.device, q.dtype)
                q = apply_rotary_emb(q, cos, sin)
                k = apply_rotary_emb(k, cos, sin)

                # QK gain
                q = q * attn.q_gain.to(dtype=q.dtype)[None, :, None, None]

                # Expand KV heads for GQA
                if attn.num_kv_heads != attn.num_heads:
                    reps = attn.num_heads // attn.num_kv_heads
                    k = k.repeat_interleave(reps, dim=1)

                # Compute raw attention scores
                scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(attn.head_dim)

                # Causal mask
                causal_mask = torch.triu(torch.ones(seqlen, seqlen, device=device, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(causal_mask[None, None, :, :], float('-inf'))

                # Softmax -> attention weights
                attn_weights = F.softmax(scores, dim=-1)  # [1, num_heads, T, T]

                # Entropy per head: -sum(p * log(p)), averaged over query positions
                log_attn = torch.log(attn_weights + 1e-10)
                entropy = -(attn_weights * log_attn).sum(dim=-1)  # [1, num_heads, T]
                mean_entropy = entropy.mean(dim=-1).squeeze(0)  # [num_heads]

                # Max attention per head (peakedness)
                max_attn = attn_weights.max(dim=-1).values.mean(dim=-1).squeeze(0)  # [num_heads]

                key = (block_idx, i)
                if key not in all_entropies:
                    all_entropies[key] = []
                    all_max_attn[key] = []
                all_entropies[key].append(mean_entropy.cpu().float().numpy())
                all_max_attn[key].append(max_attn.cpu().float().numpy())

                # Continue actual forward pass using Block.forward (includes skip connections)
                if i < num_encoder:
                    xe = block(xe, x0)
                    skips.append(xe)
                else:
                    decoder_i = i - num_encoder
                    if skips:
                        xe = xe + model.skip_weights[decoder_i].to(dtype=xe.dtype)[None, None, :] * skips.pop()
                    xe = block(xe, x0)

    # Average over sequences
    results = {}
    for key in all_entropies:
        entropies = np.stack(all_entropies[key])
        max_attns = np.stack(all_max_attn[key])
        results[key] = {
            "entropy_per_head": entropies.mean(axis=0),
            "entropy_std_per_head": entropies.std(axis=0),
            "max_attn_per_head": max_attns.mean(axis=0),
            "mean_entropy": entropies.mean(),
            "head_entropy_std": entropies.mean(axis=0).std(),  # diversity across heads
        }
    return results


def analyze_qk_gain_values(model):
    """Extract learned QK gain values from all attention heads."""
    gains = {}
    for name, param in model.named_parameters():
        if "q_gain" in name:
            gains[name] = param.data.cpu().float().numpy().tolist()
    return gains


def analyze_weight_structure(model):
    """Analyze weight matrices for quantization friendliness."""
    results = {}
    for name, param in model.named_parameters():
        if param.ndim != 2 or param.numel() < 4096:
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS) or "smear" in name:
            continue

        W = param.data.float()

        # Singular value analysis
        try:
            S = torch.linalg.svdvals(W)
            # Effective rank (entropy-based)
            p = S / S.sum()
            p = p[p > 1e-10]
            eff_rank = torch.exp(-(p * torch.log(p)).sum()).item()
            # Energy concentration
            total_energy = (S ** 2).sum()
            top10_energy = ((S[:10] ** 2).sum() / total_energy).item()
        except Exception:
            eff_rank = -1
            top10_energy = -1

        # Quantization error for appropriate bit width
        is_mlp = any(p in name for p in MLP_QUANT_PATTERNS)
        is_embed = any(p in name for p in EMBED_QUANT_PATTERNS)
        bits = 5 if is_mlp else (8 if is_embed else 6)

        max_val = 2 ** (bits - 1) - 1
        scale = W.abs().amax(dim=1, keepdim=True) / max_val
        scale = scale.clamp_min(1e-12)
        W_q = torch.clamp(torch.round(W / scale), -max_val - 1, max_val) * scale
        mse = ((W - W_q) ** 2).mean().item()

        results[name] = {
            "mse": mse,
            "bits": bits,
            "eff_rank": eff_rank,
            "top10_energy": top10_energy,
            "shape": list(W.shape),
        }

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("EXP 53: QK-GAIN ANALYSIS")
    print("Comparing exp40-D (QK=1.5) vs exp53 (QK=4.0)")
    print("=" * 70)

    # Load validation data
    print("\nLoading validation data...")
    val_tokens = torch.cat([load_data_shard(Path(s)).to(device) for s in sorted(glob.glob(
        "./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))]).long()

    random.seed(SEED)
    eval_seqs = []
    for _ in range(N_EVAL_SEQS):
        start = random.randint(0, val_tokens.numel() - 2049)
        eval_seqs.append(val_tokens[start:start + 2048])
    del val_tokens

    # ============================================================
    # ANALYSIS 1: Learned QK-Gain Values
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 1: LEARNED QK-GAIN VALUES")
    print("=" * 60)
    print("How much did each head's gain change from initialization?\n")

    all_gains = {}
    for label, path, init in [("exp40-D (init=1.5)", EXP40D_PATH, 1.5),
                               ("exp53 (init=4.0)", EXP53_PATH, 4.0)]:
        print(f"--- {label} ---")
        model, args = load_model(path, init)
        gains = analyze_qk_gain_values(model)
        all_gains[label] = gains
        for name, values in sorted(gains.items()):
            block_id = name.split(".")[1]
            formatted = " ".join([f"{v:.3f}" for v in values])
            mean_val = sum(values) / len(values)
            delta = mean_val - init
            print(f"  Block {block_id}: [{formatted}]  mean={mean_val:.3f} (delta={delta:+.3f} from init={init})")
        del model
        torch.cuda.empty_cache()

    # ============================================================
    # ANALYSIS 2: Attention Entropy
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 2: ATTENTION ENTROPY")
    print("=" * 60)
    print("Lower entropy = sharper attention = more selective.")
    print(f"Measured over {N_EVAL_SEQS} sequences of length 2048.\n")

    entropy_results = {}
    for label, path, init in [("exp40-D (QK=1.5)", EXP40D_PATH, 1.5),
                               ("exp53 (QK=4.0)", EXP53_PATH, 4.0)]:
        print(f"Computing attention entropy for {label}...")
        model, args = load_model(path, init)
        entropy_results[label] = compute_attention_entropy(model, eval_seqs[:N_EVAL_SEQS])
        del model
        torch.cuda.empty_cache()

    labels = list(entropy_results.keys())
    print(f"\n{'Pos':<5} {'Block':<7} {'Pass':<6} {labels[0]+' entropy':<22} {labels[1]+' entropy':<22} {'Delta':<10} {'Sharper?':<10}")
    print("-" * 82)

    total_e1 = total_e2 = 0
    count = 0
    for pos_idx, block_idx in enumerate(LAYER_SCHEDULE):
        pass_count = LAYER_SCHEDULE[:pos_idx + 1].count(block_idx)
        key = (block_idx, pos_idx)
        if key in entropy_results[labels[0]] and key in entropy_results[labels[1]]:
            e1 = entropy_results[labels[0]][key]["mean_entropy"]
            e2 = entropy_results[labels[1]][key]["mean_entropy"]
            delta = e2 - e1
            sharper = "exp53" if delta < 0 else "exp40-D"
            print(f"  {pos_idx:<3} B{block_idx:<5} p{pass_count:<4} {e1:<22.4f} {e2:<22.4f} {delta:+.4f}    {sharper}")
            total_e1 += e1
            total_e2 += e2
            count += 1

    if count > 0:
        print(f"\n  Average entropy: {labels[0]}={total_e1/count:.4f}  {labels[1]}={total_e2/count:.4f}  delta={total_e2/count - total_e1/count:+.4f}")

    # ============================================================
    # ANALYSIS 3: Head Specialization (Diversity)
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 3: HEAD SPECIALIZATION (DIVERSITY)")
    print("=" * 60)
    print("Higher std across heads = more specialized/diverse heads.\n")

    print(f"{'Pos':<5} {'Block':<7} {'Pass':<6} {labels[0]+' head_std':<22} {labels[1]+' head_std':<22} {'More diverse?':<15}")
    print("-" * 77)

    total_s1 = total_s2 = 0
    for pos_idx, block_idx in enumerate(LAYER_SCHEDULE):
        pass_count = LAYER_SCHEDULE[:pos_idx + 1].count(block_idx)
        key = (block_idx, pos_idx)
        if key in entropy_results[labels[0]] and key in entropy_results[labels[1]]:
            s1 = entropy_results[labels[0]][key]["head_entropy_std"]
            s2 = entropy_results[labels[1]][key]["head_entropy_std"]
            more_diverse = "exp53" if s2 > s1 else "exp40-D"
            print(f"  {pos_idx:<3} B{block_idx:<5} p{pass_count:<4} {s1:<22.4f} {s2:<22.4f} {more_diverse}")
            total_s1 += s1
            total_s2 += s2

    if count > 0:
        print(f"\n  Average head std: {labels[0]}={total_s1/count:.4f}  {labels[1]}={total_s2/count:.4f}")

    # ============================================================
    # ANALYSIS 4: Weight Structure & Quantization
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 4: WEIGHT STRUCTURE & QUANTIZATION FRIENDLINESS")
    print("=" * 60)
    print("Lower quant MSE = weights quantize better.")
    print("Lower effective rank = more structured weights.\n")

    weight_results = {}
    for label, path, init in [("exp40-D", EXP40D_PATH, 1.5),
                               ("exp53", EXP53_PATH, 4.0)]:
        print(f"Analyzing weight structure for {label}...")
        model, _ = load_model(path, init)
        weight_results[label] = analyze_weight_structure(model)
        del model
        torch.cuda.empty_cache()

    # Group by layer type
    groups = {}
    for name in weight_results["exp40-D"]:
        for part in ["c_q", "c_k", "c_v", "proj", "gate", "up"]:
            # match the last component (e.g. attn.c_q or mlp.gate)
            if f".{part}." in name or name.endswith(f".{part}.weight"):
                bits = weight_results["exp40-D"][name]["bits"]
                group_key = f"{part} (int{bits})"
                if group_key not in groups:
                    groups[group_key] = []
                mse_40d = weight_results["exp40-D"][name]["mse"]
                mse_53 = weight_results["exp53"][name]["mse"]
                rank_40d = weight_results["exp40-D"][name]["eff_rank"]
                rank_53 = weight_results["exp53"][name]["eff_rank"]
                top10_40d = weight_results["exp40-D"][name]["top10_energy"]
                top10_53 = weight_results["exp53"][name]["top10_energy"]
                groups[group_key].append({
                    "name": name,
                    "mse_40d": mse_40d, "mse_53": mse_53,
                    "rank_40d": rank_40d, "rank_53": rank_53,
                    "top10_40d": top10_40d, "top10_53": top10_53,
                })
                break

    # Also handle mlp.proj (down projection)
    for name in weight_results["exp40-D"]:
        if "mlp.proj" in name:
            bits = weight_results["exp40-D"][name]["bits"]
            group_key = f"mlp.proj (int{bits})"
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append({
                "name": name,
                "mse_40d": weight_results["exp40-D"][name]["mse"],
                "mse_53": weight_results["exp53"][name]["mse"],
                "rank_40d": weight_results["exp40-D"][name]["eff_rank"],
                "rank_53": weight_results["exp53"][name]["eff_rank"],
                "top10_40d": weight_results["exp40-D"][name]["top10_energy"],
                "top10_53": weight_results["exp53"][name]["top10_energy"],
            })

    print(f"\n{'Layer Type':<25} {'exp40-D MSE':<15} {'exp53 MSE':<15} {'Ratio':<10} {'Winner':<10}")
    print("-" * 75)

    for group_key in sorted(groups.keys()):
        entries = groups[group_key]
        avg_40d = sum(e["mse_40d"] for e in entries) / len(entries)
        avg_53 = sum(e["mse_53"] for e in entries) / len(entries)
        ratio = avg_53 / avg_40d if avg_40d > 0 else float('inf')
        winner = "exp53" if ratio < 1.0 else "exp40-D"
        print(f"  {group_key:<23} {avg_40d:<15.8f} {avg_53:<15.8f} {ratio:<10.3f} {winner}")

    print(f"\n{'Layer Type':<25} {'exp40-D rank':<15} {'exp53 rank':<15} {'Delta':<10}")
    print("-" * 65)

    for group_key in sorted(groups.keys()):
        entries = groups[group_key]
        valid = [(e["rank_40d"], e["rank_53"]) for e in entries if e["rank_40d"] > 0]
        if valid:
            avg_40d = sum(r[0] for r in valid) / len(valid)
            avg_53 = sum(r[1] for r in valid) / len(valid)
            delta = avg_53 - avg_40d
            print(f"  {group_key:<23} {avg_40d:<15.1f} {avg_53:<15.1f} {delta:+.1f}")

    print(f"\n{'Layer Type':<25} {'exp40-D top10%':<15} {'exp53 top10%':<15} {'Delta':<10}")
    print("-" * 65)

    for group_key in sorted(groups.keys()):
        entries = groups[group_key]
        valid = [(e["top10_40d"], e["top10_53"]) for e in entries if e["top10_40d"] > 0]
        if valid:
            avg_40d = sum(r[0] for r in valid) / len(valid)
            avg_53 = sum(r[1] for r in valid) / len(valid)
            delta = avg_53 - avg_40d
            print(f"  {group_key:<23} {avg_40d:<15.4f} {avg_53:<15.4f} {delta:+.4f}")

    # ============================================================
    # ANALYSIS 5: Per-Head Attention Entropy Detail
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 5: PER-HEAD ENTROPY AT KEY POSITIONS")
    print("=" * 60)
    print("Shows entropy of each of the 8 heads at B6 (most critical block).\n")

    for pos_idx, block_idx in enumerate(LAYER_SCHEDULE):
        if block_idx != 6:
            continue
        pass_count = LAYER_SCHEDULE[:pos_idx + 1].count(block_idx)
        key = (block_idx, pos_idx)
        if key in entropy_results[labels[0]] and key in entropy_results[labels[1]]:
            e1_heads = entropy_results[labels[0]][key]["entropy_per_head"]
            e2_heads = entropy_results[labels[1]][key]["entropy_per_head"]
            print(f"B6 pass {pass_count} (pos {pos_idx}):")
            print(f"  Head:    " + "  ".join([f"H{i:<5}" for i in range(len(e1_heads))]))
            print(f"  QK=1.5:  " + "  ".join([f"{v:<7.3f}" for v in e1_heads]))
            print(f"  QK=4.0:  " + "  ".join([f"{v:<7.3f}" for v in e2_heads]))
            deltas = e2_heads - e1_heads
            print(f"  Delta:   " + "  ".join([f"{v:+.3f} " for v in deltas]))
            print()

    # ============================================================
    # SUMMARY
    # ============================================================
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Three hypotheses tested:

1. ATTENTION SHARPNESS: Does QK=4.0 produce sharper attention?
   -> Check entropy comparison. Lower = sharper.

2. HEAD SPECIALIZATION: Are QK=4.0 heads more diverse?
   -> Check head_std. Higher = more specialized.

3. QUANTIZATION FRIENDLINESS: Are QK=4.0 weights easier to quantize?
   -> Check MSE comparison (lower = better).
   -> Check effective rank (lower = more structured).
   -> Check top-10 energy concentration (higher = more compressible).

These effects compound:
  Sharper attention -> specialized heads -> structured weights
  -> better quantization -> better GPTQ -> better TTT -> lower val_bpb
""")

    # Save results
    save_data = {
        "entropy_summary": {},
        "weight_summary": {},
    }
    for pos_idx, block_idx in enumerate(LAYER_SCHEDULE):
        key = (block_idx, pos_idx)
        if key in entropy_results[labels[0]] and key in entropy_results[labels[1]]:
            save_data["entropy_summary"][f"pos{pos_idx}_B{block_idx}"] = {
                "exp40d_entropy": float(entropy_results[labels[0]][key]["mean_entropy"]),
                "exp53_entropy": float(entropy_results[labels[1]][key]["mean_entropy"]),
                "exp40d_head_std": float(entropy_results[labels[0]][key]["head_entropy_std"]),
                "exp53_head_std": float(entropy_results[labels[1]][key]["head_entropy_std"]),
            }
    for group_key in sorted(groups.keys()):
        entries = groups[group_key]
        save_data["weight_summary"][group_key] = {
            "exp40d_mse": sum(e["mse_40d"] for e in entries) / len(entries),
            "exp53_mse": sum(e["mse_53"] for e in entries) / len(entries),
        }

    out_path = "/runpod-volume/exp53_analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {out_path}")
    print("Done!")
