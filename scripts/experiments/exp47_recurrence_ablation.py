"""
Exp 47: Depth Recurrence Ablation Study

Using the exp40-D float checkpoint (7 unique blocks, dim=624), evaluate
different layer schedules to measure the impact of recurrence depth.

Ablations:
  A: Full 3x recurrence   [0,1,2,3,4,3,4,3,4,5,6,5,6]  13 eff layers (baseline)
  B: 2x recurrence         [0,1,2,3,4,3,4,5,6,5,6]      11 eff layers
  C: 1x (no recurrence)    [0,1,2,3,4,5,6]               7 eff layers
  D: 3x middle only        [0,1,2,3,4,3,4,3,4,5,6]       11 eff (no end recurrence)
  E: 2x end only           [0,1,2,3,4,5,6,5,6]           9 eff (no middle recurrence)
  F: 4x recurrence         [0,1,2,3,4,3,4,3,4,3,4,5,6,5,6,5,6]  17 eff layers

For each schedule, we:
  1. Build a GPT with that schedule (same 7 unique blocks)
  2. Load matching weights from the float checkpoint
  3. Evaluate val_bpb WITHOUT TTT (pure model quality)
  4. Evaluate val_bpb WITH SGD TTT (full pipeline)

This isolates the contribution of recurrence passes to both base quality and TTT effectiveness.

Additionally, we collect per-pass activation stats:
  - For the full 3x schedule (A), record hidden states at each of block 3's
    three appearances (positions 3, 5, 7). Compute cosine similarity and L2
    distance between passes to understand convergence behavior.
"""

import torch, os, sys, glob, time, math, io, json, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

BASE_DIR = os.environ.get("BASE_DIR", "/runpod-volume/parameter-golf")
RECORDS_DIR = os.path.join(BASE_DIR, "records/track_non_record_16mb/2026-03-24_11L_XSA_SwiGLU_LoRATTT_1xH100")
FLOAT_MODEL_PATH = os.environ.get("FLOAT_MODEL_PATH", os.path.join(BASE_DIR, "exp40d_model.pt"))

TTT_ENABLED = int(os.environ.get("TTT_ENABLED", "1"))
TTT_MAX_DOCS = int(os.environ.get("TTT_MAX_DOCS", "500"))  # default 500 docs for speed
TTT_MIN_DOC_LEN = 32
TTT_TRAIN_CHUNK = 2048
SCORE_CAP = 2048
SGD_LR = float(os.environ.get("SGD_LR", "0.005"))
SGD_MOMENTUM = float(os.environ.get("SGD_MOMENTUM", "0.9"))
ACTIVATION_ANALYSIS = int(os.environ.get("ACTIVATION_ANALYSIS", "1"))
ACTIVATION_N_SEQS = int(os.environ.get("ACTIVATION_N_SEQS", "64"))
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
Block = _mod.Block
load_data_shard = _mod.load_data_shard
Hyperparameters = _mod.Hyperparameters
build_sentencepiece_luts = _mod.build_sentencepiece_luts

import sentencepiece as spm

device = torch.device("cuda")


# ================================================================
# Ablation schedules
# ================================================================

SCHEDULES = {
    "A_full_3x":       [0,1,2,3,4,3,4,3,4,5,6,5,6],     # 13 eff (baseline)
    "B_2x_recurrence": [0,1,2,3,4,3,4,5,6,5,6],           # 11 eff
    "C_no_recurrence":  [0,1,2,3,4,5,6],                    # 7 eff
    "D_3x_middle_only": [0,1,2,3,4,3,4,3,4,5,6],           # 11 eff
    "E_2x_end_only":   [0,1,2,3,4,5,6,5,6],                # 9 eff
    "F_4x_recurrence": [0,1,2,3,4,3,4,3,4,3,4,5,6,5,6,5,6],  # 17 eff
}


def compute_bpb(total_nll, total_bytes):
    return (total_nll / math.log(2.0)) / max(total_bytes, 1.0)


def build_model_with_schedule(schedule, base_state, args):
    """Build a GPT with a given layer schedule and load weights from base_state.

    The number of unique blocks is always 7 (max(schedule)+1).
    The skip_weights shape depends on the schedule length, so we
    can't just load the full state dict. We load block weights and
    other shared params, then initialize skip_weights fresh.
    """
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init, xsa_last_n=args.xsa_last_n,
        layer_schedule=schedule,
    ).to(device)

    # Load matching weights, skip mismatched skip_weights
    model_state = model.state_dict()
    loaded = 0
    skipped = 0
    for k, v in base_state.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                model_state[k].copy_(v.to(device))
                loaded += 1
            else:
                skipped += 1
        else:
            skipped += 1
    model.load_state_dict(model_state, strict=True)
    eff = len(schedule)
    print(f"    Schedule {schedule} ({eff} eff layers): loaded {loaded}, skipped {skipped} params")
    return model


def eval_no_ttt(model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, args):
    """Evaluate val_bpb without TTT (pure model quality)."""
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

    val_loss = loss_sum / token_count
    val_bpb = compute_bpb(loss_sum, byte_count)
    return val_loss, val_bpb


def eval_with_ttt(model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, max_docs):
    """Evaluate val_bpb with SGD all-weights TTT."""
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

    # Short docs: no TTT
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

    # Long docs: SGD TTT
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

    # Restore base state
    model.load_state_dict(base_state, strict=True)
    final_bpb = compute_bpb(ttt_nll, ttt_bytes)
    final_loss = ttt_nll / max(ttt_tokens, 1.0)
    elapsed = time.perf_counter() - t0
    return final_loss, final_bpb, elapsed


def activation_analysis(model, val_tokens, args):
    """Record hidden states at each appearance of block 3 in the full 3x schedule.

    Block 3 appears at positions 3, 5, 7 in schedule [0,1,2,3,4,3,4,3,4,5,6,5,6].
    We hook the output of block 3 at each position and compare across passes.
    """
    print("\n  Activation analysis: recording hidden states at block 3 passes...")

    # We need to intercept the forward pass to capture per-position outputs.
    # The model's forward() iterates layer_schedule and calls self.blocks[block_idx].
    # Block 3 is called at positions 3, 5, 7.
    # We'll hook blocks[3] and track which call number we're on.

    block3_outputs = {0: [], 1: [], 2: []}  # pass_idx -> list of tensors
    call_count = [0]  # mutable counter

    def hook_fn(module, input, output):
        pass_idx = call_count[0]
        # Only record first 3 calls per forward (block 3 appears 3 times)
        if pass_idx < 3:
            block3_outputs[pass_idx].append(output.detach().float())
        call_count[0] += 1

    hook = model.blocks[3].register_forward_hook(hook_fn)

    seq_len = args.train_seq_len
    random.seed(SEED)
    n_seqs = min(ACTIVATION_N_SEQS, (val_tokens.numel() - 1) // seq_len)

    model.eval()
    with torch.no_grad():
        for si in range(n_seqs):
            call_count[0] = 0
            start = si * seq_len
            x = val_tokens[start:start + seq_len].unsqueeze(0)
            y = val_tokens[start + 1:start + seq_len + 1].unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model(x, y)

    hook.remove()

    # Compute statistics
    results = {}
    for pass_idx in range(3):
        acts = torch.cat(block3_outputs[pass_idx], dim=0)  # [n_seqs * seq_len, dim]
        results[f"pass_{pass_idx+1}_mean_norm"] = acts.norm(dim=-1).mean().item()
        results[f"pass_{pass_idx+1}_std"] = acts.std().item()

    # Pairwise cosine similarity between passes
    for i in range(3):
        for j in range(i + 1, 3):
            a = torch.cat(block3_outputs[i], dim=0)  # [N, dim]
            b = torch.cat(block3_outputs[j], dim=0)  # [N, dim]
            # Per-token cosine similarity, then average
            cos = F.cosine_similarity(a, b, dim=-1).mean().item()
            l2 = (a - b).norm(dim=-1).mean().item()
            results[f"cos_sim_pass{i+1}_vs_pass{j+1}"] = cos
            results[f"l2_dist_pass{i+1}_vs_pass{j+1}"] = l2

    # How much does each pass change the representation?
    for i in range(3):
        if i > 0:
            a = torch.cat(block3_outputs[i - 1], dim=0)
            b = torch.cat(block3_outputs[i], dim=0)
            delta_norm = (b - a).norm(dim=-1).mean().item()
            base_norm = a.norm(dim=-1).mean().item()
            results[f"relative_change_pass{i}_to_pass{i+1}"] = delta_norm / max(base_norm, 1e-12)

    return results


if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"# Exp 47: Depth Recurrence Ablation Study")
    print(f"{'#'*60}")

    args = Hyperparameters()
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Load float model state dict
    print(f"\nLoading float checkpoint: {FLOAT_MODEL_PATH}")
    float_state = torch.load(FLOAT_MODEL_PATH, map_location="cpu", weights_only=False)
    if "model" in float_state:
        float_state = float_state["model"]
    print(f"  Keys: {len(float_state)}, Total params: {sum(v.numel() for v in float_state.values()):,}")

    # Load val tokens
    print("Loading val tokens...")
    val_shards = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))
    val_tokens = torch.cat([load_data_shard(Path(s)).to(device) for s in val_shards]).long()
    print(f"  Val tokens: {val_tokens.numel():,}")

    # ================================================================
    # Phase 1: No-TTT eval across all schedules
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Phase 1: No-TTT evaluation (pure model quality)")
    print(f"{'='*60}")

    results = {}
    for name, schedule in SCHEDULES.items():
        print(f"\n  [{name}] schedule={schedule} ({len(schedule)} eff layers)")
        model = build_model_with_schedule(schedule, float_state, args)
        t0 = time.perf_counter()
        val_loss, val_bpb = eval_no_ttt(
            model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, args
        )
        elapsed = time.perf_counter() - t0
        results[name] = {
            "schedule": schedule,
            "eff_layers": len(schedule),
            "no_ttt_val_loss": val_loss,
            "no_ttt_val_bpb": val_bpb,
            "no_ttt_eval_time_s": elapsed,
        }
        print(f"    val_loss={val_loss:.4f}  val_bpb={val_bpb:.4f}  time={elapsed:.1f}s")
        del model
        torch.cuda.empty_cache()

    # Print Phase 1 summary
    print(f"\n{'='*60}")
    print(f"Phase 1 Summary (No TTT)")
    print(f"{'='*60}")
    print(f"{'Schedule':<25} {'Eff Layers':>10} {'val_bpb':>10} {'Delta vs A':>12}")
    baseline_bpb = results["A_full_3x"]["no_ttt_val_bpb"]
    for name in SCHEDULES:
        r = results[name]
        delta = r["no_ttt_val_bpb"] - baseline_bpb
        sign = "+" if delta >= 0 else ""
        print(f"{name:<25} {r['eff_layers']:>10} {r['no_ttt_val_bpb']:>10.4f} {sign}{delta:>11.4f}")

    # ================================================================
    # Phase 2: Activation analysis on full 3x schedule
    # ================================================================
    if ACTIVATION_ANALYSIS:
        print(f"\n{'='*60}")
        print(f"Phase 2: Activation analysis (block 3, 3 passes)")
        print(f"{'='*60}")

        model = build_model_with_schedule(SCHEDULES["A_full_3x"], float_state, args)
        act_results = activation_analysis(model, val_tokens, args)

        print(f"\n  Block 3 activation statistics:")
        print(f"  {'Metric':<40} {'Value':>10}")
        print(f"  {'-'*52}")
        for k, v in sorted(act_results.items()):
            print(f"  {k:<40} {v:>10.4f}")

        results["activation_analysis"] = act_results
        del model
        torch.cuda.empty_cache()

    # ================================================================
    # Phase 3: TTT eval on key schedules
    # ================================================================
    if TTT_ENABLED:
        print(f"\n{'='*60}")
        print(f"Phase 3: SGD TTT evaluation ({TTT_MAX_DOCS} docs)")
        print(f"{'='*60}")

        # Only eval a subset with TTT (it's expensive)
        ttt_schedules = ["A_full_3x", "B_2x_recurrence", "C_no_recurrence"]
        for name in ttt_schedules:
            schedule = SCHEDULES[name]
            print(f"\n  [{name}] schedule={schedule} ({len(schedule)} eff layers)")
            model = build_model_with_schedule(schedule, float_state, args)
            ttt_loss, ttt_bpb, elapsed = eval_with_ttt(
                model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                max_docs=TTT_MAX_DOCS
            )
            results[name]["ttt_val_loss"] = ttt_loss
            results[name]["ttt_val_bpb"] = ttt_bpb
            results[name]["ttt_eval_time_s"] = elapsed
            results[name]["ttt_docs"] = TTT_MAX_DOCS
            print(f"    TTT val_loss={ttt_loss:.4f}  val_bpb={ttt_bpb:.4f}  time={elapsed:.0f}s")
            del model
            torch.cuda.empty_cache()

        # Print Phase 3 summary
        print(f"\n{'='*60}")
        print(f"Phase 3 Summary (SGD TTT, {TTT_MAX_DOCS} docs)")
        print(f"{'='*60}")
        print(f"{'Schedule':<25} {'Eff':>4} {'No-TTT BPB':>11} {'TTT BPB':>9} {'TTT Delta':>10} {'TTT Gain':>10}")
        for name in ttt_schedules:
            r = results[name]
            ttt_delta = r["ttt_val_bpb"] - results["A_full_3x"].get("ttt_val_bpb", 0)
            ttt_gain = r["no_ttt_val_bpb"] - r["ttt_val_bpb"]
            sign = "+" if ttt_delta >= 0 else ""
            print(f"{name:<25} {r['eff_layers']:>4} {r['no_ttt_val_bpb']:>11.4f} {r['ttt_val_bpb']:>9.4f} {sign}{ttt_delta:>9.4f} {ttt_gain:>10.4f}")

    # ================================================================
    # Save results
    # ================================================================
    out_path = "/runpod-volume/exp47_recurrence_ablation_results.json"
    # Convert non-serializable types
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {kk: (vv if not isinstance(vv, (torch.Tensor,)) else vv.item()) for kk, vv in v.items()}
        else:
            serializable[k] = v
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ================================================================
    # Final summary
    # ================================================================
    print(f"\n{'#'*60}")
    print(f"# FINAL SUMMARY")
    print(f"{'#'*60}")
    print(f"\nDepth recurrence impact (no TTT):")
    for name in SCHEDULES:
        r = results[name]
        delta = r["no_ttt_val_bpb"] - baseline_bpb
        sign = "+" if delta >= 0 else ""
        bar = "█" * max(0, int(abs(delta) * 1000))
        direction = "worse" if delta > 0 else "better" if delta < 0 else "same"
        print(f"  {name:<25} {r['eff_layers']:>2} layers  bpb={r['no_ttt_val_bpb']:.4f}  ({sign}{delta:.4f} {direction}) {bar}")

    if ACTIVATION_ANALYSIS and "activation_analysis" in results:
        a = results["activation_analysis"]
        print(f"\nBlock 3 convergence behavior:")
        print(f"  Cosine sim pass 1→2: {a.get('cos_sim_pass1_vs_pass2', 0):.4f}")
        print(f"  Cosine sim pass 2→3: {a.get('cos_sim_pass2_vs_pass3', 0):.4f}")
        print(f"  Cosine sim pass 1→3: {a.get('cos_sim_pass1_vs_pass3', 0):.4f}")
        print(f"  Relative change 1→2: {a.get('relative_change_pass1_to_pass2', 0):.4f}")
        print(f"  Relative change 2→3: {a.get('relative_change_pass2_to_pass3', 0):.4f}")

    print(f"\nDone!")
