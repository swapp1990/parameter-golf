"""
Exp 47b: Deep Diagnostic — How Recurrence Passes Reduce Loss

Using the exp40-D float checkpoint, we probe the model's internal state
at every position in the layer schedule to understand:

1. LOSS PROBING: At each position in the schedule, project hidden state to
   logits and measure loss. This shows exactly how much each layer/pass
   improves predictions. We can see: does block 3's 2nd pass help more
   than its 1st? Do middle blocks or end blocks contribute more to loss reduction?

2. ATTENTION ANALYSIS: For block 3's three passes, extract attention weights.
   Do they attend to different tokens on different passes? Does attention
   become more focused or more diffuse with depth?

3. PER-TOKEN DIFFICULTY: Split tokens into buckets by their final loss (easy/medium/hard).
   For each bucket, measure how loss evolves across layers. Do recurrence passes
   help hard tokens more than easy ones?

4. ACTIVATION STATISTICS: Track hidden state norms, variance, and entropy
   of logit distributions at each position. This shows how "confident" the
   model becomes at each depth level.
"""

import torch, os, sys, glob, time, math, io, json, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

BASE_DIR = os.environ.get("BASE_DIR", "/runpod-volume/parameter-golf")
RECORDS_DIR = os.path.join(BASE_DIR, "records/track_non_record_16mb/2026-03-24_11L_XSA_SwiGLU_LoRATTT_1xH100")
FLOAT_MODEL_PATH = os.environ.get("FLOAT_MODEL_PATH", os.path.join(BASE_DIR, "exp40d_model.pt"))
N_SEQS = int(os.environ.get("N_SEQS", "128"))
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
Hyperparameters = _mod.Hyperparameters
build_sentencepiece_luts = _mod.build_sentencepiece_luts
import sentencepiece as spm

device = torch.device("cuda")
SCHEDULE = [0,1,2,3,4,3,4,3,4,5,6,5,6]
SCHEDULE_LABELS = [
    "B0", "B1", "B2",
    "B3-pass1", "B4-pass1",
    "B3-pass2", "B4-pass2",
    "B3-pass3", "B4-pass3",
    "B5-pass1", "B6-pass1",
    "B5-pass2", "B6-pass2",
]


def probe_loss_at_position(model, x, targets, position):
    """Run model up to a given position in the schedule, project to logits, measure loss.

    This tells us: if the model stopped after this position, how good would its
    predictions be?
    """
    xe = model.tok_emb(x)
    if model.ngram is not None:
        xe = xe + model.ngram(x)
    xe = F.rms_norm(xe, (xe.size(-1),))
    xe = model.smear(xe)
    x0 = xe

    num_enc = model.num_encoder_layers
    skips = []

    for i in range(min(position + 1, num_enc)):
        block_idx = model.layer_schedule[i]
        xe = model.blocks[block_idx](xe, x0)
        skips.append(xe)

    if position >= num_enc:
        for i in range(position + 1 - num_enc):
            if skips:
                xe = xe + model.skip_weights[i].to(dtype=xe.dtype)[None, None, :] * skips.pop()
            block_idx = model.layer_schedule[num_enc + i]
            xe = model.blocks[block_idx](xe, x0)

    # Project to logits using tied embeddings
    normed = F.rms_norm(xe, (xe.size(-1),))
    logits = F.linear(normed.reshape(-1, normed.size(-1)), model.tok_emb.weight)
    logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)

    loss = F.cross_entropy(logits.float(), targets.reshape(-1), reduction="none")
    return loss, logits


def extract_attention_weights(model, x, block_idx, positions_in_schedule):
    """Run the full forward pass, hooking the attention module at specific positions
    to capture attention weight matrices.

    Returns: dict mapping position -> attention weights [batch, heads, seq, seq]
    """
    attn_weights = {}
    call_count = [0]
    target_calls = set()

    # Figure out which call numbers correspond to our target block
    call_num = 0
    for i, bidx in enumerate(model.layer_schedule):
        if bidx == block_idx:
            if i in positions_in_schedule:
                target_calls.add(call_num)
            call_num += 1
        elif bidx != block_idx:
            pass
    # Actually, simpler: count calls to this specific block
    call_count_for_block = [0]
    position_to_call = {}
    for i, bidx in enumerate(model.layer_schedule):
        if bidx == block_idx:
            position_to_call[i] = call_count_for_block[0]
            call_count_for_block[0] += 1

    target_call_nums = {position_to_call[p] for p in positions_in_schedule if p in position_to_call}

    block_call_count = [0]

    def hook_fn(module, args, output):
        # The CausalSelfAttention forward doesn't return attention weights by default.
        # We need to compute them manually from Q, K.
        pass

    # Instead, let's hook at the block level and manually compute attention
    # We'll do a custom forward pass
    xe = model.tok_emb(x)
    if model.ngram is not None:
        xe = xe + model.ngram(x)
    xe = F.rms_norm(xe, (xe.size(-1),))
    xe = model.smear(xe)
    x0 = xe
    skips = []

    for i in range(model.num_encoder_layers):
        bidx = model.layer_schedule[i]
        if bidx == block_idx and i in positions_in_schedule:
            # Manually extract attention weights before running the block
            block = model.blocks[bidx]
            mix = block.resid_mix.to(dtype=xe.dtype)
            x_mixed = mix[0][None, None, :] * xe + mix[1][None, None, :] * x0
            normed = block.attn_norm(x_mixed)
            # Get Q, K
            q = block.attn.c_q(normed)
            k = block.attn.c_k(normed)
            B, T, _ = q.shape
            nh = block.attn.num_heads
            nkv = block.attn.num_kv_heads
            hd = q.shape[-1] // nh
            q = q.view(B, T, nh, hd).transpose(1, 2)  # [B, nh, T, hd]
            k = k.view(B, T, nkv, hd).transpose(1, 2)  # [B, nkv, T, hd]
            # GQA: repeat K for grouped heads
            if nkv < nh:
                k = k.repeat_interleave(nh // nkv, dim=1)
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)
            # Apply causal mask
            causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask[None, None, :, :], float('-inf'))
            attn_w = F.softmax(scores.float(), dim=-1)
            attn_weights[i] = attn_w.detach().cpu()

        xe = model.blocks[bidx](xe, x0)
        skips.append(xe)

    for i in range(model.num_decoder_layers):
        if skips:
            xe = xe + model.skip_weights[i].to(dtype=xe.dtype)[None, None, :] * skips.pop()
        bidx = model.layer_schedule[model.num_encoder_layers + i]
        pos = model.num_encoder_layers + i
        if bidx == block_idx and pos in positions_in_schedule:
            block = model.blocks[bidx]
            mix = block.resid_mix.to(dtype=xe.dtype)
            x_mixed = mix[0][None, None, :] * xe + mix[1][None, None, :] * x0
            normed = block.attn_norm(x_mixed)
            q = block.attn.c_q(normed)
            k = block.attn.c_k(normed)
            B, T, _ = q.shape
            nh = block.attn.num_heads
            nkv = block.attn.num_kv_heads
            hd = q.shape[-1] // nh
            q = q.view(B, T, nh, hd).transpose(1, 2)
            k = k.view(B, T, nkv, hd).transpose(1, 2)
            if nkv < nh:
                k = k.repeat_interleave(nh // nkv, dim=1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)
            causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask[None, None, :, :], float('-inf'))
            attn_w = F.softmax(scores.float(), dim=-1)
            attn_weights[pos] = attn_w.detach().cpu()

        xe = model.blocks[bidx](xe, x0)

    return attn_weights


if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"# Exp 47b: Deep Recurrence Diagnostic")
    print(f"{'#'*60}")

    args = Hyperparameters()
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Load model
    print(f"\nLoading model...")
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
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Load val tokens
    print("Loading val tokens...")
    val_shards = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))
    val_tokens = torch.cat([load_data_shard(Path(s)).to(device) for s in val_shards]).long()
    print(f"  Val tokens: {val_tokens.numel():,}")

    seq_len = args.train_seq_len
    n_seqs = min(N_SEQS, (val_tokens.numel() - 1) // seq_len)
    print(f"  Using {n_seqs} sequences of length {seq_len}")

    # ================================================================
    # DIAGNOSTIC 1: Loss probing at every position in the schedule
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Diagnostic 1: Loss at each layer position")
    print(f"{'='*60}")

    # For each position 0..12, compute what the loss would be if we
    # projected to output at that point.
    per_position_losses = {i: [] for i in range(len(SCHEDULE))}
    per_token_final_loss = []

    with torch.no_grad():
        for si in range(n_seqs):
            start = si * seq_len
            x = val_tokens[start:start + seq_len].unsqueeze(0)
            y = val_tokens[start + 1:start + seq_len + 1]

            for pos in range(len(SCHEDULE)):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss_per_token, _ = probe_loss_at_position(model, x, y, pos)
                per_position_losses[pos].append(loss_per_token.float().cpu())

                if pos == len(SCHEDULE) - 1:
                    per_token_final_loss.append(loss_per_token.float().cpu())

            if (si + 1) % 16 == 0:
                print(f"  Sequences {si+1}/{n_seqs} done")

    # Compute average loss at each position
    print(f"\n  Loss at each layer position (lower = better predictions):")
    print(f"  {'Position':<6} {'Block':<12} {'Avg Loss':>10} {'Avg BPB':>10} {'Delta':>10}")
    print(f"  {'-'*50}")

    position_stats = {}
    prev_loss = None
    for pos in range(len(SCHEDULE)):
        all_losses = torch.cat(per_position_losses[pos])
        avg_loss = all_losses.mean().item()
        avg_bpb = avg_loss / math.log(2.0)
        delta = (avg_loss - prev_loss) if prev_loss is not None else 0.0
        delta_str = f"{delta:+.4f}" if prev_loss is not None else "—"
        print(f"  {pos:<6} {SCHEDULE_LABELS[pos]:<12} {avg_loss:>10.4f} {avg_bpb:>10.4f} {delta_str:>10}")
        position_stats[pos] = {
            "label": SCHEDULE_LABELS[pos],
            "block_idx": SCHEDULE[pos],
            "avg_loss": avg_loss,
            "avg_bpb": avg_bpb,
            "delta_from_prev": delta if prev_loss is not None else None,
        }
        prev_loss = avg_loss

    # Show the improvement from each recurrence pass of block 3
    print(f"\n  Block 3 recurrence contribution:")
    b3_positions = [3, 5, 7]  # positions where block 3 appears
    for i, pos in enumerate(b3_positions):
        loss_at = position_stats[pos]["avg_loss"]
        loss_before = position_stats[pos - 1]["avg_loss"] if pos > 0 else None
        if loss_before:
            improvement = loss_before - loss_at
            print(f"    Pass {i+1} (position {pos}): loss {loss_at:.4f}, improved by {improvement:.4f} from prior layer")

    print(f"\n  Block 4 recurrence contribution:")
    b4_positions = [4, 6, 8]
    for i, pos in enumerate(b4_positions):
        loss_at = position_stats[pos]["avg_loss"]
        loss_before = position_stats[pos - 1]["avg_loss"]
        improvement = loss_before - loss_at
        print(f"    Pass {i+1} (position {pos}): loss {loss_at:.4f}, improved by {improvement:.4f} from prior layer")

    # ================================================================
    # DIAGNOSTIC 2: Per-token difficulty analysis
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Diagnostic 2: Loss reduction by token difficulty")
    print(f"{'='*60}")

    # Split tokens into difficulty buckets based on final loss
    all_final_losses = torch.cat(per_token_final_loss)
    percentiles = [0, 25, 50, 75, 90, 100]
    thresholds = [torch.quantile(all_final_losses, p / 100.0).item() for p in percentiles]

    bucket_names = ["Easy (0-25%)", "Medium (25-50%)", "Hard (50-75%)", "Very Hard (75-90%)", "Hardest (90-100%)"]
    bucket_masks = []
    for i in range(len(bucket_names)):
        lo = thresholds[i]
        hi = thresholds[i + 1]
        if i == len(bucket_names) - 1:
            mask = all_final_losses >= lo
        else:
            mask = (all_final_losses >= lo) & (all_final_losses < hi)
        bucket_masks.append(mask)

    print(f"\n  Loss evolution by difficulty bucket:")
    print(f"  {'Bucket':<22}", end="")
    for pos in [0, 2, 3, 5, 7, 8, 10, 12]:
        print(f" {SCHEDULE_LABELS[pos]:>12}", end="")
    print()
    print(f"  {'-'*120}")

    difficulty_stats = {}
    for bi, (bname, bmask) in enumerate(zip(bucket_names, bucket_masks)):
        n_tokens = bmask.sum().item()
        print(f"  {bname:<22}", end="")
        bucket_data = {}
        for pos in [0, 2, 3, 5, 7, 8, 10, 12]:
            all_pos_losses = torch.cat(per_position_losses[pos])
            bucket_loss = all_pos_losses[bmask].mean().item()
            print(f" {bucket_loss:>12.4f}", end="")
            bucket_data[SCHEDULE_LABELS[pos]] = bucket_loss
        print(f"  (n={int(n_tokens)})")
        difficulty_stats[bname] = bucket_data

    # Show which bucket benefits most from recurrence
    print(f"\n  Improvement from block 3 recurrence (pass 1→3) by difficulty:")
    for bi, (bname, bmask) in enumerate(zip(bucket_names, bucket_masks)):
        loss_at_3 = torch.cat(per_position_losses[3])[bmask].mean().item()  # after B3 pass 1
        loss_at_7 = torch.cat(per_position_losses[7])[bmask].mean().item()  # after B3 pass 3
        improvement = loss_at_3 - loss_at_7
        pct = 100 * improvement / max(loss_at_3, 1e-12)
        print(f"    {bname:<22}: {loss_at_3:.4f} → {loss_at_7:.4f} (Δ={improvement:.4f}, {pct:.1f}%)")

    # ================================================================
    # DIAGNOSTIC 3: Attention pattern analysis
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Diagnostic 3: Attention patterns across block 3 passes")
    print(f"{'='*60}")

    # Block 3 appears at positions 3, 5, 7
    b3_schedule_positions = [3, 5, 7]
    attn_entropy_per_pass = {pos: [] for pos in b3_schedule_positions}
    attn_locality_per_pass = {pos: [] for pos in b3_schedule_positions}  # fraction attending to nearby tokens

    with torch.no_grad():
        for si in range(min(32, n_seqs)):
            start = si * seq_len
            x = val_tokens[start:start + seq_len].unsqueeze(0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                attn_weights = extract_attention_weights(model, x, block_idx=3, positions_in_schedule=b3_schedule_positions)

            for pos in b3_schedule_positions:
                if pos not in attn_weights:
                    continue
                aw = attn_weights[pos]  # [1, heads, T, T]
                # Attention entropy (higher = more diffuse)
                aw_clamped = aw.clamp_min(1e-10)
                entropy = -(aw_clamped * aw_clamped.log()).sum(dim=-1)  # [1, heads, T]
                attn_entropy_per_pass[pos].append(entropy.mean().item())

                # Locality: fraction of attention on tokens within distance 64
                T = aw.shape[-1]
                positions = torch.arange(T)
                dist_matrix = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()
                local_mask = (dist_matrix <= 64).unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
                local_fraction = (aw * local_mask).sum(dim=-1).mean().item()
                attn_locality_per_pass[pos].append(local_fraction)

    print(f"\n  Block 3 attention statistics per pass:")
    print(f"  {'Pass':<15} {'Avg Entropy':>12} {'Local (<64 tokens)':>20}")
    print(f"  {'-'*50}")
    attn_stats = {}
    for i, pos in enumerate(b3_schedule_positions):
        if attn_entropy_per_pass[pos]:
            avg_ent = sum(attn_entropy_per_pass[pos]) / len(attn_entropy_per_pass[pos])
            avg_loc = sum(attn_locality_per_pass[pos]) / len(attn_locality_per_pass[pos])
            print(f"  Pass {i+1} (pos {pos}){'':<3} {avg_ent:>12.4f} {avg_loc:>20.4f}")
            attn_stats[f"pass_{i+1}"] = {"entropy": avg_ent, "locality": avg_loc}

    # ================================================================
    # DIAGNOSTIC 4: Logit confidence at each position
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Diagnostic 4: Prediction confidence at each layer")
    print(f"{'='*60}")

    # For each position, compute top-1 probability and entropy of the logit distribution
    confidence_stats = {}
    with torch.no_grad():
        # Use a subset for speed
        for pos in range(len(SCHEDULE)):
            top1_probs = []
            logit_entropies = []
            for si in range(min(32, n_seqs)):
                start = si * seq_len
                x = val_tokens[start:start + seq_len].unsqueeze(0)
                y = val_tokens[start + 1:start + seq_len + 1]

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, logits = probe_loss_at_position(model, x, y, pos)
                probs = F.softmax(logits.float(), dim=-1)
                top1 = probs.max(dim=-1).values.mean().item()
                ent = -(probs * probs.clamp_min(1e-10).log()).sum(dim=-1).mean().item()
                top1_probs.append(top1)
                logit_entropies.append(ent)

            avg_top1 = sum(top1_probs) / len(top1_probs)
            avg_ent = sum(logit_entropies) / len(logit_entropies)
            confidence_stats[pos] = {"top1_prob": avg_top1, "logit_entropy": avg_ent}

    print(f"\n  {'Position':<6} {'Block':<12} {'Top-1 Prob':>10} {'Logit Entropy':>14}")
    print(f"  {'-'*44}")
    for pos in range(len(SCHEDULE)):
        cs = confidence_stats[pos]
        print(f"  {pos:<6} {SCHEDULE_LABELS[pos]:<12} {cs['top1_prob']:>10.4f} {cs['logit_entropy']:>14.4f}")

    # ================================================================
    # Save all results
    # ================================================================
    results = {
        "position_stats": position_stats,
        "difficulty_stats": difficulty_stats,
        "attention_stats": attn_stats,
        "confidence_stats": {str(k): v for k, v in confidence_stats.items()},
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "schedule": SCHEDULE,
        "schedule_labels": SCHEDULE_LABELS,
    }
    out_path = "/runpod-volume/exp47b_deep_diagnostic_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"\nDone!")
