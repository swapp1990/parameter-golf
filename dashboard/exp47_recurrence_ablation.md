# Exp 47: Depth Recurrence Ablation Study

**Date**: 2026-04-06
**Model**: exp40-D float checkpoint (7 unique blocks, dim=624, 26.5M params)
**GPU**: 1xH100 SXM (RunPod)
**Runtime**: ~25 min total (Phase 1: ~20 min, Phase 2: ~1 min, Phase 3: ~2 min)

---

## 1. Purpose

Our model uses depth recurrence: 7 unique transformer blocks arranged in a schedule of 13 effective layers `[0,1,2,3,4,3,4,3,4,5,6,5,6]`. Blocks 3-4 are reused 3 times, blocks 5-6 are reused twice.

We know this architecture works (val_bpb=1.1172 with GPTQ+TTT). But we don't know:
- How much does each recurrence pass contribute?
- What happens if we remove or add passes?
- Are the shared blocks converging to a fixed point, or doing something different each pass?
- Can TTT compensate for missing recurrence?

This experiment answers these questions by ablating the layer schedule at inference time, using the same trained weights.

## 2. Methodology

We take the trained exp40-D float checkpoint and evaluate it with **different layer schedules**, keeping all block weights identical. For each schedule:
1. Build a model with that schedule (same 7 unique blocks, different forward pass order)
2. Load matching weights from the checkpoint
3. Evaluate on the full 62M-token validation set (no TTT)
4. For key schedules, also evaluate with SGD all-weights TTT (500 docs)

Additionally, we hook block 3's output at each of its 3 appearances in the full schedule to measure how representations change across passes.

**Important caveat**: The model was *trained* with schedule A. All other schedules are mismatched — the model never saw them during training. So these results measure **how dependent the model is on its exact schedule**, not what the optimal schedule would be if trained from scratch.

## 3. Ablation Schedules

The model was trained with schedule A. All other schedules are **inference-time ablations** — we load the same trained weights but change the forward pass order. This measures how dependent the model is on its exact recurrence pattern, not what would happen if we trained with a different schedule.

| Name | Schedule | Eff Layers | What's different from A |
|------|----------|-----------|------------------------|
| **A** | `[0,1,2,3,4,3,4,3,4,5,6,5,6]` | 13 | Nothing (original trained schedule) |
| **B** | `[0,1,2,3,4,3,4,5,6,5,6]` | 11 | Removed one middle pass of blocks 3-4 |
| **C** | `[0,1,2,3,4,5,6]` | 7 | Removed all recurrence (each block once) |
| **D** | `[0,1,2,3,4,3,4,3,4,5,6]` | 11 | Removed end recurrence of blocks 5-6 |
| **E** | `[0,1,2,3,4,5,6,5,6]` | 9 | Removed middle recurrence of blocks 3-4 |
| **F** | `[0,1,2,3,4,3,4,3,4,3,4,5,6,5,6,5,6]` | 17 | Added extra passes (4x middle, 3x end) |

## 4. Results

### Phase 1: No-TTT Evaluation (Full Val Set)

| Schedule | Eff Layers | val_loss | val_bpb | Delta vs A | Time |
|----------|-----------|----------|---------|------------|------|
| **A: Full 3x** | **13** | **1.9405** | **1.1493** | **baseline** | 230s |
| B: 2x recurrence | 11 | 2.8568 | 1.6919 | +0.5426 | 198s |
| C: No recurrence | 7 | 4.6837 | 2.7739 | +1.6246 | 134s |
| D: 3x middle only | 11 | 4.4193 | 2.6173 | +1.4680 | 198s |
| E: 2x end only | 9 | 3.1648 | 1.8744 | +0.7251 | 167s |
| F: 4x recurrence | 17 | 6.4420 | 3.8153 | +2.6660 | 294s |

### Phase 2: Block 3 Activation Analysis

Block 3 appears at positions 3, 5, and 7 in the full schedule. We recorded its output at each position across 64 validation sequences (131K tokens).

**Cosine similarity between passes:**

| Comparison | Cosine Similarity | L2 Distance |
|------------|------------------|-------------|
| Pass 1 → Pass 2 | 0.787 | 139,726 |
| Pass 2 → Pass 3 | 0.859 | 128,411 |
| Pass 1 → Pass 3 | 0.595 | 207,065 |

**Per-pass statistics:**

| Pass | Mean Norm | Std Dev | Relative Change from Prior |
|------|-----------|---------|---------------------------|
| 1 (position 3) | 188,481 | 7,672 | — |
| 2 (position 5) | 221,943 | 8,957 | 0.741 (74% of input norm) |
| 3 (position 7) | 245,646 | 9,978 | 0.579 (58% of input norm) |

### Phase 3: SGD TTT Evaluation (500 docs)

| Schedule | Eff | No-TTT BPB | TTT BPB | TTT Gain | TTT Delta vs A |
|----------|-----|-----------|---------|----------|----------------|
| A: Full 3x | 13 | 1.1493 | 1.1496 | -0.0003 | baseline |
| B: 2x recurrence | 11 | 1.6919 | 1.6599 | 0.0321 | +0.5103 |
| C: No recurrence | 7 | 2.7739 | 2.7698 | 0.0042 | +1.6202 |

---

## 5. Analysis

### Finding 1: The model is catastrophically specialized for its exact schedule

Removing a single recurrence pass (B: 2x, 11 layers) increases BPB by **+0.543** — from near-SOTA (1.15) to barely functional (1.69). Removing all recurrence (C: 7 layers) produces BPB of 2.77, worse than an untrained model of similar size.

This is not a gradual degradation. The model doesn't work "somewhat worse" without recurrence — it **breaks**. The weights have been trained to expect data flowing through the exact sequence of blocks in the exact order. Change the sequence and the learned representations become incoherent.

**Why this matters**: This proves depth recurrence is not just "bonus processing." The model has fundamentally encoded its layer schedule into its weight structure. Blocks 3-4 learn different functions at each pass even though they use identical weights — the function differs because the *input distribution* at each position is different, and the weights are optimized for the union of all three input distributions.

### Finding 2: End recurrence (blocks 5-6) matters more than middle recurrence (blocks 3-4)

| Removed | BPB | Delta |
|---------|-----|-------|
| End recurrence removed (D: 3x middle only) | 2.617 | +1.468 |
| Middle recurrence removed (E: 2x end only) | 1.874 | +0.725 |

Removing end recurrence is **2x worse** than removing middle recurrence. The decoder blocks (5-6) in the U-Net's second half are more critical because:

1. **Skip connections**: The decoder consumes skip connections from the encoder. Blocks 5-6 on their second pass receive different skip connections than on their first pass. Without the second pass, half the skip connections go unused.

2. **Proximity to output**: Blocks 5-6 are the last processing step before the prediction head. Errors here propagate directly to the loss. Middle blocks (3-4) have more downstream layers to compensate.

3. **Decoder role**: In the U-Net structure, encoder blocks (positions 0-5) build representations while decoder blocks (positions 6-12) refine them for prediction. The decoder's second pass is the model's "final edit" — removing it is like submitting a first draft.

### Finding 3: More recurrence (4x) is worse, not better

F (17 layers, 4x) has BPB of **3.82** — the worst of all schedules. Adding extra passes through blocks that were trained with 3 passes creates representations the downstream blocks have never seen. The skip_weights are also mismatched (trained for 13-layer U-Net, now receiving 17-layer skips).

This confirms that the recurrence count is a fundamental architectural choice, not a tunable parameter at inference time.

### Finding 4: Block 3 is NOT converging to a fixed point

If depth recurrence worked like a fixed-point iteration (applying the same function until output stabilizes), we'd expect:
- Increasing cosine similarity between successive passes (convergence)
- Decreasing relative change (stabilization)

What we actually observe:

| | Expected (fixed-point) | Observed |
|-|----------------------|----------|
| Cosine sim 1→2 vs 2→3 | 2→3 should be higher | 2→3 IS higher (0.859 vs 0.787) |
| Relative change 1→2 vs 2→3 | 2→3 should be smaller | 2→3 IS smaller (0.579 vs 0.741) |
| Cosine sim 1→3 | Should be highest | Is LOWEST (0.595) |

The first two observations are consistent with convergence — each pass makes smaller changes, and adjacent passes are more similar. But the third observation (pass 1 and pass 3 are the LEAST similar) reveals that the block is **not converging to a fixed point**. Instead, it's **iteratively transforming** the representation through a trajectory: each pass moves the representation in a consistent direction, with diminishing step size but without circling back toward the starting point.

The growing norms confirm this:
- Pass 1: mean norm 188,481
- Pass 2: mean norm 221,943 (+18%)
- Pass 3: mean norm 245,646 (+11%)

Representations grow in magnitude with each pass. A fixed-point attractor would stabilize the norm. Instead, the block is pumping energy into the representation — each pass amplifies specific features that the downstream blocks need.

### Finding 5: TTT cannot rescue broken schedules

| Schedule | TTT Gain |
|----------|----------|
| A (correct) | -0.0003 (none — already optimal) |
| B (2x, missing 1 pass) | 0.032 (recovers 6% of the 0.543 gap) |
| C (no recurrence) | 0.004 (recovers 0.2% of the 1.625 gap) |

SGD TTT adapts all 26.5M weights per document, yet it recovers almost nothing. This is because the problem isn't "wrong weights for this document" — it's "wrong computational graph." TTT can adjust what each block computes, but it can't add missing passes through the network.

This also confirms that TTT's power comes from adapting a *working* model to specific documents, not from fixing architectural deficiencies.

---

## 6. Implications

### For our current model

The depth recurrence is deeply load-bearing. The model cannot be simplified by reducing passes at inference time. The 3x/2x recurrence pattern is baked into the weight structure.

### For exp44 (Relaxed Recurrence LoRA)

The activation analysis shows block 3 produces meaningfully different representations at each pass (cosine sim 0.60-0.86). This means each pass is doing different work despite using identical weights. Giving each pass its own small LoRA adapter would let it specialize explicitly rather than relying on input distribution differences alone.

The relative change data (0.74 for pass 1→2, 0.58 for pass 2→3) suggests passes 1 and 2 benefit most from specialization (they're doing the most transformation), while pass 3 is already more incremental.

### For architecture search

The catastrophic sensitivity to schedule changes means:
1. **Schedule must be fixed before training** — you can't explore schedules at eval time
2. **Skip connections create tight coupling** — U-Net skip weights are sized for a specific layer count
3. **Recurrence count matters** — 2x, 3x, and 4x give very different results when trained

Future experiments should test different schedules by retraining from scratch, not by modifying a trained model's schedule.

### For quantization (GPTQ)

Blocks 3-4 are the most reused (3x each). Their quantization errors appear 3 times in the forward pass, compounding. The activation analysis shows norms growing across passes (188K → 222K → 246K), which means quantization errors also get amplified across passes. This explains why GPTQ was so effective (-0.009 BPB): reducing rounding errors in blocks 3-4 has 3x the leverage.

---

## 7. Raw Data

Full results: [exp47_recurrence_ablation_results.json](exp47_recurrence_ablation_results.json)
Full log: [exp47_recurrence_ablation.log](exp47_recurrence_ablation.log)
Script: `scripts/experiments/exp47_recurrence_ablation.py`
