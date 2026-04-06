# Exp 47: How Depth Recurrence Reduces Loss

**Date**: 2026-04-06
**Model**: exp40-D float checkpoint (7 unique blocks, dim=624, 26.5M params)
**GPU**: 1xH100 SXM (RunPod)
**Schedule**: `[0,1,2,3,4,3,4,3,4,5,6,5,6]` — 13 effective layers, blocks 3-4 reused 3x, blocks 5-6 reused 2x

---

## 1. The Question

Our model gets 13 effective layers from 7 unique blocks by reusing some blocks multiple times. We know this works (val_bpb=1.1172), but we don't understand *how* the recurrence passes contribute. Specifically:

- Where in the 13-layer pipeline does loss actually decrease?
- Do later passes of the same block help as much as earlier ones?
- Do hard tokens benefit more from recurrence than easy tokens?
- What changes in the attention patterns across passes?

Understanding this guides future work: if recurrence passes are doing distinct work, per-pass adapters (exp44) should help. If they're redundant, we should look elsewhere.

## 2. Method

We probed the trained model's internal state at every position in the 13-layer schedule. At each position, we projected the hidden state to logits (via the tied embedding head) and measured cross-entropy loss. This tells us: **if the model stopped here, how good would its predictions be?**

We also tracked attention patterns, prediction confidence, and token difficulty across layers.

## 3. Where Loss Drops: The Layer-by-Layer Loss Curve

```
Position  Block        Loss     Delta     What's happening
────────────────────────────────────────────────────────────
0         B0           7.858    —         Initial: almost random predictions
1         B1           6.494    -1.364    ▼▼▼▼ Biggest single-layer drop
2         B2           6.366    -0.128    ▼
3         B3-pass1     6.071    -0.295    ▼▼ Block 3 helps on first pass
4         B4-pass1     5.876    -0.195    ▼▼
5         B3-pass2     5.936    +0.060    ▲ LOSS GOES UP (block 3, 2nd pass)
6         B4-pass2     6.003    +0.066    ▲ LOSS GOES UP (block 4, 2nd pass)
7         B3-pass3     6.242    +0.240    ▲▲ LOSS GOES UP MORE (block 3, 3rd pass)
8         B4-pass3     5.672    -0.570    ▼▼▼▼ HUGE drop — B4 pass 3 is critical
9         B5-pass1     5.627    -0.046    ▼
10        B6-pass1     4.361    -1.266    ▼▼▼▼ Second biggest drop (B6 first pass)
11        B5-pass2     4.381    +0.020    ~ flat
12        B6-pass2     1.926    -2.455    ▼▼▼▼▼ LARGEST DROP — final layer
```

### The surprise: middle recurrence passes *increase* loss

Positions 5-7 (block 3 pass 2, block 4 pass 2, block 3 pass 3) all show **increasing** loss when probed independently. The model's intermediate representations at these positions are *worse* for direct prediction than positions 3-4.

This doesn't mean these passes are useless — it means they're **transforming the representation for the benefit of downstream layers**, not improving predictions directly. They're building features that blocks 5-6 and the final layer need, even though those features look worse when projected through the output head.

Think of it like an assembly line: the middle station might make the product look worse (raw components disassembled) because it's rearranging parts for the final assembly station.

### Where the actual loss reduction happens

| Layer group | Loss drop | % of total |
|------------|-----------|------------|
| B0-B1 (unique early blocks) | -1.49 | 25% |
| B2-B4 pass 1 (entry into recurrence) | -0.62 | 10% |
| B3-B4 passes 2-3 (middle recurrence) | +0.20 | builds features, doesn't reduce loss directly |
| B4 pass 3 (exit from recurrence) | -0.57 | 10% |
| B5-B6 pass 1 (decoder entry) | -1.31 | 22% |
| B6 pass 2 (final layer) | -2.46 | 41% |

**The final layer (B6 pass 2) does 41% of the loss reduction.** It takes the features built by all prior layers — including the recurrence passes that increased probed loss — and converts them into accurate predictions. The model concentrates its prediction capability at the very end.

### Block 3 vs Block 4 recurrence

Block 3 and block 4 play different roles in recurrence:

| Pass | Block 3 (delta) | Block 4 (delta) | Interpretation |
|------|----------------|----------------|----------------|
| 1 | -0.295 (helps) | -0.195 (helps) | Both reduce loss on first pass |
| 2 | +0.060 (hurts) | +0.066 (hurts) | Both reorganize features |
| 3 | +0.240 (hurts more) | **-0.570 (huge help)** | B3 destabilizes, B4 recovers |

Block 3 pass 3 produces the worst probed loss of any recurrence position — but block 4 pass 3 immediately follows with the biggest improvement in the recurrence section. **Block 3 appears to deconstruct representations while block 4 reconstructs them** into a form that the decoder blocks need.

## 4. Which Tokens Benefit From Depth?

We split tokens into difficulty buckets based on their final loss and tracked how loss evolves across layers:

```
                    After B0   After B2   After B3p1  After B3p3  After B4p3  After B6p1  Final(B6p2)
Easy (0-25%)         6.87       5.81       5.46        6.23        4.96        2.39        0.01
Medium (25-50%)      7.37       6.08       5.86        5.69        5.41        4.22        0.50
Hard (50-75%)        7.83       6.32       6.07        5.89        5.71        5.01        2.20
Very Hard (75-90%)   8.54       6.63       6.33        6.29        5.98        5.40        4.06
Hardest (90-100%)   10.60       8.18       7.74        8.47        7.56        6.48        6.37
```

### Key findings:

**Easy tokens are solved early.** Easy tokens reach loss 0.01 at the final layer — essentially perfect prediction. They benefit hugely from every layer, including the final one (2.39 → 0.01).

**Recurrence passes help medium-difficulty tokens most.** Medium and hard tokens show modest improvement through recurrence (block 3 pass 1 → pass 3: 5.86 → 5.69 for medium, 6.07 → 5.89 for hard). These tokens need the extra processing.

**The hardest tokens get WORSE through block 3 recurrence.** Hardest tokens go from 7.74 (after B3 pass 1) to 8.47 (after B3 pass 3) — the model is actively de-prioritizing them in the intermediate representation. But block 4 pass 3 partially recovers (8.47 → 7.56).

**The final layer (B6 pass 2) is where easy/medium tokens separate from hard ones.** Before B6p2, all buckets have loss 4-7. After B6p2, easy tokens hit 0.01 while hard ones stay at 4-6. The final layer is essentially a classifier that perfectly predicts easy tokens and gives up on hard ones.

## 5. Prediction Confidence Through Depth

How confident is the model at each layer? We measured top-1 probability and logit entropy:

```
Position  Block        Top-1 Prob   Logit Entropy
────────────────────────────────────────────────────
0         B0           0.003        6.11          Almost uniform over 1024 tokens
1         B1           0.002        6.30          Still near-uniform
2         B2           0.003        6.09
3         B3-pass1     0.003        5.92          Slight sharpening
4         B4-pass1     0.004        5.74
5         B3-pass2     0.006        5.58          Gradually sharpening
6         B4-pass2     0.007        5.42
7         B3-pass3     0.007        5.47          ← slight reversal
8         B4-pass3     0.006        5.52          ← slight reversal
9         B5-pass1     0.004        5.82          ← confidence DROPS at decoder entry
10        B6-pass1     0.058        4.66          ▼▼ First real confidence jump
11        B5-pass2     0.061        4.67
12        B6-pass2     0.530        1.99          ▼▼▼ 53% top-1 probability, entropy halved
```

**The model builds confidence in two sharp jumps**, not gradually:
1. **B6 pass 1** (position 10): top-1 jumps from 0.4% to 5.8%. This is when the model first "has an opinion."
2. **B6 pass 2** (position 12): top-1 jumps from 6.1% to 53%. This is when it commits to a prediction.

The recurrence passes (positions 3-8) barely change confidence. They're working on internal features, not on making predictions. The decoder (B5-B6) is where features get converted to predictions.

**B5 actually reduces confidence** (position 9: top-1 drops from 0.006 to 0.004). Like block 3's recurrence passes, B5 is reorganizing features for B6's benefit, not directly improving predictions.

## 6. Attention Patterns Across Passes

Block 3's attention at each of its 3 passes:

| Pass | Avg Entropy | Local Attention (within 64 tokens) |
|------|-------------|-----------------------------------|
| Pass 1 (pos 3) | 0.073 | 16.0% |
| Pass 2 (pos 5) | 0.073 | 17.0% |
| Pass 3 (pos 7) | 0.069 | 15.6% |

Attention patterns are **remarkably stable** across passes. The same block attends to roughly the same patterns regardless of which pass it's on. Entropy slightly decreases on pass 3 (more focused attention), and locality slightly decreases (attending slightly further).

This suggests the attention mechanism isn't the primary differentiator between passes — the MLP and the `resid_mix` blending with `x0` are what make each pass produce different outputs.

## 7. Implications

### The recurrence pipeline is not what it seems

The naive mental model is: "more passes = better predictions at each step." The data shows the opposite: **middle recurrence passes make probed predictions worse.** They're building abstract features that only become useful when the decoder (B6) processes them.

This is important because it means:
- **Per-pass probing is misleading.** You can't evaluate recurrence quality by looking at intermediate predictions.
- **The final layer does most of the work.** 41% of loss reduction happens in a single layer (B6 pass 2). This makes B6 the most important block to protect during quantization.
- **Block 3 and block 4 have different roles.** B3 destabilizes/reorganizes, B4 stabilizes/compresses. They work as a pair.

### For exp44 (Relaxed Recurrence LoRA)

The data strongly supports per-pass adapters, but the reasoning changes:
- Each pass does **qualitatively different work** (pass 1 reduces loss, passes 2-3 build features for downstream)
- A fixed `resid_mix` can't optimize for these different roles
- Per-pass LoRA would let pass 1 optimize for immediate loss reduction AND feature building, while passes 2-3 could specialize in feature construction

### For GPTQ

Block 6 (B6) is the most critical block — it does 41% + 22% = 63% of total loss reduction across its two passes. Quantization errors in B6 directly impact the final prediction. **GPTQ should prioritize B6 over all other blocks.** Currently we quantize all attention blocks at int6 equally — per-block bit allocation could help.

### For architecture search

The "hourglass" pattern (confidence drops at B5, then jumps at B6) suggests the decoder structure matters more than the recurrence depth. Future architectures might benefit from:
- More unique decoder blocks (currently just B5, B6)
- Allocating more parameters to the final layers
- Fewer recurrence passes with wider final blocks

---

## Appendix A: Activation Analysis (from exp47 Phase 2)

Block 3's output at each of its 3 passes, measured across 131K tokens:

| Comparison | Cosine Similarity |
|------------|------------------|
| Pass 1 → Pass 2 | 0.787 |
| Pass 2 → Pass 3 | 0.859 |
| Pass 1 → Pass 3 | 0.595 |

Norms grow monotonically: 188K → 222K → 246K. Each pass moves the representation further from the start, with diminishing step size. This is iterative directional refinement, not fixed-point convergence.

## Appendix B: Schedule Ablation

We also tested the trained model with different schedules at inference time. Since the weights were trained for schedule A, all other schedules break the model (the weights expect a specific computational graph). The main finding: removing end recurrence (blocks 5-6) hurts 2x more than removing middle recurrence (blocks 3-4), consistent with the loss probing data showing B6 does most of the prediction work.

Full ablation data: [exp47_recurrence_ablation_results.json](exp47_recurrence_ablation_results.json)

---

**Raw data**: [exp47b_deep_diagnostic_results.json](exp47b_deep_diagnostic_results.json)
**Logs**: [exp47b_deep_diagnostic.log](exp47b_deep_diagnostic.log), [exp47_recurrence_ablation.log](exp47_recurrence_ablation.log)
**Scripts**: `scripts/experiments/exp47_recurrence_ablation.py`, `scripts/experiments/exp47b_recurrence_deep_diagnostic.py`
