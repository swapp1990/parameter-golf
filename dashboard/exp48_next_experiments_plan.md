# Exp 48+: Next Experiments Plan

**Current best**: val_bpb = 1.1172 (exp40-D + GPTQ + SGD TTT)
**SOTA**: 1.1086-1.1147
**Gap**: 0.0025-0.0086

Based on exp47/47b findings, we now understand WHERE loss reduction happens in our model. This guides which changes are most likely to close the remaining gap.

---

## What the Data Tells Us

From exp47b, the loss reduction breakdown:

| Layer group | Loss reduction | Share | Implication |
|-------------|---------------|-------|-------------|
| B6 (both passes) | -3.72 | **63%** | Most critical block — protect during quantization, invest more params |
| B1 | -1.36 | 23% | Important early block |
| B4 pass 3 | -0.57 | 10% | Exit from recurrence — reconstructs features |
| Middle recurrence (B3/B4 p2-3) | +0.20 | builds features | Feature factory for B6, not direct prediction |
| Everything else | -0.28 | 5% | Marginal |

The model's prediction power is concentrated in B6. Middle recurrence builds features that B6 needs. This suggests two classes of improvements:

**A. Make B6 more powerful** (more params, better quantization, wider)
**B. Make the feature factory more efficient** (per-pass adapters, better recurrence)

---

## Experiments: No Retraining Required

These use the existing exp40-D checkpoint. Fast to test.

### Exp 48: Per-Block GPTQ Bit Allocation

**Thesis**: B6 does 63% of loss reduction but gets the same int6 quantization as every other attention block. Giving B6 higher precision should disproportionately help.

**Smoke test**:
1. Re-quantize with B6 at int8 (instead of int6), all else unchanged
2. Re-quantize with B6 at float16 (no quantization), all else int5/int6
3. Measure artifact size — does it still fit 16MB?
4. Eval 500 docs with SGD TTT each variant

**Expected**: -0.002 to -0.005 BPB from protecting B6. Artifact grows by ~200-400KB (B6 has ~3.7M params, int8 vs int6 adds ~1 bit/param = ~460KB).

**Cost**: ~$1 (eval only, no training)
**Priority**: HIGH — cheapest test, directly motivated by data

### Exp 49: TTT with More Scoring Tokens

**Thesis**: We currently score only the first 2048 tokens per document (SCORE_CAP=2048). The loss probing shows B6 is most effective at the end of the sequence (confidence builds over positions). Scoring more tokens lets B6 work on positions where it has more context.

**Smoke test**:
1. SCORE_CAP=4096 (score first 4096 tokens instead of 2048)
2. SCORE_CAP=8192
3. Eval 500 docs each

**Expected**: Small improvement (-0.001 to -0.003). Risk: longer scoring means more compute per document.

**Cost**: ~$2 (longer eval)
**Priority**: MEDIUM — cheap but uncertain gain

### Exp 50: Adaptive TTT Learning Rate by Document Difficulty

**Thesis**: Medium-difficulty tokens benefit most from recurrence. SGD TTT uses a fixed lr=0.005 for all documents. Documents with mostly easy tokens may benefit from lower lr (less perturbation), while documents with mostly hard tokens may need higher lr (more aggressive adaptation).

**Smoke test**:
1. Run 1 SGD step, measure loss. Use loss as difficulty proxy.
2. Low-loss docs: lr=0.002. High-loss docs: lr=0.008. Medium: lr=0.005.
3. Compare vs fixed lr=0.005 on 500 docs.

**Expected**: -0.001 to -0.002 BPB. Marginal — SGD TTT is already near-optimal.

**Cost**: ~$1
**Priority**: LOW — narrow expected gain

---

## Experiments: Short Retraining Required (500-1000 steps)

These modify the architecture or training and need short smoke tests. ~10-15 min each on H100.

### Exp 51: Per-Pass resid_mix (Minimal Relaxed Recurrence)

**Thesis**: The `resid_mix` bottleneck is real — block 3 uses the same blend ratio on pass 1 (where it reduces loss) and pass 3 (where it destabilizes for B4). Giving each pass its own `resid_mix` lets the block adapt its x/x0 blend per pass.

**Implementation**: In the forward pass, index into `resid_mix_per_pass[pass_idx]` instead of `resid_mix`. Adds ~3,744 parameters (negligible).

**Smoke test** (500 steps):
1. Per-pass resid_mix for blocks 3-4 (3 passes × 2 blocks)
2. Per-pass resid_mix for all shared blocks (3-4 and 5-6)
3. Compare val_loss at step 500 vs baseline

**Expected**: -0.005 to -0.015 at 500 steps. If even this tiny change helps, full per-pass LoRA (exp44) will help more.

**Cost**: ~$1
**Priority**: HIGH — cheapest retraining test, directly answers Q3

### Exp 52: Per-Pass LoRA (Relaxed Recurrence — exp44)

**Thesis**: The loss probing shows passes 1, 2, and 3 have qualitatively different roles (loss-reducing, feature-building, destabilizing). LoRA adapters let each pass specialize without adding many parameters.

**Implementation**: For each shared block at each pass, add rank-4 LoRA on Q and V projections. ~60K params total.

**Smoke test** (500 steps):
1. LoRA on Q/V for blocks 3-4 only (3 passes × 2 blocks × 2 projections)
2. LoRA on Q/V for all shared blocks
3. LoRA on Q/V/MLP for all shared blocks
4. Compare val_loss at step 500

**Expected**: -0.01 to -0.02 at 500 steps. This is the most motivated architecture change we have.

**Cost**: ~$2 (3 variants × 500 steps)
**Priority**: HIGH — strongest theoretical motivation from exp47b data

### Exp 53: QK-Gain = 4.0

**Thesis**: Competition found -0.004 BPB from QK-Gain=4.0 (we use 1.5). This is a simple hyperparameter change validated externally.

**Smoke test** (500 steps):
1. QK-Gain=4.0, all else identical
2. Compare val_loss at step 500 vs baseline

**Expected**: -0.005 to -0.010 at 500 steps (competition-validated).

**Cost**: ~$0.50
**Priority**: HIGH — externally validated, zero-risk

### Exp 54: Larger Batch (786K tokens)

**Thesis**: Competition found -0.004 BPB from batch=786K vs 524K. Larger batches give cleaner gradient estimates.

**Smoke test** (500 steps):
1. TRAIN_BATCH_TOKENS=786432, all else identical
2. Compare val_loss at step 500

**Expected**: -0.003 to -0.008 at 500 steps.

**Cost**: ~$0.50
**Priority**: HIGH — externally validated

### Exp 55: Wider B6 (Asymmetric Block Widths)

**Thesis**: B6 does 63% of loss reduction. What if we made B6 wider (more params) at the expense of the recurrence blocks? Since middle recurrence passes build features (not predictions), they might tolerate being narrower.

**Implementation**: This is a bigger change — would need a custom model where B5-B6 have dim=700 and B3-B4 have dim=560 (or similar). Projection layers bridge between widths.

**Smoke test** (500 steps):
1. B6 at dim=700, B0-B4 at dim=560. Bridge with linear projections.
2. Verify total params ≈ 26M and artifact fits 16MB.

**Expected**: Uncertain. Could be -0.01 or could fail if the narrower recurrence blocks can't build the features B6 needs.

**Cost**: ~$2 (implementation + 500 steps)
**Priority**: MEDIUM — high potential but risky and complex to implement

---

## Results So Far

### Phase 1: Exp 48 — Per-block GPTQ (DEAD END)

Gave B6 int8 instead of int6. Full val set pre-TTT eval (62M tokens, deterministic):

| Variant | Pre-TTT val_bpb | Delta |
|---------|----------------|-------|
| A: All int6 (baseline) | 1.164032 | — |
| B: B6 at int8 | 1.163826 | -0.0002 |

**Verdict**: GPTQ already optimizes B6's rounding at int6. Extra bits add nothing. Dead end.

### Phase 2: Exp 53/54 — QK-Gain and Batch Size

700-second training runs on 1xH100. Full analysis: [exp53_54_smoke_test_results.md](exp53_54_smoke_test_results.md)

**At matched steps (step 500, same tokens):**

| | val_bpb | Delta vs baseline |
|-|---------|-------------------|
| Baseline (QK=1.5, batch=524K) | 1.4256 | — |
| Exp 53 (QK=4.0) | 1.4196 | **-0.006** |

**At matched wallclock (700s):**

| | Steps | val_bpb | Delta |
|-|-------|---------|-------|
| Baseline (est.) | ~720 | ~1.33 | — |
| Exp 53 (QK=4.0) | 718 | **1.3299** | ~-0.003 |
| Exp 54 (batch=786K) | 491 | 1.3709 | ~+0.04 (WORSE) |

**Key finding**: Batch=786K is worse at 700 steps because the 47% slower step time means fewer gradient updates. QK-Gain=4.0 is a consistent winner.

**Verdicts**:
- **Exp 53 (QK-Gain=4.0)**: CONFIRMED improvement. Include in next full training.
- **Exp 54 (batch=786K)**: WORSE at short runs. Needs 2000+ step test to verify if it helps at convergence.

### Phase 1: Exp 49 — SCORE_CAP (FREE IMPROVEMENT)

500-doc TTT eval with different scoring windows. No retraining.

| SCORE_CAP | val_bpb | Delta | Time |
|-----------|---------|-------|------|
| 2048 (current) | 1.1385 | — | 35s |
| **4096** | **1.1369** | **-0.0017** | 30s |
| 8192 | 1.1372 | -0.0013 | 30s |

**500-doc result was misleading.** Full 50K eval:

| SCORE_CAP | 500 docs | 50K docs |
|-----------|---------|---------|
| 2048 | 1.1385 | 1.1172 |
| 4096 | 1.1369 (-0.0017) | **1.1214 (+0.0042 WORSE)** |

Scoring more tokens hurts on the full dataset. The extra positions (2049-4096) include tokens the model just trained on via SGD TTT, diluting the BPB metric. **Dead end.**

### Still to run

| Experiment | Status |
|-----------|--------|
| Exp 51 (per-pass resid_mix) | Not started |
| Exp 52 (per-pass LoRA) | Not started |

---

## Recommended Next Steps

### Option A: Run exp 51/52 smoke tests (1 hour, ~$3)
Test per-pass resid_mix and LoRA. These are our original architecture ideas from the depth recurrence analysis. If either shows signal at 500 steps, combine with QK-Gain=4.0 for full training.

### Option B: Full training with QK-Gain=4.0 now (4 hours, ~$11)
We have one confirmed winner. Train 240 min with QK-Gain=4.0 + batch=524K, apply GPTQ + TTT. Expected result: val_bpb ~1.113-1.116.

### Option C: Both in parallel
Run exp 51/52 on 1xH100 while starting a full training run on another pod.

**Target**: val_bpb < 1.110

---

## Predictions vs Actuals

| Experiment | Predicted | Actual | Correct? |
|-----------|-----------|--------|----------|
| Exp 48 (B6 int8 GPTQ) | -0.003 | -0.0002 | WRONG — GPTQ already handles B6 |
| Exp 49 (SCORE_CAP=4096) | -0.001 | **+0.0042 on 50K docs** | WRONG — 500-doc test was misleading |
| Exp 53 (QK-Gain=4.0) | -0.007 | **-0.006 at step 500** | ~RIGHT |
| Exp 54 (Batch=786K) | -0.005 | **+0.04 worse at matched wallclock** | WRONG — hurts at short runs |
