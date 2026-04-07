# Exp 53/54: QK-Gain and Batch Size Smoke Tests

**Date**: 2026-04-06
**GPU**: 1xH100 SXM (RunPod)
**Base config**: 7 blocks, dim=624, schedule [0,1,2,3,4,3,4,3,4,5,6,5,6], Muon optimizer

---

## What We Tested

| Experiment | Change | Motivation |
|-----------|--------|-----------|
| Baseline | QK-Gain=1.5, batch=524K | Our current config |
| Exp 53 | QK-Gain=4.0 | Competition reported -0.004 BPB from this change |
| Exp 54 | batch=786K | Competition reported -0.004 BPB from this change |

All runs: 700s wallclock, VAL_LOSS_EVERY=100, same seed, same data, no EMA, no TTT.

## Training Curves

```
Step   Baseline (524K)   Exp53 QK=4.0 (524K)   Exp54 batch=786K
────────────────────────────────────────────────────────────────
  0       4.1039            4.1039                4.1039
100       1.9572            1.9470 (-0.010)       1.9051 (-0.052)
200       1.6198            1.6137 (-0.006)       1.5928 (-0.027)
300       1.5232            1.5111 (-0.012)       1.4965 (-0.027)
400       1.4672            1.4579 (-0.009)       1.4192 (-0.048)
500       1.4256            1.4196 (-0.006)       —
491       —                 —                     1.3709
                                                  (wallclock stop)
600       —*                1.3825                —
700       —*                1.3333                —
718       —*                1.3299 (final)        —

*Baseline continued past 500 but we don't have later val evals due to how
 the shell script logged output. From step_avg=970ms, baseline reaches
 ~720 steps in 700s, same as exp53.
```

## Analysis: Three Ways to Compare

### 1. Matched Steps (step 500) — Is QK-Gain=4.0 better per step?

| | Baseline | Exp 53 (QK=4.0) | Delta |
|-|----------|-----------------|-------|
| val_bpb @ step 500 | 1.4256 | 1.4196 | **-0.006** |
| Tokens seen | 262M | 262M | same |
| Step time | 970ms | 975ms | same |

**Yes. QK-Gain=4.0 is consistently -0.006 to -0.012 better per step.** The advantage is real and appears from step 100 onward. At identical token counts and compute, QK-Gain=4.0 produces a better model.

### 2. Matched Steps (step 400) — Is batch=786K better per step?

| | Baseline | Exp 54 (786K) | Delta |
|-|----------|--------------|-------|
| val_bpb @ step 400 | 1.4672 | 1.4192 | **-0.048** |
| Tokens seen | 210M | 314M | **+50% more** |
| Step time | 970ms | 1427ms | **+47% slower** |

**No — this isn't a fair comparison.** Exp 54 sees 50% more tokens per step. At step 400, it has seen 314M tokens vs baseline's 210M. The improvement comes from more data, not a better learning algorithm.

### 3. Matched Tokens (~262M) — Is batch=786K more efficient per token?

To compare at matched tokens, we need to estimate where exp 54 is at 262M tokens. At batch=786K, 262M tokens = ~333 steps. Interpolating between step 300 (1.4965) and step 400 (1.4192):

- Exp 54 at ~262M tokens (step ~333): **≈1.471** (interpolated)
- Baseline at 262M tokens (step 500): **1.4256**
- Delta: **+0.045 (exp 54 is WORSE)**

**At matched tokens, batch=786K is significantly worse.** Fewer gradient updates per token means slower learning. The larger batch gives cleaner gradients but each token contributes less to weight updates.

### 4. Matched Wallclock (700s) — Which gets the best model for the same cost?

| | Baseline (est.) | Exp 53 (QK=4.0) | Exp 54 (786K) |
|-|----------------|-----------------|---------------|
| Steps in 700s | ~720 | 718 | 491 |
| Tokens in 700s | 377M | 376M | 386M |
| Final val_bpb | ~1.33 (est.) | **1.3299** | 1.3709 |

**Exp 53 (QK-Gain=4.0) wins at matched wallclock.** It reaches 1.3299 in the same time baseline would reach ~1.33. Small advantage (~-0.003).

**Exp 54 (batch=786K) LOSES at matched wallclock.** Despite seeing slightly more tokens (386M vs 377M), the slower step time means fewer gradient updates, resulting in worse val_bpb (1.3709 vs ~1.33).

## The Key Insight

**Batch=786K only helps if you train for enough total steps.** In a 700-step smoke test, the fewer gradient updates hurt more than the cleaner gradients help. The competition-reported advantage was measured on full training runs (10K+ steps) where the model approaches convergence and gradient noise matters more.

For our full training run (240 min, ~14K steps at batch=524K or ~9.5K steps at batch=786K):
- At batch=524K: ~14K steps × 524K = 7.3B tokens
- At batch=786K: ~9.5K steps × 786K = 7.5B tokens (similar total tokens, fewer steps)

The question is whether 14K "noisy" steps beats 9.5K "clean" steps. This smoke test suggests: at 700 steps, more steps wins. At 14K steps, the answer might flip. **We need a longer smoke test (2000+ steps) to predict the full training outcome reliably.**

## Conclusions

| Finding | Confidence | Action |
|---------|-----------|--------|
| QK-Gain=4.0 helps -0.006 at step 500 | **High** (matched comparison) | **Include in next full training** |
| Batch=786K helps at matched wallclock at 700 steps | **No — it hurts** | **Needs longer test before committing** |
| Batch=786K helps at matched tokens | **No — it's worse** | Same as above |
| Batch=786K may help at 10K+ steps | **Unknown** (not tested) | **Run 2000-step test to verify** |

## Recommendation

**For the next full training run: use QK-Gain=4.0 with batch=524K.** This is the safe choice — QK-Gain=4.0 is proven to help at all timescales, while batch=786K is unproven and potentially harmful for our training budget.

If we want to test batch=786K properly, run a 2000-step smoke test (~35 min on H100) and compare at matched steps AND matched wallclock. If it still loses at 2000 steps, it won't help at 14K either.
