# Experiment 7 — Deep Analysis

## Summary

Experiment 7 tested **warmdown_iters=2000** (up from 600 in Exp 6), the single highest-leverage change identified in our Exp 6 analysis. **Result: val_bpb = 1.3062** (int8+zlib roundtrip), improving on Exp 6's 1.3124 by 0.006 bpb.

The most surprising finding: **layer 3 activated**. A previously dead middle layer became a meaningful contributor, suggesting the longer warmdown gives underutilized layers time to converge.

---

## Training Results

| Metric | Exp 6 (warmdown=600) | Exp 7 (warmdown=2000) | Change |
|---|---|---|---|
| Steps | 11,909 | 11,597 | -312 |
| Pre-quant val_bpb | 1.3083 | 1.3015 | **-0.0068** |
| Int8+zlib val_bpb | 1.3124 | **1.3062** | **-0.0062** |
| ms/step | 100.8 | 103.5 | +2.7 |
| Compressed size | 15.75 MB | 15.80 MB | +0.05 |

Slightly slower per step (103.5 vs 100.8ms), likely due to different GPU allocation on the new pod. Fewer total steps but better result — the warmdown phase is doing more work per step.

### Val BPB Trajectory

| Step | Exp 6 bpb | Exp 7 bpb | Delta |
|---|---|---|---|
| 1000 | 1.567 | 1.565 | -0.002 |
| 2000 | 1.481 | 1.480 | -0.001 |
| 3000 | 1.445 | 1.445 | 0.000 |
| 4000 | 1.416 | 1.415 | -0.001 |
| 5000 | 1.400 | 1.399 | -0.001 |
| 6000 | 1.385 | 1.383 | -0.002 |
| 7000 | 1.373 | 1.373 | 0.000 |
| 8000 | 1.367 | 1.365 | -0.002 |
| 9000 | 1.359 | 1.359 | 0.000 |
| 10000 | 1.353 | 1.344 | **-0.009** |
| 11000 | 1.344 | 1.314 | **-0.030** |
| Final | 1.308 | **1.302** | **-0.006** |

The curves are nearly identical until step ~9600 where Exp 7's warmdown kicks in (2000 steps before the end). The divergence at step 10000 (-0.009) and 11000 (-0.030) shows the warmdown is the difference — Exp 7 was already in warmdown while Exp 6 hadn't started yet.

---

## Loss Distribution

| Bucket | Exp 2 | Exp 4 | Exp 6 | Exp 7 | Trend |
|---|---|---|---|---|---|
| Easy (<1) | 16.4% | 17.3% | 18.1% | 18.2% | Slowly growing |
| Medium (1-3) | 29.4% | 29.2% | 28.9% | 29.0% | Flat |
| Hard (3-5) | 32.0% | 32.6% | 32.7% | 32.8% | Flat |
| Very hard (>5) | 22.2% | 20.9% | 20.3% | 20.0% | Slowly shrinking |

Marginal improvement — 0.3% fewer very-hard tokens compared to Exp 6. The warmdown is not changing the distribution shape, just slightly sharpening predictions across the board. The hard/medium buckets are essentially unchanged across all experiments.

---

## Position Degradation

| Metric | Exp 2 | Exp 4 | Exp 6 | Exp 7 |
|---|---|---|---|---|
| context_benefit | -0.90 | -1.13 | -1.08 | **-1.20** |
| First 64 avg loss | — | — | 2.76 | 2.62 |
| Last 64 avg loss | — | — | 3.84 | 3.81 |

Position degradation **worsened** from -1.08 to -1.20. The early positions improved more (2.76→2.62) than late positions (3.84→3.81). The longer warmdown causes the model to settle deeper into its existing short-range bias — the LR decay phase reinforces whatever patterns the model has already learned, including its preference for nearby context.

**Implication**: warmdown is a double-edged sword for position utilization. It makes the model better overall but amplifies the architectural short-range preference.

### Position-by-position loss (Exp 7)

| Position range | Avg loss |
|---|---|
| 0-64 | 2.62 |
| 64-128 | 2.40 |
| 128-192 | 2.36 |
| 192-256 | 2.33 |
| 256-320 | 2.88 |
| 320-384 | 3.03 |
| 384-448 | 3.01 |
| 448-512 | 3.14 |
| 512-576 | 3.53 |
| 576-640 | 3.74 |
| 640-704 | 3.77 |
| 704-768 | 3.67 |
| 768-832 | 3.73 |
| 832-896 | 3.69 |
| 896-960 | 3.78 |
| 960-1023 | 3.82 |

The sharp jump at position 256 persists — the model effectively uses context for the first ~256 positions, then degrades. This is unchanged from Exp 6.

---

## Layer Analysis — Layer 3 Activation

| Layer | Exp 6 impact | Exp 7 impact | Change |
|---|---|---|---|
| 0 | ~5.3 | 5.37 | stable (critical) |
| 1 | ~1.5 | 1.54 | stable (critical) |
| 2 | ~0.4 | 0.44 | stable |
| **3** | **~0.1** | **0.58** | **+0.48 (activated!)** |
| 4 | ~0.1 | 0.11 | stable (still dead) |
| 5 | ~0.1 | 0.16 | slight improvement |
| 6 | ~0.1 | 0.15 | slight improvement |
| 7 | ~0.1 | 0.16 | slight improvement |
| 8 | ~2.7 | 2.71 | stable (critical) |

**The most significant structural finding: layer 3 activated.** It went from ~0.1 loss impact (effectively dead weight) to 0.58 (meaningful contributor). The longer warmdown gave this layer enough low-LR gradient updates to converge into a useful configuration.

In Exp 6, layers 3-6 were identified as "dead middle layers" contributing <0.13 loss impact each (~6M wasted parameters). In Exp 7, layer 3 is no longer dead. Layers 4-6 remain underutilized but show slight improvement (0.11-0.16 vs ~0.10).

**Why this matters**: If warmdown=2000 activated layer 3, warmdown=3000 might activate layers 4-6 too. Each activated layer represents ~1.7M parameters becoming productive. This could be worth 0.01-0.02 bpb.

---

## Entropy Analysis

| Quadrant | Exp 4 | Exp 6 (est) | Exp 7 | Interpretation |
|---|---|---|---|---|
| Confident-wrong | 1.7% | ~1.7% | **1.4%** | Model less overconfident |
| Uncertain-wrong | 32.2% | ~32% | **36.3%** | Model more honest about uncertainty |
| Confident-right | 12.2% | ~13.5% | 13.5% | Stable |
| Uncertain-right | — | — | 0.0% | N/A |

The confident-wrong reduction (1.7%→1.4%) is meaningful — these are the highest-leverage tokens where the model thinks it knows the answer but is wrong. The warmdown's settling phase corrected some of these overconfident predictions.

The increase in uncertain-wrong (32%→36%) suggests the model traded false confidence for honest uncertainty. It's making fewer catastrophic mistakes at the cost of being less decisive on ambiguous tokens.

---

## Hard Token Profile

| Metric | Exp 6 | Exp 7 | Change |
|---|---|---|---|
| Very hard word-initial % | ~65% | 65.7% | Stable |
| Very hard late position % | ~61% | 61.0% | Stable |
| Very hard avg_loss | ~6.3 | 6.33 | Stable |

The hard token profile is unchanged. The hardest bigrams remain juncture-point predictions:

| Bigram | Avg loss | Pattern |
|---|---|---|
| `,` → `▁pro` | 6.96 | Comma → word-initial |
| `,` → `▁ind` | 6.64 | Comma → word-initial |
| `,` → `▁first` | 6.48 | Comma → word-initial |
| `▁the` → `▁New` | 6.34 | Function word → proper noun |
| `▁the` → `▁We` | 6.31 | Function word → proper noun |
| `,` → `▁per` | 6.30 | Comma → word-initial |
| `▁to` → `▁comm` | 6.56 | Function word → word-initial |

This confirms the core challenge is architectural — predicting which word follows at syntactic boundaries requires long-range context that the model can't effectively use past position 256.

---

## What We Learned

### Confirmed
1. **Warmdown=2000 > warmdown=600** — 0.006 bpb improvement, confirmed
2. **Layer activation is possible** — longer warmdown can revive dead middle layers
3. **Position degradation worsens with warmdown** — the settling phase reinforces short-range bias
4. **Confident-wrong reduces with warmdown** — better calibrated predictions
5. **Hard token profile is invariant** — no training change affects juncture-point predictions

### New questions
1. Can warmdown=3000 activate layers 4-6? (potential 0.01-0.02 bpb)
2. Is there a warmdown length where position degradation starts hurting overall BPB?
3. Would combining longer training (2400s) with warmdown=2000 be better than warmdown=3000 at 1200s?

---

## Recommendations for Experiment 8

Two viable paths:

**Option A: warmdown=3000, 1200s** — Push warmdown further to see if more layers activate. Risk: may hit diminishing returns on warmdown and worsen position degradation enough to negate gains.

**Option B: warmdown=2000, 2400s** — Double training time while keeping the validated warmdown setting. More total steps means more learning in the grind phase + more steps in warmdown. Expected: ~1.28-1.29 based on the logarithmic extrapolation.

**Recommendation**: Option A first (cheaper, same 1200s). If it plateaus, try Option B.
