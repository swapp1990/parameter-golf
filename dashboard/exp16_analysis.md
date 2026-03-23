# Experiment 16 — Optimization Sweep Results

## Setup

All experiments compared against a **control** run using Exp 15 config (11L SwiGLU SWA, batch=524K, seq=2048) on 1xH100 SXM for 300s. Control val_bpb=1.6870 at 496 steps.

## Batch 1: Eval-Only (on Exp 15 checkpoint)

| Experiment | Metric | Result | Verdict |
|-----------|--------|--------|---------|
| Sliding window stride=64 | val_loss delta | **-0.0255** (vs baseline) | Better than stride=256 (-0.037 full set, -0.025 on 100K subset) |
| Head ablation (88 heads) | Dead heads found | **0** (all impact < 0.01) | Every head matters |
| MLP removal L5 | val_loss delta | +0.156 | MLPs not expendable |
| MLP removal L6 | val_loss delta | +0.144 | MLPs not expendable |
| MLP removal L7 | val_loss delta | +0.146 | MLPs not expendable |
| MLP removal L5,6,7 | val_loss delta | +0.805 | Catastrophic without middle MLPs |
| Without SmearGate | val_loss delta | **+1.745** | SmearGate is critical |
| Without BigramHash | val_loss delta | **+0.000** | **BigramHash does nothing!** |
| Without both | val_loss delta | +1.745 | Same as without SmearGate alone |

### Key Findings from Batch 1

1. **BigramHash is dead weight.** Zero contribution to the trained model. Can remove it entirely, saving ~590K params with no quality loss.

2. **SmearGate is critical.** Removing it causes +1.745 loss — catastrophic. The model has completely internalized SmearGate's bigram signal.

3. **No dead heads.** All 88 attention heads contribute meaningfully. Head pruning is not viable.

4. **MLP layers are all necessary.** Even the weakest middle layers (L5-L7) each contribute ~0.15 loss. Heterogeneous MLP allocation won't work by removing middle MLPs.

---

## Batch 2: Short Training Runs (300s, 1xH100, batch=524K, seq=2048)

| Experiment | Params | Steps | val_bpb | Delta vs Control | Verdict |
|-----------|--------|-------|---------|-----------------|---------|
| **Control** | 27.1M | 496 | **1.6870** | — | Baseline |
| Partial RoPE (16/64 dims) | 27.1M | 499 | 1.8268 | **+0.140 (worse)** | Failed |
| LN Scale (1/√(layer+1)) | 27.1M | 504 | 1.6899 | +0.003 (noise) | No effect |
| Heterogeneous MLP (4x endpoints, 2x middle) | 23.8M | 507 | 1.6976 | +0.011 (worse) | Failed |

### Why They Failed

**Partial RoPE (-0.14 worse):** The model needs full positional encoding across all 64 dims. With only 16 dims getting RoPE, 75% of each head's capacity lacks position information. This breaks causal attention — the model can't properly distinguish token positions. PR #315's success with partial RoPE may be specific to their architecture or require different head_dim.

**LN Scale (no effect):** Scaling normalization output by 1/√(layer+1) had zero impact. The model's existing RMSNorm + learned scale parameters (attn_scale, mlp_scale) already handle layer-level magnitude control. Adding another scaling factor is redundant.

**Heterogeneous MLP (+0.011 worse):** Despite L0 and L10 having high ablation impact, giving them 4x MLP while shrinking middle layers to 2x hurt overall. The 3.3M param reduction (27.1M → 23.8M) outweighed the concentration benefit. The middle layers' MLP contribution (0.15 loss each) is too important to reduce.

---

## Overall Summary

| Technique | Expected impact | Actual impact | Worth pursuing? |
|-----------|----------------|---------------|-----------------|
| Remove BigramHash | Save 590K params | **Zero quality loss** | **Yes — free params** |
| Sliding window stride=64 | -0.02 BPB | **-0.025 val_loss** | **Yes — eval only** |
| Partial RoPE (16/64) | -0.005 to -0.010 | **+0.140 (catastrophic)** | No |
| LN Scale | -0.002 to -0.004 | **+0.003 (noise)** | No |
| Heterogeneous MLP | -0.005 to -0.008 | **+0.011 (worse)** | No |
| Head pruning | -0.002 to -0.005 | N/A (no dead heads) | No |
| MLP removal L5-L7 | -0.001 to -0.003 | **+0.15 per layer** | No |

---

## Exp 17: XSA (Cross-token Self-Attention bias removal)

### What it does
After attention computes output y, subtract the self-value projection: `z = y - (dot(y,v)/dot(v,v)) * v`. Forces attention to contribute only contextual information, not redundant self-information. Applied to last 4 layers.

### Result

| Experiment | Steps | val_bpb | Delta vs Control |
|-----------|-------|---------|-----------------|
| Control | 496 | 1.6870 | — |
| **XSA (last 4 layers)** | 492 | **1.6863** | **-0.0007 (noise)** |

### Verdict: No effect at this scale
The 0.0007 difference is within noise. Possible reasons:
- The attention similarity bias may not be a bottleneck at 11L/512d scale (paper showed gains at 0.7B+)
- SmearGate already provides strong local context, reducing the need for attention self-information
- 490 steps is too few to show the effect (though Exp 16 experiments showed clear signals at similar step counts)

XSA added ~7% step time overhead (665ms vs 622ms) for zero benefit.

---

---

## Exp 18: XSA + Partial RoPE + EMA (Full Run)

### Config
Exp 17 stack + Partial RoPE (16/64 dims) + EMA (decay=0.997, every 10 steps) replacing SWA.
1xH100 SXM, batch=524K, seq=2048, 4850s wallclock, 7,795 steps at 615ms/step.

### Result: val_bpb = 1.1977 — WORSE than Exp 17 (1.1826) by 0.015

| Step | Exp 17 (XSA) | Exp 18 (XSA+RoPE+EMA) | Gap |
|------|-------------|----------------------|-----|
| 1000 | 1.3348 | 1.3491 | +0.014 |
| 2000 | 1.2719 | 1.2815 | +0.010 |
| 3000 | 1.2453 | 1.2540 | +0.009 |
| 4000 | 1.2294 | 1.2373 | +0.008 |
| 5000 | 1.2179 | 1.2254 | +0.008 |
| 6000 | 1.2006 | 1.2070 | +0.006 |
| 7000 | 1.1840 | 1.1901 | +0.006 |
| **Final** | **1.1826** | **1.1977** | **+0.015** |

### Why it failed

**Partial RoPE** caused a 0.014 BPP deficit at step 1000 that slowly narrowed to 0.006 by step 7000. The model was never able to fully compensate for having only 25% of dims carry positional information. This model (512d / 8 heads = 64 head_dim) simply doesn't have enough dimensions to sacrifice 75% to content-only matching.

**EMA made the final result worse.** The gap was 0.006 during training but widened to 0.015 after EMA averaging. EMA (decay=0.997) over-smoothed the warmdown weights, losing sharp features. SWA's uniform average of 15 discrete checkpoints was better.

### Verdict
- **Partial RoPE: conclusively dead** for 64 head_dim. Might work at larger head_dim (128+).
- **EMA: worse than SWA** at these settings. SWA remains the better weight averaging strategy.
- **Exp 17 (XSA only) remains the best result at val_bpb = 1.1826.**

---

### What to do next

Only two actionable findings:

1. **Remove BigramHash** — saves 590K params for free. Those params can be redistributed or the model compresses smaller.

2. **Int5+Int6+zstd quantization with Late QAT** — the submission blocker. None of the architecture tweaks beat the control. The path forward is making the existing Exp 15 model fit in 16MB.

The current model (27M params) with BigramHash removed (26.5M params) at int5-MLP + int6-attn + zstd should fit in ~14.2MB. With sliding window stride=64 at eval, the expected submission BPP is **~1.17-1.19**.
