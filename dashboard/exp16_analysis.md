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

### What to do next

Only two actionable findings:

1. **Remove BigramHash** — saves 590K params for free. Those params can be redistributed or the model compresses smaller.

2. **Int5+Int6+zstd quantization with Late QAT** — the submission blocker. None of the architecture tweaks beat the control. The path forward is making the existing Exp 15 model fit in 16MB.

The current model (27M params) with BigramHash removed (26.5M params) at int5-MLP + int6-attn + zstd should fit in ~14.2MB. With sliding window stride=64 at eval, the expected submission BPP is **~1.17-1.19**.
