# Experiments 28-32: Next Steps to Improve BPB

**Current best: val_bpb = 1.1540** (exp26, XSA-all + EMA + LoRA TTT)
**Current SOTA: 1.1086-1.1147 BPB**
**Gap to close: ~0.04 BPB**

Based on the XSA impact analysis (see `exp26_xsa_impact_report.md`), we identified where our model is strong and weak. These experiments target the weaknesses.

---

## Exp 28: SGD TTT (Top Submission Recipe)

**Expected gain: -0.003 to -0.01 BPB | Retrain: NO | Time: ~2 hrs eval**

### What

Replace our Adam-based LoRA TTT with the top submission's proven recipe:
- SGD(lr=0.002, momentum=0.9) instead of Adam(lr=0.05)
- ALL model weights unfrozen, not just LoRA on Q/V
- 3 epochs per 32K-token chunk instead of 1 epoch per 1024-token chunk
- Cosine LR decay within each chunk
- Gradient clipping at 1.0

### Why

This is the highest-priority experiment because:
1. No retraining needed — runs on our existing exp26 checkpoint
2. The top submission's TTT recipe is proven to work on a similar architecture
3. Our current TTT recipe (Adam, LoRA-only, 1 epoch, 1024-token chunks) was never optimized — we just used what worked first
4. The top submission gets -0.0025 BPB from TTT on a 1.1218 pre-TTT baseline; we get -0.0374 from TTT on a 1.1877 baseline. Their TTT is more efficient per-token despite less total improvement, suggesting a better optimization recipe

### Quick Eval Method

1. Modify TTT eval script to use SGD instead of Adam
2. First test: 500 docs with both recipes, compare BPB
3. If SGD shows improvement, run full 50K docs
4. **Time: ~30 min quick test, ~2 hrs full eval on H100**

### Risk

- SGD hyperparameters may need tuning for our model (different from top submission's model)
- Unfreezing all weights on a quantized model may cause instability (int5/int6 weights being modified)
- 3 epochs might overfit on short documents

---

## Exp 29: Scaled XSA (Partial Self-Value Removal)

**Expected gain: -0.001 to -0.003 BPB | Retrain: YES (300s smoke test) | Time: 10 min**

### What

The XSA analysis showed XSA-all hurts on mid-range repetitions (-0.006 avg delta for tokens 50-200 back) and confident predictions (-0.005 for low entropy). Full self-value removal kills a natural "copy" mechanism. Instead of full removal, use partial:

```python
# Current: full removal
y = y - (dot_yv / dot_vv) * v

# New: partial removal
y = y - alpha * (dot_yv / dot_vv) * v
```

### Variants to test

- alpha = 0.3 (keep 70% of self-value)
- alpha = 0.5 (keep 50%)
- alpha = 0.7 (keep 30%)
- Per-layer learnable alpha (if fixed alpha helps)

### Quick Eval Method

1. Modify CausalSelfAttention to accept xsa_alpha parameter
2. Train 300s with each alpha value
3. Compare val_loss at step 500 vs XSA-all baseline
4. **Time: ~10 min per variant on H100**

---

## Exp 30: Selective XSA Layer Pattern

**Expected gain: -0.001 to -0.003 BPB | Retrain: YES (300s smoke test) | Time: 10 min**

### What

XSA-all helps with novel tokens (cross-token feature building in early layers) but hurts with repetitions (copy mechanism in middle layers). Try patterns that preserve both:

### Variants to test

- Pattern A: XSA on layers 0-3 and 9-10, standard attention on 4-8 (protect middle "copy" layers)
- Pattern B: XSA on even layers only (0, 2, 4, 6, 8, 10)
- Pattern C: XSA on all layers except 5 and 6 (U-Net skip connection layers)

### Why these patterns

- The XSA analysis showed the improvement is strongest for first-time tokens (contextual understanding), which benefits from early-layer XSA
- Mid-range repetitions (50-200 tokens back) are hurt, suggesting middle layers need self-value for the copy mechanism
- Late layers benefit from XSA for final reasoning (already proven in our XSA-last4 baseline)

### Quick Eval Method

1. Modify GPT constructor to accept a list of XSA layer indices
2. Train 300s with each pattern
3. Compare val_loss at step 500
4. **Time: ~10 min per variant on H100**

---

## Exp 31: Higher LoRA Rank

**Expected gain: -0.001 to -0.002 BPB | Retrain: NO | Time: ~2 hrs eval**

### What

Current LoRA: rank=8 on Q/V projections only (157,696 params). The XSA-all model has richer cross-token features in all layers, which means LoRA has more to work with.

### Variants to test

- Rank 16 on Q/V (2x params)
- Rank 32 on Q/V (4x params)
- Rank 8 on Q/K/V/proj (all 4 attention matrices, 2x params)
- Rank 16 on Q/K/V/proj (4x params)

### Why

The XSA analysis showed TTT is slightly more effective on the XSA-all model (-0.0374 vs -0.0341). Higher rank LoRA might unlock even more TTT benefit by giving more capacity to adapt per-document.

### Quick Eval Method

1. Modify TTT eval script to accept rank and target modules as parameters
2. Test on 500 docs with each variant
3. If improvement found, run full 50K docs
4. **Time: ~30 min quick test per variant**

---

## Exp 32: Attention Pattern Analysis

**Expected gain: Diagnostic only | Retrain: NO | Time: ~10 min**

### What

Extract and compare attention maps from both models (XSA-last4 vs XSA-all) on the same documents to understand WHY XSA-all helps where it does.

### Specifically

- For the top-50 improved tokens (from xsa_comparison.json), save attention weights from all 11 layers and 8 heads
- For the top-50 regression tokens, do the same
- Compare:
  - Which positions does each head attend to? Do XSA-all heads attend to more distant positions?
  - Attention entropy: does XSA-all produce sharper (more focused) or more diffuse attention?
  - Do early layers (0-3) in XSA-all attend differently than in XSA-last4? (They should — XSA-last4 uses standard attention here)
  - Are there "dead" heads in either model?

### Why

This tells us whether the improvement comes from:
- **Better attention routing** (XSA forces heads to look at other tokens → they learn to find relevant ones)
- **Better feature quality** (XSA forces representations to encode cross-token info → features are richer even if attention patterns are similar)

If attention patterns are similar → focus future work on feature quality (MLP changes, normalization, etc.)
If attention patterns are very different → focus on attention mechanisms (more heads, different routing, etc.)

### Method

1. Modify forward pass to save attention weights (before softmax or after)
2. Run both models on 20 docs, save attention for top-improved and top-regressed tokens
3. Generate visualization (heatmaps or summary stats)
4. **Time: ~10 min on H100**

---

## Predictions

| # | Experiment | Predicted BPB | Expected Gain | Actual BPB | Actual Time | Verdict |
|---|-----------|---------------|---------------|------------|-------------|---------|
| 28 | SGD TTT | 1.1490 | -0.005 | **1.1429** | 65 min | Beat prediction by 0.006 |
| 29 | Scaled XSA | 1.1520 | -0.002 | Not run | — | Skipped (SGD win too big) |
| 30 | Selective XSA | 1.1530 | -0.001 | Not run | — | Skipped (SGD win too big) |
| 31 | Higher LoRA rank | 1.1525 | -0.0015 | **1.1795** (r16) | 36s/500docs | WORSE than baseline |
| 32 | Attention analysis | 1.1540 | diagnostic | Not run | — | Deprioritized |

Predicted BPB = what we expect if the experiment works (on 500-doc quick test, extrapolated).

## Execution Plan (Batched)

Run quick tests for all experiments on a single pod session, then only do full 50K eval for the winners. This cuts cost from ~$27 to ~$7.

### Phase 1: Quick Tests (single H100 pod, ~30 min, ~$1.35)

All on the same pod, one after another:

| Order | Experiment | What to run | Time |
|-------|-----------|-------------|------|
| 1 | Exp 28: SGD TTT | 500-doc TTT eval with SGD recipe on exp26 checkpoint | ~5 min |
| 2 | Exp 31: Higher LoRA rank | 500-doc TTT eval with rank=16 and rank=32 | ~10 min |
| 3 | Exp 32: Attention analysis | Extract attention maps on 20 docs | ~5 min |
| 4 | Exp 29: Scaled XSA | 300s smoke test training (alpha=0.3, 0.5, 0.7) | ~5 min |
| 5 | Exp 30: Selective XSA | 300s smoke test training (3 patterns) | ~5 min |

No-retrain experiments first (28, 31, 32), then retrain smoke tests (29, 30).

### Phase 2: Full Eval (same or new pod, ~2 hrs, ~$5.40)

Only for experiments that showed improvement in Phase 1:
- Full 50K-doc TTT eval for the best 1-2 TTT variants (Exp 28 and/or 31)
- If Exp 29/30 smoke tests are promising, full 80-min H100 training + quantization + 50K TTT eval

### Phase 3: Combine (if multiple experiments help)

If both TTT improvements (Exp 28) and architecture improvements (Exp 29/30) work:
- Train new model with best XSA variant
- Eval with best TTT recipe
- Final submission run

### Cost Summary

| Phase | Time | Cost |
|-------|------|------|
| Phase 1: Quick tests | ~30 min | ~$1.35 |
| Phase 2: Full eval (1-2 winners) | ~2 hrs | ~$5.40 |
| Phase 3: Combined run (if needed) | ~2.5 hrs | ~$6.75 |
| **Total** | **~5 hrs** | **~$13.50** |

Phase 3 only happens if both TTT and architecture changes show improvement. Worst case (nothing works): $1.35 spent, 30 min wasted.

**Optimistic target: val_bpb < 1.145** (from 1.1540 with better TTT + refined XSA)
**Conservative target: val_bpb < 1.150** (at least one experiment helps)

## Results

### Exp 28: SGD All-Weights TTT — SUCCESS

**Quick test (500 docs):**

| Variant | BPB | Delta vs baseline |
|---------|-----|-------------------|
| A: SGD all-weights (lr=0.002, m=0.9) | 1.1652 | **-0.0139** |
| B: SGD LoRA r8 | 1.1905 | +0.0113 |
| C: Adam LoRA r8 (baseline) | 1.1792 | — |

**Full eval (50K docs):** val_bpb = **1.1429** (delta vs LoRA TTT: -0.0111)

SGD with all weights unfrozen is massively better than any LoRA variant. The improvement comes from adapting ALL weights (MLP, K, Proj, embeddings) not just Q/V.

### Exp 29/30: Scaled/Selective XSA — SKIPPED

Deprioritized after SGD TTT gave -0.011 for free. The XSA architecture changes require retraining (~$6 each) for an expected -0.001 to -0.003 gain. Not worth it given the SGD result.

### Exp 31: Higher LoRA Rank — FAILED

| Variant | BPB | Delta |
|---------|-----|-------|
| Adam LoRA r8 (baseline) | 1.1792 | — |
| Adam LoRA r16 | 1.1795 | +0.0003 |
| Adam LoRA r32 | 1.1801 | +0.0009 |

Higher rank doesn't help — the bottleneck was never LoRA's expressiveness, it was that LoRA only touches Q/V.

### Exp 32: Attention Analysis — DEPRIORITIZED

Not run. The XSA comparison data (xsa_comparison.json) already provided sufficient insight. The main finding — XSA-all helps on never-seen tokens and hurts on mid-range repetitions — was enough to inform decisions.

### Prediction Accuracy

- Exp 28: Predicted 1.1490, actual **1.1429** — underestimated by 0.006 (too conservative)
- Exp 31: Predicted 1.1525 (improvement), actual **1.1795** (regression) — completely wrong direction
- Lesson: Overvalued targeted complexity (LoRA rank), undervalued brute-force simplicity (SGD all-weights)
