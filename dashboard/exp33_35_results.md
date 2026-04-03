# Experiments 33-35: Base Model & TTT Optimization Results

**Baseline entering these experiments: val_bpb = 1.1429** (exp26 model + SGD all-weights TTT)

---

## Exp 33: Quantization, Softcap, LR Sweep, Epochs

**Date**: 2026-04-03 | **Hardware**: RunPod 1xH100 SXM | **Cost**: ~$0.50

### Goal
Test four independent levers for BPB improvement on 500 docs.

### Results

#### A. GPTQ-lite Quantization (pre-TTT, no training)

| Variant | BPB | Delta vs mixed quant |
|---------|-----|---------------------|
| Mixed quant (current) | 1.1905 | — |
| Unquantized (float32) | 1.1704 | -0.0201 |
| GPTQ-lite int6 (optimal clip) | 1.2476 | +0.0571 (MUCH WORSE) |
| GPTQ-lite int8 (optimal clip) | 1.1754 | -0.0151 |

**Verdict: FAILED.** Naive per-row percentile clipping makes int6 catastrophically worse. Real GPTQ uses second-order Hessian information and calibration data to minimize output error, not just weight reconstruction MSE. Our implementation only searched for the best clip percentile per row — fundamentally the wrong objective at low bit-widths.

**The unquantized result (1.1704) confirms 0.0201 BPB is lost to quantization** — this remains the single largest recoverable gap.

#### B. Softcap Sweep (pre-TTT, no training)

| Softcap | BPB |
|---------|-----|
| 15.0 | 2.019 |
| 20.0 | 1.488 |
| 25.0 | 1.247 |
| **30.0** | **1.191** |

**Verdict: 30.0 already optimal.** The model was trained with softcap=30; changing it at eval time breaks the calibrated output distribution. Lower softcaps dramatically clip the logit range, destroying the model's ability to make confident correct predictions.

**Lesson learned**: Eval-time hyperparameters can't be tuned independently of training — the model's weight landscape is shaped around the training hyperparameters.

#### D. SGD Learning Rate Sweep (with TTT, 500 docs)

| LR | BPB | Delta vs 0.002 |
|----|-----|----------------|
| 0.001 | 1.1681 | +0.0028 |
| 0.002 | 1.1653 | — |
| 0.003 | 1.1644 | -0.0009 |
| **0.005** | **1.1637** | **-0.0016** |

**Verdict: lr=0.005 is slightly better.** The model can handle more aggressive per-document adaptation than 0.002. But the gain is small (0.0016 on 500 docs).

#### E. Multiple TTT Epochs (with TTT, lr=0.002, 500 docs)

| Epochs | BPB | Time | Delta |
|--------|-----|------|-------|
| 1 | 1.1653 | 39s | — |
| 2 | 1.1645 | 72s | -0.0008 |
| 3 | 1.1640 | 105s | -0.0013 |

**Verdict: Marginal.** 3 epochs gives 0.0013 BPB for 3x the compute. Not worth it at lr=0.002. The model extracts most of the per-document signal in a single pass.

### Exp 33 Post-Mortem

**What we learned:**
1. Quantization is confirmed as the #1 gap (0.02 BPB), but naive GPTQ doesn't help — need proper implementation with calibration data and Hessian-based compensation
2. Softcap can't be tuned at eval time — model is calibrated to training settings
3. SGD lr and epochs have diminishing returns — the big win was switching from LoRA to all-weights, not tuning the optimizer

**What was wrong in our assumptions:**
- GPTQ: Assumed per-row clip optimization would help. Wrong — the objective (minimize weight MSE) doesn't align with the goal (minimize prediction error)
- Softcap: Assumed eval-time tuning is independent. Wrong — training and eval settings are coupled
- LR/epochs: Correctly identified as marginal — confirmed the SGD recipe is already near-optimal

---

## Exp 34: SGD Combo Tests

**Date**: 2026-04-03 | **Hardware**: RunPod 1xH100 SXM | **Cost**: ~$0.50

### Goal
Test combinations of improvements: lr, chunk size, gradient clipping, epochs.

### Results (500 docs)

| Variant | BPB | Time | vs baseline |
|---------|-----|------|-------------|
| F: lr=0.005 chunk=1024 | 1.1637 | 39s | — |
| G: lr=0.005 chunk=2048 | 1.1634 | 27s | -0.0003 |
| H: lr=0.005 chunk=1024 clip=1 | 1.1633 | 37s | -0.0004 |
| I: lr=0.005 2ep chunk=1024 | 1.1630 | 67s | -0.0007 |
| J: lr=0.005 chunk=2048 clip=1 | 1.1632 | 28s | -0.0005 |
| K: lr=0.005 2ep chunk=2048 clip=1 | 1.1627 | 51s | -0.0010 |
| **L: lr=0.01 chunk=2048 clip=1** | **1.1626** | **28s** | **-0.0011** |

### Exp 34 Post-Mortem

**Verdict: Differences are within noise (0.001 BPB range).** All variants perform nearly identically. The best (L: lr=0.01, chunk=2048, clip=1) is only 0.0011 better than the simplest (F: lr=0.005, chunk=1024).

**Key finding**: Chunk size 2048 is slightly faster than 1024 (27s vs 39s) because there's only 1 forward/backward pass per doc instead of 2. Same BPB, less time.

**What we learned:**
1. SGD all-weights TTT is robust to hyperparameter variation — all combos give similar results
2. Gradient clipping doesn't help meaningfully (the small lr already prevents destructive updates)
3. 2 epochs barely help; the model gets most of the signal in 1 pass
4. The big win was the switch from LoRA to all-weights, not optimizer tuning

---

## Exp 35: Adaptive Per-Document TTT

**Date**: 2026-04-03 | **Hardware**: RunPod 1xH100 SXM | **Cost**: ~$0.50

### Goal
Address the hard token problem by adapting TTT hyperparameters per document. Based on analysis showing short docs are harder (mean_loss 2.09 vs 1.92) and hard tokens are mostly first-time subword predictions.

### Variants

| Variant | Idea |
|---------|------|
| A: Fixed lr=0.005 | Baseline control |
| B: Adaptive LR | Double/halve lr based on first-step loss drop |
| C: Adaptive epochs | Add epoch if training loss >2.5 |
| D: LR grid search | Try 3 LRs per doc, pick best |
| E: Short-doc boost | Higher lr (0.01) for docs <500 tokens |

### Results (500 docs)

| Variant | BPB | Time | vs baseline |
|---------|-----|------|-------------|
| A: Fixed lr=0.005 | 1.1637 | 39s | — |
| B: Adaptive LR | 1.1668 | 38s | +0.0031 (WORSE) |
| C: Adaptive epochs | 1.1636 | 40s | -0.0001 |
| D: LR grid search | **1.1631** | 108s | -0.0006 |
| E: Short-doc boost | 1.1636 | 36s | -0.0001 |

### Exp 35 Post-Mortem

**Verdict: Per-document adaptation doesn't help.**

**What we learned:**
1. **Adaptive LR hurts** — the heuristic (measure loss drop, adjust lr) makes worse choices than fixed lr. The first-step loss drop isn't a reliable signal for optimal lr.
2. **Adaptive epochs don't help** — docs with high training loss (>2.5) don't benefit from more training. The model extracts what it can in 1 pass; more passes don't fix fundamentally hard predictions.
3. **LR grid search is technically best but 3x slower** — trying 3 LRs per doc gives 0.0006 BPB for 3x compute. Not practical.
4. **Short-doc boost doesn't help** — short docs are harder due to less context, not due to suboptimal lr. More aggressive adaptation can't compensate for missing information.

**Root cause**: The hard tokens (subword splits, rare words) are hard because of **model capacity and tokenizer limitations**, not because of TTT hyperparameters. No amount of per-document tuning can fix a model that doesn't know the word "rehearsal" splits as "re-her-sal" — that's baked into the tokenizer and the model's learned vocabulary.

---

## Overall Session Summary (Exp 28-35)

### What worked
| Change | BPB improvement | Cost |
|--------|----------------|------|
| SGD all-weights TTT (exp 28) | **-0.0111** | $3 eval |
| SGD lr=0.005 (exp 33D) | -0.0016 | free |

### What didn't work
| Change | Result | Why |
|--------|--------|-----|
| Higher LoRA rank (exp 31) | +0.0003 to +0.0009 (worse) | Bottleneck was which weights, not rank |
| Naive GPTQ-lite (exp 33A) | +0.057 (much worse) | Wrong optimization objective |
| Softcap tuning (exp 33B) | +0.83 (catastrophic) | Can't decouple from training |
| Multiple epochs (exp 33E) | -0.0013 for 3x cost | Diminishing returns |
| SGD combos (exp 34) | -0.001 max | Already near-optimal |
| Adaptive TTT (exp 35) | -0.0006 max for 3x cost | Hard tokens are a model problem, not TTT |

### Key insight
**Removing constraints >> adding complexity.** The single best change (LoRA → SGD all-weights) gave 10x more improvement than all tuning experiments combined. The remaining gap to SOTA (~0.03 BPB) is in the base model: quantization loss (0.02), training convergence (0.005-0.008), and model capacity.

### Current best
**val_bpb = 1.1429** (exp26 XSA-all + EMA model, SGD all-weights TTT lr=0.002)

### Remaining levers (not yet tried)
1. **QAT with STE gradients** — train model to be robust to quantization (0.01-0.015 BPB potential)
2. **Sliding window eval** — score with overlapping windows stride=64 (0.005-0.01 BPB, no retrain)
3. **Depth recurrence** — reuse layers for more effective depth (0.005-0.01 BPB, retrain)
4. **Longer training** — extend wallclock beyond 80 min (0.005-0.008 BPB)
