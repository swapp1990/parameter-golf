# Exp 57: GPTQ with 128 Calibration Sequences

**Date**: 2026-04-08
**Change**: N_CALIB=128 (from 64) for GPTQ Hessian estimation
**Status**: Not started
**Priority**: HIGH — cheapest experiment, no retraining needed

---

## What This Changes

GPTQ quantization works by:
1. Running calibration data through the model to capture activation statistics
2. Computing the Hessian H = X^T X / n for each weight matrix
3. Using H to determine how to compensate for rounding errors column-by-column

Currently we use **64 calibration sequences** of length 2048 (131K tokens). The Hessian estimate gets more accurate with more data — 128 sequences (262K tokens) should give a cleaner estimate.

```python
# Current
N_CALIB = 64
cd = [random_seq() for _ in range(N_CALIB)]  # 131K tokens
H = (X.T @ X) / X.shape[0]  # noisy estimate

# New
N_CALIB = 128
cd = [random_seq() for _ in range(N_CALIB)]  # 262K tokens
H = (X.T @ X) / X.shape[0]  # cleaner estimate
```

## Why This Should Work

### The Hessian Accuracy Argument

The Hessian H tells GPTQ which weight columns matter most (high-variance activations) and how columns correlate with each other (for compensation). With 64 sequences:

- Each row of X has 64 × 2048 = 131K samples
- For a 624-dim layer, we're estimating a 624×624 matrix from 131K samples
- That's ~337 samples per matrix element — decent but noisy
- For MLP layers (1280-dim), it's 1280×1280 from 131K = ~80 samples per element — quite noisy

Doubling to 128 sequences:
- 262K samples, ~160 per element for MLP layers
- The off-diagonal elements of H (cross-column correlations) become more reliable
- Better off-diagonal H → better compensation when quantizing each column

### Where This Helps Most

From exp43's per-layer analysis:
- **MLP proj (int5, wide)**: Largest absolute quantization error, only 1.3-1.5x GPTQ improvement. These wide matrices (1280→624) have the noisiest Hessian with 64 samples. More calibration data helps most here.
- **Attention proj (int6)**: 1.5-2.0x GPTQ improvement. Already benefiting well, but more calibration can squeeze out more.

### No Retraining Required

This uses the existing exp53 checkpoint (final_model.pt). Just re-run GPTQ with N_CALIB=128 and evaluate. Total cost: ~$2 for 1 hour of H100 time (GPTQ application + 50K doc TTT eval).

## Eval Plan

```python
# Modify exp53_gptq_ttt_eval.py:
N_CALIB = 128  # was 64

# Also test N_CALIB=256 while we're at it
# Three variants:
#   A: N_CALIB=64  (current baseline, already measured: 1.1159)
#   B: N_CALIB=128
#   C: N_CALIB=256
```

Run on existing exp53 checkpoint. Compare GPTQ output MSE per layer AND final val_bpb after TTT.

### Quick Validation (500 docs, ~5 min each)

Before committing to 50K-doc eval, run 500-doc quick check:
1. If delta < 0.0005 on 500 docs, likely not worth full eval
2. If delta > 0.001 on 500 docs, run full 50K eval

**Lesson from exp49**: 500-doc results can mislead. But for GPTQ calibration changes (not scoring changes), the signal should be more stable because it's a deterministic model change, not a metric change.

## Predictions

| Variant | Expected val_bpb | Delta vs N=64 | Basis |
|---------|-----------------|---------------|-------|
| N=64 (baseline) | 1.1159 | — | Measured |
| N=128 | **1.1149-1.1155** | -0.0004 to -0.001 | Cleaner Hessian, better MLP compensation |
| N=256 | **1.1147-1.1155** | -0.0004 to -0.001 | Diminishing returns |

**Risk**: Very low. If more calibration doesn't help, we've only spent $2 and 1 hour. The Hessian with 64 samples may already be sufficient for our 624-dim model.

## Artifact Size Impact

More calibration changes the GPTQ rounding decisions but not the quantization bit widths. The compressed artifact size should be similar (±10KB). No risk of exceeding 16MB.

## Results

*(To be filled after eval)*
