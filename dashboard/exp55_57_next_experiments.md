# Exp 55-57: Next Experiments Plan

**Current best**: val_bpb = 1.1159 (exp53, QK-Gain=4.0 + GPTQ + SGD TTT)
**SOTA**: 1.1086-1.1147
**Gap**: 0.0012-0.0073

---

## What Worked and Why

Only two types of changes have survived our full pipeline (train → GPTQ → TTT):

| Experiment | Type | BPB gain | Why it worked |
|-----------|------|----------|--------------|
| Exp 43 (GPTQ) | Post-training quantization | -0.0086 | Reduced information loss from rounding |
| Exp 53 (QK-Gain=4.0) | Training-time hyperparameter | -0.0013 | Changed training trajectory → weights that quantize 5x better |

**Failed**: Per-block GPTQ bits (-0.0002), SCORE_CAP (+0.0042 worse), batch=786K (+0.04 worse). All were post-training or infrastructure changes that didn't change what the model learns.

**Key insight**: At this stage, improvements must either (a) change the training trajectory to produce better/more quantizable weights, or (b) improve GPTQ itself. Everything else is already near-optimal.

---

## Experiment 55: Per-Pass Resid_Mix + QK-Gain=4.0

**Priority**: HIGH | **Cost**: ~$1 smoke, ~$11 full | **Expected**: -0.002 to -0.005

### What

Give each recurrence pass its own `resid_mix` blend ratio. Currently B3 uses the same x/x0 blend on all 3 passes, despite pass 1 reducing loss and pass 3 destabilizing. Add a `resid_mix_per_pass[pass_idx]` parameter — ~15K extra params total (negligible).

### Why

Directly motivated by exp47b data:

| Block | Pass 1 | Pass 2 | Pass 3 |
|-------|--------|--------|--------|
| B3 | Reduces loss (-1.36) | Builds features | **Destabilizes** (+0.24) |
| B4 | Reduces loss | Builds features | **Reconstructs** (-0.57) |

Forcing the same resid_mix across passes with such different roles is a real bottleneck. This is a training-time architecture change — follows the pattern.

### Smoke Test

500 steps with QK_GAIN_INIT=4.0, PER_PASS_RESID_MIX=1. Requires modifying `Block.forward()` to accept pass index. Two variants:
1. Per-pass resid_mix for B3-B4 only
2. Per-pass resid_mix for all shared blocks (B3-B4, B5-B6)

Compare val_bpb @ step 500 vs exp53 baseline (1.4196).

---

## Experiment 56: Extended Warmdown (5000 steps)

**Priority**: MEDIUM | **Cost**: ~$2 smoke, ~$11 full | **Expected**: -0.001 to -0.003

### What

Increase WARMDOWN_ITERS from 3000 to 5000. With ~14,700 total steps, warmdown goes from 21% to 34% of training. LR starts decaying earlier, SWA averages more checkpoints.

### Why

Exp53 showed the advantage over exp40-D **doubled during warmdown** (-0.001 → -0.002). The warmdown phase is where weights settle into stable minima and SWA kicks in. Longer warmdown = more time in this beneficial regime.

| Phase | Current (3000) | Extended (5000) |
|-------|---------------|-----------------|
| Constant LR | Steps 20-11,700 (79%) | Steps 20-9,700 (66%) |
| Warmdown | Steps 11,700-14,700 (21%) | Steps 9,700-14,700 (34%) |
| SWA checkpoints | ~15 | ~25 |

Top competition entries typically use 30-40% warmdown. We're currently at 21%.

### Smoke Test

2000-step runs comparing WARMDOWN_ITERS=600 (30%) vs 1000 (50%). Both with QK_GAIN_INIT=4.0.

**Risk**: Less time at full LR means fewer learning steps. If model isn't converged by step 9,700, earlier decay could hurt.

---

## Experiment 57: GPTQ with 128 Calibration Sequences

**Priority**: HIGH | **Cost**: ~$2, no retraining | **Expected**: -0.0005 to -0.001

### What

Double the GPTQ calibration data from 64 to 128 sequences (131K → 262K tokens). The Hessian H = X^T X / n gets more accurate, meaning better column compensation during quantization.

### Why

Uses existing exp53 checkpoint — no retraining. The cheapest possible experiment.

Current Hessian accuracy:
- 624-dim attention layers: ~337 samples per H element — decent
- 1280-dim MLP layers: ~80 samples per H element — **noisy**

MLP proj weights had the weakest GPTQ improvement (1.3-1.5x vs 2.0x for MLP gate). The noisy Hessian for wide matrices is likely the cause. Doubling calibration data helps most here.

### Eval Plan

Test 3 variants on existing checkpoint:
- N_CALIB=64 (baseline, already measured: 1.1159)
- N_CALIB=128
- N_CALIB=256

Quick 500-doc check first. If delta > 0.001, run full 50K eval.

**Risk**: Very low. If 64 samples was already sufficient, we've only lost $2 and 1 hour.

---

## Execution Order

| Step | Experiment | Why first |
|------|-----------|-----------|
| 1 | **Exp 57** (GPTQ 128 calib) | Cheapest, no retraining, 1 hour |
| 2 | **Exp 55** (per-pass resid_mix smoke) | Strongest theoretical motivation, 15 min |
| 3 | **Exp 56** (extended warmdown smoke) | Low risk schedule change, 30 min |
| 4 | Full training with winners | Combine everything that showed signal |

Can run exp 57 while preparing the code changes for exp 55. If exp 55 shows signal at 500 steps, combine exp 55 + exp 56 + QK-Gain=4.0 for one full training run.

---

## Combined Full Training (if all show signal)

```bash
RUN_ID=exp55_56_combined \
QK_GAIN_INIT=4.0 \
PER_PASS_RESID_MIX=1 \
WARMDOWN_ITERS=5000 \
# ... same base config as exp53 ...
```

Apply GPTQ with N_CALIB=128 (or 256) post-training.

**Target**: val_bpb < 1.110

## Predictions vs Actuals

| Experiment | Predicted | Actual | Correct? |
|-----------|-----------|--------|----------|
| Exp 55 (per-pass resid_mix) | -0.003 | — | — |
| Exp 56 (extended warmdown) | -0.002 | — | — |
| Exp 57 (GPTQ 128 calib) | -0.001 | — | — |
| Combined | -0.004 to -0.006 | — | — |
