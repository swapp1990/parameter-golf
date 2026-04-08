# Exp 56: Extended Warmdown (5000 steps)

**Date**: 2026-04-08
**Change**: WARMDOWN_ITERS=5000 (from 3000) + QK_GAIN_INIT=4.0
**Status**: Not started
**Priority**: MEDIUM — low risk, motivated by exp53 warmdown amplification

---

## What Extended Warmdown Does

During training, the learning rate follows a schedule:
1. **Warmup** (20 steps): LR ramps from 0 to max
2. **Constant** (steps 20 to N-WARMDOWN): LR stays at max
3. **Warmdown** (last WARMDOWN_ITERS steps): LR decays from max to 0

With WARMDOWN_ITERS=3000 and ~14,700 total steps:
- Constant phase: steps 20-11,700 (79% of training)
- Warmdown phase: steps 11,700-14,700 (21% of training)

With WARMDOWN_ITERS=5000:
- Constant phase: steps 20-9,700 (66% of training)
- Warmdown phase: steps 9,700-14,700 (34% of training)

The warmdown phase is when SWA (Stochastic Weight Averaging) kicks in, averaging multiple checkpoints. Longer warmdown = more time for weights to settle into a stable minimum and more checkpoints averaged.

## Why This Should Work

### Evidence from Exp53

Exp53 (QK-Gain=4.0) showed a clear pattern — the advantage over exp40-D **widened during warmdown**:

| Training Phase | Delta (exp53 vs exp40-D) |
|---------------|-------------------------|
| Steps 1000-10000 (constant LR) | -0.001 BPB |
| Steps 11000-14000 (warmdown) | **-0.002 BPB** |

The warmdown phase doubled the advantage. This suggests our model benefits disproportionately from the LR decay period. Extending warmdown gives more time in this beneficial regime.

### Theoretical Basis

1. **SWA averaging**: More warmdown steps = more checkpoints averaged. SWA finds wider minima that generalize better and are more robust to quantization.

2. **Gradual refinement**: During warmdown, the model fine-tunes weights without large perturbations. This is especially important for quantization — smaller weight changes mean the final weight distribution is tighter, with fewer outliers that cause large quantization errors.

3. **Competition practice**: Top competition entries typically use aggressive warmdown schedules (30-40% of training).

## Smoke Test Plan (2000 steps, ~$2)

Two variants to compare at step 2000:

```bash
# Variant A: Current warmdown (3000 iters, starts at step ~-1000 in a 2000-step run)
# Actually for 2000 steps, warmdown_iters=600 (30%) vs 1000 (50%)

# Baseline: 30% warmdown
RUN_ID=exp56_warmdown_30pct \
WARMDOWN_ITERS=600 \
MAX_WALLCLOCK_SECONDS=2000 \
QK_GAIN_INIT=4.0 \
# ... same config as exp53 ...

# Extended: 50% warmdown
RUN_ID=exp56_warmdown_50pct \
WARMDOWN_ITERS=1000 \
MAX_WALLCLOCK_SECONDS=2000 \
QK_GAIN_INIT=4.0 \
# ... same config as exp53 ...
```

### Full Training Config (if smoke test positive)

```bash
RUN_ID=exp56_warmdown5000_full \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 NUM_LAYERS=7 MODEL_DIM=624 MLP_MULT=3 \
LAYER_SCHEDULE=0,1,2,3,4,3,4,3,4,5,6,5,6 \
XSA_LAST_N=7 TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=524288 \
MAX_WALLCLOCK_SECONDS=14400 \
WARMDOWN_ITERS=5000 \
QK_GAIN_INIT=4.0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

**Duration**: 240 min on 1xH100 (~14.7K steps, warmdown starts at step ~9,700)
**Cost**: ~$11

## Predictions

| Metric | Exp53 (warmdown=3000) | Predicted (warmdown=5000) | Basis |
|--------|----------------------|---------------------------|-------|
| Raw val_bpb | 1.1458 | 1.143-1.145 | Wider SWA window helps |
| Post-GPTQ+TTT | 1.1159 | **1.113-1.115** | Better quantization from tighter weights |
| SWA checkpoints | 15 | ~25 | More averaging |

**Risk**: Extended warmdown means less time at full learning rate. If the model isn't converged enough by step 9,700, the earlier decay could hurt. The 2000-step smoke test will reveal this.

## Results

*(To be filled after smoke test)*
