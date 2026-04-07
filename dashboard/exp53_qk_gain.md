# Exp 53: QK-Gain = 4.0

**Date**: 2026-04-07
**Change**: `QK_GAIN_INIT=4.0` (from 1.5)
**Status**: Smoke test confirmed, full training needed

---

## What QK-Gain Does

In self-attention, the model computes how much each token should attend to every other token:

```
scores = (Q @ K^T) / sqrt(head_dim) * qk_gain
attention = softmax(scores)
```

`qk_gain` is a learnable per-head scalar that controls attention sharpness. Higher values → more peaked attention (model focuses on fewer tokens). Lower values → more diffuse attention (spreads across many tokens).

Our current model initializes `qk_gain = 1.5`. The competition found that `qk_gain = 4.0` gives -0.004 BPB, validated across a 45-experiment sweep.

## Why This Should Work

With `qk_gain = 4.0`, from step 1 the model is forced to make sharper attention decisions. Instead of hedging across many tokens, it commits to attending to specific positions. Over thousands of training steps, this compounds into:

1. **More specialized attention heads** — each head learns to look for a specific pattern rather than averaging
2. **Better feature separation** — sharper attention means cleaner routing of information through the recurrence passes
3. **Stronger long-range dependencies** — diffuse attention dilutes long-range signals; sharp attention preserves them

This is a training-time change, not a post-training fix. That's why it works when post-training experiments (GPTQ bits, SCORE_CAP) failed — those can't change what the model learned.

## Smoke Test Results (700s, 1xH100)

| | val_bpb @ step 500 | Delta |
|-|-------------------|-------|
| Baseline (QK=1.5) | 1.4256 | — |
| **QK=4.0** | **1.4196** | **-0.006** |

Consistent advantage across all checkpoints (step 100 through 718). Full training curve in [exp53_54_smoke_test_results.md](exp53_54_smoke_test_results.md).

## Full Training Plan

Same config as exp40-D but with QK-Gain=4.0:

```bash
RUN_ID=exp53_qkgain4_full \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 NUM_LAYERS=7 MODEL_DIM=624 MLP_MULT=3 \
LAYER_SCHEDULE=0,1,2,3,4,3,4,3,4,5,6,5,6 \
XSA_LAST_N=7 TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=524288 \
MAX_WALLCLOCK_SECONDS=14400 \
WARMDOWN_ITERS=3000 \
QK_GAIN_INIT=4.0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

**Duration**: 240 min on 1xH100 (~14K steps)
**Cost**: ~$11
**Post-training**: GPTQ + SGD TTT eval (50K docs)

## Predictions

| Metric | Exp40-D (QK=1.5) | Predicted (QK=4.0) | Basis |
|--------|------------------|-------------------|-------|
| Raw val_bpb | 1.1466 | 1.140-1.144 | -0.006 at step 500, scales with more training |
| Post-GPTQ+TTT | 1.1172 | **1.111-1.115** | Same relative improvement |
| Gap to SOTA | 0.0025-0.0086 | **0.000-0.004** | Could match or beat merged SOTA |

## Results

*(To be filled after full training)*
