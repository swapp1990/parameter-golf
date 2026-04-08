# Exp 55: Per-Pass Resid_Mix + QK-Gain=4.0

**Date**: 2026-04-08
**Change**: Per-pass `resid_mix` for shared blocks (B3-B4, B5-B6) + QK_GAIN_INIT=4.0
**Status**: Not started
**Priority**: HIGH — directly motivated by exp47b data

---

## What Per-Pass Resid_Mix Does

In each block, `resid_mix` controls the blend between the running hidden state `x` and the original embedding `x0`:

```python
# Current: same blend for every pass through a shared block
mix = self.resid_mix  # shape [2, dim] — one for all passes
x = mix[0] * x + mix[1] * x0
```

With depth recurrence, blocks 3-4 are used 3 times and blocks 5-6 are used 2 times. Each pass has a qualitatively different role (from exp47b):

| Block | Pass 1 | Pass 2 | Pass 3 |
|-------|--------|--------|--------|
| B3 | Reduces loss (-1.36 BPB) | Builds features | **Destabilizes** (+0.24) |
| B4 | Reduces loss | Builds features | **Reconstructs** (-0.57) |

Forcing the same `resid_mix` on all passes is a bottleneck — B3 pass 3 needs a very different x/x0 blend than B3 pass 1.

### The Change

```python
# New: separate blend per pass
self.resid_mix_per_pass = nn.Parameter(
    torch.stack([self.resid_mix.clone() for _ in range(n_passes)])
)  # shape [n_passes, 2, dim]

# In forward, index by pass number:
mix = self.resid_mix_per_pass[pass_idx]
x = mix[0] * x + mix[1] * x0
```

This adds ~3,744 parameters per shared block (3 passes × 2 × 624 dim). Total across B3-B6: ~15K params. Negligible impact on model size.

## Why This Should Work

1. **Empirically motivated**: Exp47b showed passes have different roles. The resid_mix bottleneck is real — B3 uses the same blend when it reduces loss (pass 1) and when it destabilizes (pass 3).

2. **Training-time change**: Following the pattern from exp43/exp53 — only changes to what the model learns survive the full pipeline.

3. **Minimal complexity**: Just indexing into a per-pass tensor. No new layer types, no LoRA adapters, no hyperparameter search.

4. **Combines with QK-Gain=4.0**: Train from scratch with both changes. QK-Gain gives sharper attention for feature routing; per-pass resid_mix lets each pass control its own x/x0 blend.

## Smoke Test Plan (500 steps, ~$1)

```bash
RUN_ID=exp55_per_pass_resid_mix \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 NUM_LAYERS=7 MODEL_DIM=624 MLP_MULT=3 \
LAYER_SCHEDULE=0,1,2,3,4,3,4,3,4,5,6,5,6 \
XSA_LAST_N=7 TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=524288 \
MAX_WALLCLOCK_SECONDS=600 \
WARMDOWN_ITERS=100 \
QK_GAIN_INIT=4.0 \
PER_PASS_RESID_MIX=1 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

**Requires**: Modifying `Block.forward()` to accept pass index and use per-pass resid_mix.

### Variants to test at 500 steps:
1. Per-pass resid_mix for B3-B4 only (3 passes × 2 blocks)
2. Per-pass resid_mix for all shared blocks (B3-B4 and B5-B6)

## Predictions

| Metric | Exp53 (QK=4.0) | Predicted (+ per-pass resid_mix) | Basis |
|--------|----------------|----------------------------------|-------|
| val_bpb @ step 500 | 1.4196 | 1.410-1.415 | -0.005 to -0.010 from removing bottleneck |
| Full training raw | 1.1458 | 1.142-1.145 | Scales with more steps |
| Post-GPTQ+TTT | 1.1159 | **1.112-1.115** | Same relative improvement |

## Results

*(To be filled after smoke test)*
