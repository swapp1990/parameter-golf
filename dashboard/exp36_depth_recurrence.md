# Exp 36: Depth Recurrence

**Goal**: Reuse layer weights to get more effective depth without storing more parameters. Use saved parameter budget for wider dimensions or better quantization.

## Background

Our model has 11 unique layers = 26.5M params. Each block is ~2.4M params (attn + MLP). The 16MB artifact stores all 11 unique blocks.

**Depth recurrence** = reuse the same block weights multiple times in a forward pass. A model with 6 unique blocks looped 2x has 12 effective layers but only stores 6 blocks of weights.

Top submissions mention depth recurrence as a key technique. The intuition: more depth helps with complex token predictions (our hard token problem), and weight sharing is free at inference time.

## Design Options

### Option A: Simple Loop (all layers shared)
- 6 unique blocks, each used 2x = 12 effective layers
- Params: ~14.4M (vs 26.5M current) — saves 12.1M params
- Use savings to increase model_dim from 512 to ~640

### Option B: Shared Middle, Unique Endpoints
- Blocks 0-2: unique (input processing)
- Blocks 3-5: shared, looped 2x = 6 effective middle layers
- Blocks 6-8: unique (output processing)
- Total: 9 unique blocks, 12 effective layers
- Params: ~21.6M — saves 4.8M params

### Option C: Universal Transformer Style
- 1 shared block, looped 11x = 11 effective layers
- Params: ~4.8M — massive savings
- Risk: too constrained, each "layer" can't specialize
- Use savings to increase model_dim from 512 to ~900+

### Recommended: Option B (safest)

Unique endpoints handle input/output specialization (embedding projection, final prediction). Shared middle layers handle the bulk of contextual processing — these layers do similar work anyway (attention + MLP transform), so sharing weights should work.

## Implementation

Modify GPT.__init__ and forward:

```python
# Option B: 9 unique blocks, middle 3 shared (looped 2x)
# blocks[0..2] = unique encoder
# blocks[3..5] = shared middle (used twice)
# blocks[6..8] = unique decoder

# Forward:
# Pass 1: blocks 0,1,2,3,4,5,6,7,8 (normal)
# Pass 2: blocks 0,1,2,3,4,5,3,4,5,6,7,8 (middle repeated)
```

Need to handle:
- U-Net skip connections with different layer count
- XSA config per effective layer position
- RoPE positions (should be same since sequence position doesn't change)

## Quick Test Plan

300s smoke test on H100:
1. Implement Option B (shared middle)
2. Train 300s, compare val_loss at step 500 vs current baseline
3. If promising, test Option A (wider model with full sharing)

## Variants to Test (500 docs, quick eval)

| Variant | Unique blocks | Loops | Effective layers | model_dim | Params |
|---------|--------------|-------|-----------------|-----------|--------|
| Current | 11 | 1 | 11 | 512 | 26.5M |
| B1 | 9 (shared middle 3) | 2 | 12 | 512 | ~21.6M |
| B2 | 9 (shared middle 3) | 2 | 12 | 576 | ~26.5M (same budget, wider) |
| A1 | 6 (all shared) | 2 | 12 | 512 | ~14.4M |
| A2 | 6 (all shared) | 2 | 12 | 640 | ~22M |

## Expected Results

Conservative: depth recurrence with same param budget gives -0.003 to -0.005 BPB pre-TTT.
Optimistic: wider model + depth recurrence gives -0.005 to -0.01 BPB pre-TTT.

## Risks

1. Shared layers may not specialize enough → worse than unique layers
2. Training instability with weight sharing (gradients from multiple positions)
3. U-Net skip connections need rethinking for shared layers
4. Quantization of shared layers — stored once but used multiple times, errors compound

## Results

### Smoke Test (500 steps, H100)

| Variant | Params | Val Loss @500 | vs BASE |
|---------|--------|---------------|---------|
| BASE: 11 unique, dim=512 | 26.5M | 2.8079 | — |
| B1: 9 unique (mid shared), dim=512 | 21.8M | 2.7876 | -0.0203 (better, fewer params!) |
| **B2: 9 unique (mid shared), dim=576** | 27.5M | **2.7827** | **-0.0252 (best)** |
| A1: 6 unique (all shared), dim=512 | 14.7M | 2.9013 | +0.0934 (worse) |
| A2: 6 unique (all shared), dim=640 | 22.8M | 2.8899 | +0.0820 (worse) |

Full sharing (A1/A2) hurts badly. Partial sharing with unique endpoints (B1/B2) works.

### Full Training: dim=576 (FAILED — over 16MB)

- Mixed quant artifact: 16,326,579 bytes — **OVER 16MB limit by 327KB**
- val_bpb=1.1712 raw, 1.1760 int8 roundtrip
- Wasted ~$5.40 on training. Should have checked artifact size before training.

**Lesson**: Added pre-training size check to train_gpt.py (`SKIP_SIZE_CHECK=1` to override).

### Full Training: dim=560 (SUCCESS)

**Config**: 9 unique blocks, middle 3 shared (looped 2x = 12 effective layers), dim=560, XSA all, EMA 0.997, 120 min wallclock.

| Metric | Exp26 (dim=512, 11 unique) | Exp36 (dim=560, 9+recurrence) |
|--------|---------------------------|-------------------------------|
| Params | 26,502,232 | 26,483,032 |
| Steps | 8,165 | 8,366 |
| Raw val_bpb | 1.1682 | **1.1594** (-0.0088) |
| Int8 roundtrip | 1.1731 | **1.1639** (-0.0092) |
| Mixed quant size | 15.08 MB | 15.88 MB (FITS) |

### SGD All-Weights TTT Eval (50K docs)

| Metric | Exp26 + SGD TTT | Exp36 + SGD TTT |
|--------|----------------|-----------------|
| val_bpb | 1.1429 | **1.1290** |
| Delta | — | **-0.0139** |
| Time | 3859s | 3990s |

### Summary

**New personal best: val_bpb = 1.1290**

Improvement breakdown from original submission (1.1573):
- XSA-all + EMA (exp26): -0.0033
- SGD all-weights TTT (exp28): -0.0111
- Depth recurrence + wider dim (exp36): -0.0139
- **Total: -0.0283**
