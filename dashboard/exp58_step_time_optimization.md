# Exp 58: Step Time Optimization for 10-Min Track

**Goal**: Reduce step time on 8xH100 from 126ms to <90ms → more steps in 600s → better val_bpb

---

## The Problem

| Config | ms/step | Steps in 600s | Raw val_bpb |
|--------|---------|---------------|-------------|
| Baseline (9L, dim=512, 2x MLP) | 44 | ~13,400 | 1.22 |
| PR #1019 (11L, dim=512, 3x MLP) | 86 | ~6,900 | 1.11 |
| **Ours (13 eff, dim=624, 3x MLP)** | **126** | **4,747** | **1.18** |

Our model is ~3x slower than baseline per step. With only 4747 steps we significantly undertrain.

## Where the Time Goes

Per step (126ms on 8xH100, grad_accum=1):
- 13 effective layers × forward+backward through each
- Each layer: attention (QKV proj + softmax + output proj) + SwiGLU MLP (gate+up+proj)
- XSA on all 7 blocks adds cross-sample attention overhead
- U-Net skip connections (minor)
- SmearGate (minor)

**Main cost drivers** (roughly proportional to FLOPs):
1. **13 effective layers** vs 9 baseline (1.44x)
2. **dim=624** vs 512 (1.49x for matmuls, which scale as dim²)
3. **3x SwiGLU MLP** (1872 hidden) vs 2x MLP (1024) (2.74x MLP FLOPs)
4. **XSA on all blocks** vs 4 blocks (adds ~10-15% overhead)

Combined: ~3x slower, which matches observed 126ms vs 44ms.

---

## Quick Benchmark Plan

**Method**: Run 200-step tests (no warmdown, GPTQ/TTT disabled) on 8xH100 to measure step time. Each test takes ~30s + compile time.

All tests use: `MAX_WALLCLOCK_SECONDS=60 WARMDOWN_ITERS=0 GPTQ_ENABLED=0 TTT_ENABLED=0 TRAIN_LOG_EVERY=10 VAL_LOSS_EVERY=0`

### Test A: Baseline (current config)
```bash
NUM_LAYERS=7 MODEL_DIM=624 MLP_MULT=3 XSA_LAST_N=7 \
LAYER_SCHEDULE=0,1,2,3,4,3,4,3,4,5,6,5,6 QK_GAIN_INIT=4.0
```
**Expected**: 126ms/step, ~4740 steps in 600s

### Test B: Reduce recurrence (2x instead of 3x for middle blocks)
```bash
NUM_LAYERS=7 MODEL_DIM=624 MLP_MULT=3 XSA_LAST_N=7 \
LAYER_SCHEDULE=0,1,2,3,4,3,4,5,6,5,6 QK_GAIN_INIT=4.0
```
11 effective layers instead of 13. Saves 2 layer passes.
**Expected**: ~107ms/step (~15% faster), ~5600 steps

### Test C: Reduce dim to 512 (keep 13 effective layers)
```bash
NUM_LAYERS=7 MODEL_DIM=512 MLP_MULT=3 XSA_LAST_N=7 \
LAYER_SCHEDULE=0,1,2,3,4,3,4,3,4,5,6,5,6 QK_GAIN_INIT=4.0
```
dim² scaling: (512/624)² = 0.67x compute for attention+MLP.
**Expected**: ~85ms/step (~33% faster), ~7000 steps
**Risk**: Need fewer params → artifact might not fill 16MB → wasted capacity

### Test D: Reduce XSA to last 4 blocks (not all 7)
```bash
NUM_LAYERS=7 MODEL_DIM=624 MLP_MULT=3 XSA_LAST_N=4 \
LAYER_SCHEDULE=0,1,2,3,4,3,4,3,4,5,6,5,6 QK_GAIN_INIT=4.0
```
XSA only on blocks 4,5,6 (+ their recurrence copies). Blocks 0-3 use standard attention.
**Expected**: ~115ms/step (~9% faster), ~5200 steps

### Test E: 2x MLP instead of 3x
```bash
NUM_LAYERS=7 MODEL_DIM=624 MLP_MULT=2 XSA_LAST_N=7 \
LAYER_SCHEDULE=0,1,2,3,4,3,4,3,4,5,6,5,6 QK_GAIN_INIT=4.0
```
MLP hidden: 1248 vs 1872. Huge compute savings on the MLP side.
**Expected**: ~95ms/step (~25% faster), ~6300 steps
**Risk**: Smaller MLP = less capacity. But more steps may compensate.

### Test F: Combined — dim=512 + 2x recurrence + XSA last 4
```bash
NUM_LAYERS=7 MODEL_DIM=512 MLP_MULT=3 XSA_LAST_N=4 \
LAYER_SCHEDULE=0,1,2,3,4,3,4,5,6,5,6 QK_GAIN_INIT=4.0
```
11 effective layers, dim=512, XSA on last 4.
**Expected**: ~75ms/step (~40% faster), ~8000 steps

### Test G: Match competitor speed — 11 unique layers, dim=512
```bash
NUM_LAYERS=11 MODEL_DIM=512 MLP_MULT=3 XSA_LAST_N=4 \
LAYER_SCHEDULE= QK_GAIN_INIT=4.0
```
No recurrence. 11 unique blocks like PR #1019 uses. Direct comparison.
**Expected**: ~86ms/step, ~6900 steps

---

## Evaluation

For each test, record:
1. **ms/step** (stable, after step 50)
2. **val_bpb @ step 200** (early quality signal)
3. **Projected steps in 600s**
4. **Estimated model size** (does it fill 16MB?)

### Model Size Estimates

Calibrated against actual config A: 25.6M params → 15.1MB artifact.
MLP hidden = int(2 * mlp_mult * dim / 3) rounded to multiple of 64.

| Config | Unique blocks | Params | MLP hidden | ~Artifact | Headroom |
|--------|--------------|--------|------------|-----------|----------|
| A (current) | 7 | 26.0M | 1280 | **15.1 MB** | 0.9 MB |
| B (2x recurrence) | 7 | 26.0M | 1280 | **15.1 MB** | 0.9 MB |
| C (dim=512) | 7 | 17.3M | 1024 | **10.1 MB** | 5.9 MB ⚠️ |
| D (XSA=4) | 7 | 26.0M | 1280 | **15.1 MB** | 0.9 MB |
| E (2x MLP) | 7 | 20.1M | 832 | **12.0 MB** | 4.0 MB ⚠️ |
| F (combined) | 7 | 17.3M | 1024 | **10.1 MB** | 5.9 MB ⚠️ |
| G (11L no recur) | 11 | 26.8M | 1024 | **15.4 MB** | 0.6 MB |

**Key insight**: B, D have the same params as A (same 7 unique blocks). C, E, F waste 4-6MB of the 16MB budget — should increase dim or add blocks to fill it.

### Decision Matrix

Best config maximizes: `quality_per_step × steps_in_600s` while filling ~15MB.

| Config | ms/step | Steps@600s | Fills 16MB? | Notes |
|--------|---------|------------|-------------|-------|
| A (current) | 126 | 4747 | ✓ 15.1MB | Baseline |
| B (2x recurrence) | ~107 | ~5600 | ✓ 15.1MB | Same params, fewer layers |
| C (dim=512) | ~85 | ~7000 | ✗ 10.1MB | Wastes 6MB! Need dim~580 |
| D (XSA=4) | ~115 | ~5200 | ✓ 15.1MB | Marginal speedup |
| E (2x MLP) | ~95 | ~6300 | ✗ 12.0MB | Wastes 4MB! Need mlp_mult~2.5? |
| F (combined) | ~75 | ~8000 | ✗ 10.1MB | Wastes 6MB |
| G (11L no recur) | ~86 | ~6900 | ✓ 15.4MB | Direct competitor comparison |

### Size-Filling Variants

Configs C/E/F waste capacity. Better alternatives that fill ~15MB:

**C2: dim=576 (instead of 512), 13 effective layers, 3x MLP**
- MLP hidden = int(2*3*576/3) = 1152, rounded to 1152
- ~22M params → ~13MB. Still wastes 3MB.

**C3: dim=512, 9 unique blocks, 3x recurrence on middle 3**
- Schedule: `[0,1,2,3,4,5,3,4,5,6,7,8,6,7,8]` = 15 effective, 9 unique
- ~24.6M params → ~14.5MB. Better fill.

**E2: 7 unique, dim=624, mlp_mult=3 but 2x recurrence (11 effective)**
- Same as B. Already fills 15.1MB and is faster.

**H: 9 unique blocks, dim=512, 3x MLP, recurrence on middle 3**
```bash
NUM_LAYERS=9 MODEL_DIM=512 MLP_MULT=3 XSA_LAST_N=9 \
LAYER_SCHEDULE=0,1,2,3,4,5,3,4,5,6,7,8,6,7,8 QK_GAIN_INIT=4.0
```
15 effective layers, 9 unique blocks. ~24.6M params → ~14.5MB.
More layers than A (15 vs 13) but faster per step (dim=512).
**Expected**: ~90ms/step, ~6600 steps

## After Benchmarks

Pick the 2-3 configs with best step time that still show reasonable val_bpb@200. Run those for full 600s (training only, no GPTQ/TTT) to get raw val_bpb. The winner gets the full pipeline (GPTQ + TTT).

**Key insight**: A config at 85ms/step getting 7000 steps only needs to match the quality-per-step of our current 126ms model to produce a better final model. Even if quality-per-step is 10% worse, 47% more steps more than compensates.

---

## Results (60s benchmarks on 8xH100 SXM, CA-MTL-1)

| Test | ms/step | Steps | val_bpb (final) | Steps@600s | ~Artifact MB | Verdict |
|------|---------|-------|-----------------|------------|-------------|---------|
| **A** (7L dim=624, 13eff, XSA=7) | **127.0** | 473 | 1.4296 | ~4,730 | 15.1 | Baseline |
| **B** (7L dim=624, 11eff, XSA=7) | **108.1** | 556 | **1.4025** | ~5,560 | 15.1 | +18% steps, better quality |
| **D** (7L dim=624, 13eff, XSA=4) | **126.1** | 477 | 1.4431 | ~4,770 | 15.1 | No speedup, worse quality |
| **E** (7L dim=624, 2xMLP, 13eff) | **120.1** | 500 | 1.4403 | ~5,000 | 12.0 ⚠️ | Marginal, wastes 4MB |
| **G** (11L dim=512, no recur, XSA=4) | **71.1** | 845 | **1.3516** | **~8,450** | 15.4 | **BEST: speed+quality** |
| **H** (9L dim=512, 15eff, XSA=9) | **96.5** | 622 | 1.3981 | ~6,220 | 14.5 | Fast, but loses to G |

### Predictions vs Actuals

| Test | Predicted ms/step | Actual ms/step | Prediction accuracy |
|------|-------------------|----------------|---------------------|
| A | 126 | 127.0 | ✓ within 1% |
| B | 107 | 108.1 | ✓ within 1% |
| D | 115 | 126.1 | ✗ XSA doesn't affect speed |
| E | 95 | 120.1 | ✗ MLP speedup smaller than expected |
| G | 86 | 71.1 | ✗ much faster than predicted! |
| H | 90 | 96.5 | ~close |

### Key Findings

1. **Config G is the clear winner**: 11 unique layers at dim=512, no depth recurrence. 71ms/step = 8,450 steps in 10 min. Best quality at same wall time (1.3516 vs 1.4296).

2. **Depth recurrence hurts on the 10-min track**: It saves parameters but costs compute per step. With unlimited time it's great; with 10-min constraint, the extra effective layers aren't worth the slower steps.

3. **XSA count doesn't affect step time** (D vs A: 126 vs 127ms). The overhead is negligible. Use XSA everywhere.

4. **2x MLP barely helps speed** (E: 120ms vs A: 127ms) but wastes 4MB. Not worth it.

5. **Config B is a free win over A** if we keep depth recurrence: 108ms vs 127ms with better quality. Reducing from 3x to 2x recurrence on the middle blocks is strictly better.

### Recommendation

For the 10-min record track: **use Config G** (11L, dim=512, 3x MLP, XSA=4, no recurrence). 
- Add QK-Gain=4.0, GPTQ, SGD TTT on top
- ~8,450 steps in 600s → significantly more training
- 15.4MB artifact fills the 16MB budget
- This is essentially the same architecture as the top merged record (PR #1019) but with our GPTQ+TTT stack
