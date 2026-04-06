# Experiments 21-25: Next Steps to Improve BPB

**Current baseline: val_bpb = 1.1573** (TTT-only, train-then-score, submission checkpoint)
**Current SOTA: 1.1086-1.1147 BPB** (XSA-all + GPTQ + advanced TTT)
**Gap to close: ~0.04-0.05 BPB**

---

## Exp 21: GPTQ-lite Quantization

**Expected gain: -0.001 to -0.003 BPB**
**Retrain needed: NO**

### What
Replace fixed row-max clip for int6 quantization with optimal clip search. Try 5 percentiles (0.999, 0.9995, 0.9999, 0.99999, 1.0) per weight matrix row, pick the one minimizing reconstruction MSE.

### Why
Proven -0.0006 BPB in the GPTQ-lite submission (#414). Zero training cost — post-training only. Our current mixed quantization uses fixed row-max which over-clips outlier weights.

### Quick Eval Method
1. Load existing unquantized checkpoint (exp17 final_model.pt, 106MB)
2. Apply GPTQ-lite clip search during quantization
3. Run eval_val on the requantized model
4. Compare BPB with current mixed quant (1.1914 baseline without TTT)
5. **Time: ~5 min on Vast.ai RTX 4060 Ti**

### Success Criteria
BPB improvement > 0.0005 on eval_val (without TTT).

---

## Exp 22: LeakyReLU(0.5)² Activation

**Expected gain: -0.003 to -0.005 BPB**
**Retrain needed: YES (300s smoke test)**

### What
Replace `relu²` with `F.leaky_relu(x, negative_slope=0.5).square()` in MLP.

```python
# Current
x = torch.relu(self.gate(x)).square() * self.up(x)

# New
x = F.leaky_relu(self.gate(x), negative_slope=0.5).square() * self.up(x)
```

Note: Our model uses SwiGLU (`F.silu(gate) * up`), not relu². Need to verify if leaky_relu² applies to SwiGLU or if we need to switch MLP type.

### Why
Proven -0.003 BPB in the #1 submission's ablation. Preserves negative gradient flow, eliminates dead neurons. The squaring maintains non-negative output bias.

### Quick Eval Method
1. Modify train_gpt.py MLP activation
2. Train 300s on Vast.ai (RTX 4060 Ti, single GPU)
3. Compare val_loss at equivalent step count vs baseline training log
4. If val_loss is lower by >0.005 at same step, proceed to full H100 run
5. **Time: ~10 min total (300s train + eval)**

### Success Criteria
val_loss improvement > 0.005 at step 1000 vs baseline (baseline: 2.2538 at step 1000).

---

## Exp 23: Legal Sliding-Window TTT

**Expected gain: -0.005 to -0.02 BPB**
**Retrain needed: NO**

### What
Replace our per-doc LoRA TTT with the top submission's TTT recipe:
- **Score**: Sliding window eval (stride=64) with full 2048-token context
- **Train**: SGD(lr=0.002, momentum=0.9), 3 epochs per 32K-token chunk, ALL weights unfrozen, cosine LR decay, grad clip 1.0
- Score-first: each chunk scored before training on it

### Why
Our current TTT: per-doc, LoRA only (Q/V), Adam, 1024-token train chunks, 1 epoch.
Top submission TTT: 32K-token chunks, SGD, 3 epochs, all weights, sliding window scoring.
The top submission's TTT gives -0.0025 BPB on top of their 1.1218 pre-TTT baseline.

### Quick Eval Method
1. Implement sliding-window TTT eval in standalone script
2. Load existing submission checkpoint (15.75MB mixed quant)
3. Run on first 2000 docs (~5% of val set)
4. Compare BPB with our current TTT result (1.1573 → should be lower)
5. **Time: ~30-60 min on Vast.ai**

### Success Criteria
BPB < 1.155 on the 2000-doc subset (current TTT gives ~1.17 on early docs).

---

## Exp 24: EMA Weight Averaging

**Expected gain: -0.001 to -0.003 BPB**
**Retrain needed: YES (300s smoke test)**

### What
Replace SWA-only with EMA(decay=0.997) every training step, plus Tight SWA(every 50 steps when LR scale < 0.2).

```python
# Add after each optimizer.step():
if not hasattr(base_model, '_ema_state'):
    base_model._ema_state = {n: p.clone() for n, p in base_model.state_dict().items()}
for n, p in base_model.state_dict().items():
    base_model._ema_state[n].lerp_(p, 1 - 0.997)
```

### Why
Proven -0.0006 BPB in the GPTQ-lite submission. EMA provides continuous smoothing (every step) while SWA captures discrete checkpoints. They stack — EMA smooths the trajectory, SWA averages the final phase.

### Quick Eval Method
1. Add EMA to training loop
2. Train 300s on Vast.ai
3. Compare val_loss with EMA-averaged weights vs SWA-only
4. **Time: ~10 min total**

### Success Criteria
val_loss improvement > 0.002 vs SWA-only at same step count.

---

## Exp 25: XSA on All 11 Layers

**Expected gain: -0.002 to -0.005 BPB**
**Retrain needed: YES (300s smoke test)**

### What
Apply Cross-Sample Attention (XSA) to all 11 layers instead of just the last 4.

```python
# Current
use_xsa=(i >= num_layers - 4)  # layers 7-10

# New
use_xsa=True  # all layers
```

### Why
The #1 submission (#1019) uses "XSA-all (11 layers)" and achieves 1.1147 BPB. Our model only uses XSA on layers 7-10. XSA forces the model to rely on cross-token information by removing self-value bias from attention output.

### Quick Eval Method
1. Change XSA config to all layers
2. Train 300s on Vast.ai
3. Compare val_loss trajectory vs baseline
4. **Time: ~10 min total**

### Success Criteria
val_loss improvement > 0.003 at step 1000.

---

## Priority & Execution Order

| Priority | Experiment | Retrain? | Est. Time | Expected BPB Gain |
|----------|-----------|----------|-----------|-------------------|
| 1 | Exp 21: GPTQ-lite | No | 5 min | -0.001 to -0.003 |
| 2 | Exp 23: Sliding TTT | No | 30-60 min | -0.005 to -0.02 |
| 3 | Exp 22: LeakyReLU² | Yes (300s) | 10 min | -0.003 to -0.005 |
| 4 | Exp 24: EMA | Yes (300s) | 10 min | -0.001 to -0.003 |
| 5 | Exp 25: XSA-all | Yes (300s) | 10 min | -0.002 to -0.005 |

**No-retrain experiments first** (Exp 21, 23), then retrain experiments (Exp 22, 24, 25).
Total estimated time for all 5 quick evals: ~2 hours on Vast.ai (~$0.17).

---

## Full Train Decision

After quick evals, any experiment showing promise gets a full training run:
- **Platform**: RunPod H100 SXM or 8xH100 (for record submission)
- **Duration**: 600s (10 min) for record track, unlimited for non-record
- **Cost**: ~$2.69/15min (1xH100) or ~$21/15min (8xH100)
- **Final eval**: Full 50K-doc TTT eval using the proven train-then-score approach
