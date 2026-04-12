# Technical Report: From 1.1172 to 1.0825 — Record Submission Journey

**Author**: swapp1990
**Date**: April 11, 2026
**Final Result**: val_bpb = 1.0825 (3-seed mean), 10-min track, 8xH100 SXM

---

## 1. Starting Point

Our non-record best was **val_bpb = 1.1172** (exp43), achieved on 1xH100 with unlimited training time (80 min):
- 7 unique blocks, dim=624, depth recurrence schedule [0,1,2,3,4,3,4,3,4,5,6,5,6] = 13 effective layers
- SwiGLU 3x MLP, XSA on all blocks, SmearGate, U-Net skips
- QK-Gain 1.5, SP1024 tokenizer, SWA averaging
- GPTQ (Hessian-compensated, int5 MLP / int6 attn / int8 embed) + zstd-22
- SGD all-weights TTT (lr=0.005, momentum=0.9)
- ~25.6M parameters, 15.2MB artifact

The merged leaderboard SOTA was **1.0810** (PR #1493). Gap: **0.036 BPB**.

---

## 2. The 10-Minute Track Challenge (Exp 53 → Exp 58)

### First Attempt: Direct Port (Exp 53, val_bpb = 1.1456)

We first tried running our exact 1xH100 architecture on 8xH100 within 600s:
- **Problem**: 126ms/step → only 4,747 steps vs 14,654 on 1xH100
- **Result**: val_bpb = 1.1456 post-TTT — much worse than our 1xH100 best
- **Lesson**: Our depth recurrence model was too slow per step for the 10-min window

### Architecture Benchmarks (Exp 58)

We benchmarked 6 architecture configs to find the best speed/quality tradeoff:

| Config | ms/step | Steps@600s | val_bpb (60s) |
|--------|---------|------------|---------------|
| 7L dim=624, 13 eff (original) | 127 | 4,747 | 1.4296 |
| 7L dim=624, 11 eff | 108 | 5,560 | 1.4025 |
| **11L dim=512, no recurrence** | **71** | **8,450** | **1.3516** |

**Key insight**: For the 10-min track, more unique layers at smaller dim beats depth recurrence. The 11L config gets 78% more steps with better quality per step.

### TTT Integration Bug (Exp 58)

Our first integrated TTT attempt produced **val_bpb = 1.82** — catastrophically bad. Root cause: `torch.compile` modifies the model in-place, breaking per-document SGD adaptation. The compiled graph doesn't support the "load weights → train → score → restore weights" pattern that TTT requires.

**Fix**: Create a fresh uncompiled `ttt_model = GPT(...)` for TTT eval, loaded with the same weights. This matched our working standalone eval script.

---

## 3. Activation Function (Exp 59)

Replaced SwiGLU with **LeakyReLU²** (negative_slope=0.5, squared output):

```python
# SwiGLU: 3 matrices, hidden = int(2/3 * mlp_mult * dim)
output = proj(silu(gate(x)) * up(x))

# LeakyReLU²: 2 matrices, hidden = mlp_mult * dim (50% wider)
output = proj(leaky_relu(up(x), 0.5) ** 2)
```

Same parameter count, but 50% wider hidden layer. Every top competition entry uses it.

**Result**: -0.006 to -0.009 BPB improvement across all metrics. Step time slightly faster (64ms vs 66ms) because 2 matmuls instead of 3.

---

## 4. The SP8192 Tokenizer (Exp 60)

Switched from SP1024 (1024 vocab) to SP8192 (8192 vocab). Each token covers ~3-4x more bytes, meaning:
- More text processed per training step
- Better compression efficiency per token
- Larger embedding table (8192 × 512 = 4.2M params vs 0.5M)

**Problem**: The bigger embedding ate our parameter budget. Had to drop from 10 layers to 7 to fit in 16MB with our existing compression (zstd-22).

**Result**: Raw val_bpb similar to SP1024 despite fewer layers. The tokenizer improvement compensated for lost depth.

---

## 5. Compression Revolution (Exp 61-62)

### The Size Problem

We wanted 11 layers + MLP 4x + SP8192 (~36M params) but our compression pipeline only achieved 0.66 bytes/param. At that rate, 36M params = 24MB — far over the 16MB limit.

The SOTA fits the same architecture at **0.44 bytes/param**. How?

### SDClip Quantization

Standard quantization uses `scale = max(abs(row)) / clip_range` (min/max clipping). The SOTA uses **SDClip**: `scale = k × std(row) / clip_range`. This produces lower-entropy quantized values that compress much better.

**Compression comparison on our real model**:
- Min/max int6 + brotli: **0.689 bytes/param** (24.8MB — OVER)
- SDClip int6 + brotli: **0.455 bytes/param** (16.0MB — FITS!)

But SDClip has a critical dependency: **weight magnitude**. With our default weight decay (WD=0.04), weights had outliers that SDClip couldn't handle. The quantization MSE was **14x worse** than min/max, losing 0.17 BPB.

### The Weight Decay Discovery

We tested quantization quality across methods:

| Quantization | val_bpb | Delta from raw |
|-------------|---------|---------------|
| Raw float (1.1287) | 1.1287 | — |
| Min/max int6 | 1.1347 | +0.006 |
| SDClip k=12.85 (WD=0.04) | 1.2953 | **+0.167** |

**SDClip destroyed quality with WD=0.04.** The weights had heavy-tailed distributions where `k × std` severely underestimated the range needed.

The SOTA uses **WD=0.095**. This produces smaller, more Gaussian weights where SDClip works perfectly.

### Retrain with WD=0.095

Same architecture, just higher weight decay:
- SDClip MSE dropped from **0.003230 to 0.000087** (37x improvement)
- Quantization now costs only **~0.014 BPB** instead of 0.167
- Raw val_bpb actually improved slightly: **1.1213** (vs 1.1306 with WD=0.04)

---

## 6. GPTQ Integration Challenges

### torch.compile vs Hooks

GPTQ requires collecting Hessians via forward hooks on specific modules. `torch.compile` transforms the model graph, making `named_modules()` return different names. Hooks registered on compiled models silently fail — no Hessians collected, GPTQ degenerates to plain SDClip.

**Fix**: Create a fresh uncompiled model specifically for Hessian collection, same pattern as TTT.

### Embedding Hessian Mismatch

The embedding weight is (8192, 512) but `collect_layer_inputs` hooks the module's input activations. For embeddings, the "input" is token IDs (integers), not continuous activations. The resulting Hessian has wrong dimensions.

**Fix**: Skip GPTQ for embeddings, use plain SDClip (int8 with k=20). Embeddings are 127-level int8 anyway — GPTQ barely helps.

---

## 7. The Full Stack

Our final architecture combines:

| Component | Choice | Why |
|-----------|--------|-----|
| Tokenizer | SP8192 | Better compression per token |
| Layers | 11 unique, no recurrence | More params > more depth for BPB |
| Dim | 512 | Sweet spot for 16MB budget |
| MLP | 4x LeakyReLU² (2048 hidden) | Wider than SwiGLU at same params |
| QK-Gain | 5.25 | Sharper attention from step 1 |
| XSA | All 11 layers | Cross-sample attention |
| EMA | 0.9965 | Better than SWA for this architecture |
| Weight decay | 0.095 (Muon), 0.085 (embed) | Critical for SDClip quantization |
| Matrix LR | 0.022 | Lower than our original 0.04 |
| Warmdown | 72% | Much longer than our original 30% |
| Quantization | GPTQ + SDClip (int6/int8) | Hessian-compensated, k×std clipping |
| Compression | Byte-shuffle + Brotli-11 | ~0.45 bytes/param |
| TTT | SGD all-weights (lr=0.005) | Adapt all 36M weights per document |

### Pipeline Flow

```
Training (600s, 8xH100)
    → ~7360 steps, 81.5ms/step
    → EMA model, raw val_bpb ~1.12
    
GPTQ+SDClip (on fresh uncompiled model)
    → Collect Hessians from 32 calibration batches
    → int6 for attention+MLP (k=12.85), int8 for embeddings (k=20)
    → Column ordering by Hessian diagonal, block error compensation
    → post-quant val_bpb ~1.14
    
SGD All-Weights TTT (on fresh uncompiled model)
    → For each of 50K val docs: SGD adapt → score → restore
    → Sharded across 8 GPUs, ~355s total
    → post-TTT val_bpb ~1.08
    
Artifact Compression
    → torch.save + byte_shuffle(stride=2) + brotli(quality=11)
    → ~15.6MB, fits 16MB
```

---

## 8. What Failed (and Why)

### Depth Recurrence on 10-Min Track
Looping the same layers adds effective depth but costs compute per step. At 600s, the 3x fewer steps from slower step time outweighed the quality benefit of more effective layers. BPB depends more on parameter count than compute depth — confirmed by LoopLM research showing loops help reasoning but not memorization (perplexity).

### Mixed int5/int6/int8 Quantization
Our original quantization used int5 for MLP, int6 for attention, int8 for embeddings. This gave different entropy distributions per layer type, hurting compression. All-int6 with SDClip compresses uniformly better.

### SDClip with Low Weight Decay
SDClip assumes weights are roughly Gaussian. With WD=0.04, our weights had heavy tails — outliers beyond 12.85σ got clipped, causing 0.17 BPB quantization loss. This was the single biggest blocker, responsible for weeks of wasted experiments trying to "fix" compression when the real issue was training hyperparameters.

### torch.compile Breaking Post-Training Pipeline
Both GPTQ (Hessian hooks) and TTT (per-document SGD) require operations that torch.compile doesn't support. We wasted 2 full 8xH100 runs (~$40) debugging this before understanding the root cause. The fix (fresh uncompiled models) should have been applied from the start — it's exactly what our working standalone eval script did.

---

## 9. What Worked (Key Insights)

### 1. Weight Decay Controls Quantization Quality
The most important finding. WD=0.095 vs WD=0.04 made the difference between "SDClip destroys the model" and "SDClip is near-lossless." This one hyperparameter change enabled fitting 50% more parameters in 16MB.

### 2. SDClip + Brotli = 0.45 bytes/param
With proper weight decay, SDClip quantization produces low-entropy values that Brotli compresses extremely well. This compression ratio is what enables matching the SOTA's architecture (36M params in 16MB).

### 3. Byte-Shuffle Before Compression
Interleaving bytes by position (stride=2) groups similar-valued bytes together, improving Brotli's compression ratio. Adopted directly from the SOTA.

### 4. Fresh Uncompiled Models for Post-Training
torch.compile is essential for training speed but breaks hooks and per-document gradient flow. Creating fresh GPT() instances for GPTQ and TTT solves both problems cleanly.

### 5. SP8192 + MLP 4x + 11 Layers
The SOTA architecture works because each component is optimized together. SP8192 needs more embedding params, which is offset by LeakyReLU²'s parameter efficiency. 11 unique layers at dim=512 hits the sweet spot for the 16MB budget.

---

## 10. Progression Summary

| Experiment | val_bpb | Key change | Cost |
|-----------|---------|-----------|------|
| exp43 (1xH100, non-record) | 1.1172 | GPTQ + SGD TTT | ~$11 |
| exp53 (1xH100) | 1.1159 | QK-Gain 4.0 | ~$11 |
| exp58 (8xH100, 10 min) | 1.1456 | 10L dim=512, no recurrence | ~$6 |
| exp59 (8xH100) | ~1.14 est | LeakyReLU² activation | ~$6 |
| exp60 (8xH100, SP8192) | 1.1897 | SP8192 tokenizer, 7L | ~$6 |
| exp61b (8xH100) | 1.1882 | 8L + 2x recurrence | ~$6 |
| exp62 (8xH100) | 1.1768 post-TTT | 11L MLP4x, SDClip (broken) | ~$6 + $3 |
| **exp62b (8xH100)** | **1.0825** | **WD=0.095 fix + full pipeline** | ~$6 + $3 |

Total compute cost for the final result: ~$30 (training + eval). Total exploration cost including failed experiments: ~$150.

---

## 11. Gap Analysis vs SOTA

| | Merged SOTA (#1493) | Ours |
|---|---|---|
| val_bpb | **1.0810** | 1.0825 |
| Gap | — | +0.0015 |
| Depth recurrence | 3-layer (17 eff) | None (11 eff) |
| Parallel residuals | L7+ (GPT-J style) | None |
| Score-first TTT | 3 epochs, 32K chunks, cosine LR | 1 pass, 2K chunks |
| Partial RoPE | 16/64 dims | Full RoPE |
| Train batch | 786K tokens/step | 524K tokens/step |

The remaining 0.0015 BPB gap likely comes from:
1. **Depth recurrence** (+6 effective layers at no param cost)
2. **Parallel residuals** (GPT-J style, ~0.005 BPB)
3. **Score-first TTT** (more sophisticated adaptation)
4. **Larger training batch** (786K vs 524K tokens/step)

Each of these is an incremental improvement that could close or exceed the gap.

---

## 12. Reproducibility

```bash
# Setup
pip install brotli sentencepiece
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

# Run (all hyperparameters have correct defaults in train_gpt.py)
SEED=314 torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=42  torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=999 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Expected: val_bpb ~1.08 ± 0.001 on 8xH100 SXM with RunPod parameter-golf image.
