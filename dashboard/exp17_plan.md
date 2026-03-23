# Experiment 17 — Plan

## Base: XSA (Cross-token Self-Attention)

**Status: Complete. val_bpb = 1.1826 (int8+zlib, 27M params, 24.7MB)**

### What
Remove self-value projection from attention output in last 4 layers. Forces attention to contribute only contextual information, not redundant self-information.

```
z = y - (dot(y, v) / dot(v, v)) * v
```

Two lines of code. Zero new parameters. Applied to layers 7-10.

### Why
Attention in deeper layers has high cosine similarity with the token's own value vector — it's mostly passing self-information through. The MLP already handles self-information via the residual connection. XSA removes this redundancy.

### Result
Consistent -0.003 to -0.005 BPP improvement across all training steps. L10 impact increased from 2.92 → 3.05. Full analysis: [exp17_analysis.md](exp17_analysis.md)

---

## Addendum 1: Int5+Int6+Zstd Quantization

**Status: Complete. val_bpb ≈ 1.191 (quantized, 15.0MB)**

### What
Post-process the Exp 17 checkpoint to fit in 16MB:
- MLP weights (gate, up, proj) → int5 (range [-16, 15], per-row scale)
- Attention weights (Q, K, V, O) → int6 (range [-32, 31], per-row scale)
- Tied embedding (tok_emb) → int8 (range [-128, 127], per-row scale)
- Control tensors (scales, gates, gains) → fp16 passthrough
- BigramHash removed (590K params, zero quality loss — confirmed by Exp 16)
- Compress with zstd level 22

### Result

| Metric | Value |
|--------|-------|
| Pre-quant val_loss | 1.9969 |
| Quantized val_loss | **2.0111** |
| Quant penalty | **+0.014 nats (+0.009 BPP)** |
| Pre-quant val_bpb | 1.1826 |
| Quantized val_bpb | **~1.191** |
| Artifact size | **15.00 MB** |
| Headroom | 212 KB |

Quantization penalty (+0.009 BPP) is lower than estimated (+0.015-0.020). Muon WD=0.04 makes weights quantization-friendly.

**Key lesson:** Must include XSA in the eval forward pass — without it, val_loss jumps to 3.83 (model was trained WITH XSA, weights depend on it).

---

## Addendum 2: Sliding Window Eval (on quantized model)

**Status: Complete. val_loss = 1.9744, val_bpb ≈ 1.170**

### What
Score the quantized model using overlapping 2048-token windows with stride=256. Only the last 256 tokens per window are scored, ensuring each scored token has ~1792 tokens of context.

### Why
Standard eval uses non-overlapping windows where early tokens have minimal context. Sliding window gives every token rich context. Already measured on the pre-quant model:
- Pre-quant standard: val_loss = 1.9969
- Pre-quant sliding (stride=256): val_loss = 1.9531 (delta = -0.044)
- BPP improvement: ~-0.026

### Result

| Eval Method | val_loss | Approx BPP |
|-------------|---------|------------|
| Standard (non-overlapping 2048) | 2.0111 | ~1.191 |
| **Sliding window (stride=256)** | **1.9744** | **~1.170** |
| Delta | **-0.0367** | **~-0.022** |

Sliding window delta on quantized model (-0.037 nats) is similar to pre-quant (-0.044 nats). Quantization didn't significantly affect the sliding window benefit.

---

## Addendum 3: Document-Aware TTT (Test-Time Training)

**Status: After sliding window eval**

### What
Adapt the model to each document during evaluation. For each of the 50,000 documents (separated by BOS token id=1):
1. Forward pass over the document's text
2. Compute gradients, update parameters (LoRA or full SGD)
3. Score the document with adapted weights
4. Reset weights for next document

### Why
Document analysis found:
- 50,000 documents, median length 733 tokens
- Cold-start penalty: first 50 tokens cost 5.0 nats vs 4.1 settled (~0.9 nats gap)
- 8% of all tokens are cold-start (first 100 of each document)
- Theoretical ceiling: ~0.07 BPP

### Concern
Competition data: TTT + SmearGate = only -0.001 BPP (vs -0.033 without SmearGate). Our model has SmearGate, so gain may be minimal.

### Expected result
- Optimistic: -0.010 to -0.033 BPP
- Realistic with SmearGate: -0.001 to -0.010 BPP

### Run on
Pod. ~10-20 min depending on TTT method.

---

## Execution Order

1. ~~**Addendum 1 (int5+int6)**~~ ✅ Done. val_bpb ≈ 1.191, 15.0MB.
2. ~~**Addendum 2 (sliding window)**~~ ✅ Done. val_bpb ≈ 1.170.
3. **Addendum 3 (TTT)** ← Next. Can it push below 1.17?

## Current Submission Numbers

| Step | val_loss | val_bpb (approx) | Status |
|------|---------|-------------------|--------|
| Exp 17 pre-quant | 1.9969 | 1.1826 | ✅ |
| + Int5+Int6+Int8 quant | 2.0111 | ~1.191 (+0.009) | ✅ |
| + Sliding window (stride=256) | 1.9744 | **~1.170 (-0.022)** | ✅ |
| + TTT (if it works) | ? | ~1.14-1.17 | Next |

**Current best submission BPP: ~1.170**
Competition baseline: 1.2244. We beat it by **0.054**.
Competition #1 (official): 1.1748. We beat it by **0.005**.
Competition #1 (pending): 1.1326. We're 0.037 behind.
