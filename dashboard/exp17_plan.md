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

**Status: Not started**

### What
Post-process the Exp 17 checkpoint to fit in 16MB:
- MLP weights (gate, up, proj) → int5 (range [-16, 15], per-row scale)
- Attention weights (Q, K, V, O) → int6 (range [-32, 31], per-row scale)
- Tied embedding → fp16 passthrough
- Control tensors (scales, gates, gains) → fp16 passthrough
- Compress with zstd level 22

### Why
Current size: 24.7MB (int8+zlib). Competition limit: 16MB. Without this, we cannot submit. Int5 for MLP is safe — MLP weights are more quantization-tolerant than attention weights (confirmed by PR #180, 3rd place at 1.1428).

### Expected result
- Size: ~14.8MB (under 16MB)
- Quantization penalty: +0.015-0.020 BPP
- Submission val_bpb: ~1.20 (still beats baseline 1.2244)

### Implementation
1. Write `quantize_int5_per_row()` function (per-row scale = abs_max / 15)
2. Write `quantize_state_dict_mixed()` — classify tensors by name pattern, apply int5 or int6
3. Save as `final_model.mixed.ptz`
4. Roundtrip validation: decompress → dequantize → eval val_bpb
5. Verify total artifact size < 16MB

### Run on
Pod (needs validation data for roundtrip eval). ~5 min.

---

## Addendum 2: Document-Aware TTT (Test-Time Training)

**Status: Not started**

### What
Adapt the model to each document during evaluation:
1. For each document in the validation set (50,000 documents, separated by BOS token id=1):
   - Do a quick forward pass over the document's text
   - Compute gradients and update a small set of parameters (LoRA or full SGD)
   - Score the document with the adapted model
   - Reset weights for next document

### Why
Our document analysis found:
- 50,000 documents, median length 733 tokens
- First 50 tokens of each document cost 5.0 nats vs 4.1 nats settled (cold-start penalty ~0.9 nats)
- 8% of all tokens are within the first 100 tokens of a document
- Cold-start penalty accounts for up to ~0.07 BPP

TTT reduces cold-start by "reading" the document before scoring it. The model learns the document's topic, vocabulary, and style.

### Concern
Competition data shows TTT + SmearGate = only -0.001 BPP (vs -0.033 without SmearGate). SmearGate may already capture what TTT provides. But our document analysis shows a clear cold-start signal that SmearGate doesn't address (SmearGate only uses the immediately previous token, not the document's overall topic).

### Expected result
- Optimistic: -0.010 to -0.033 BPP
- Realistic with SmearGate: -0.001 to -0.010 BPP
- Combined with Addendum 1: submission val_bpb ~1.17-1.20

### Implementation options
1. **Full-model SGD**: Single epoch, LR=3e-4, momentum=0.95. Simple but slow (~200s eval budget).
2. **LoRA TTT**: Rank-8 LoRA adapters, faster adaptation. Used by PR #77.
3. **Stride-OGD**: Online gradient descent on vocab bias only. Lightest, fastest. PR #241.

### Run on
Pod (needs full validation data + GPU for fast forward passes). ~10-20 min depending on TTT method.

---

## Addendum 3: Sliding Window Eval (stride=64 or document-aware)

**Status: Partially done (stride=256 measured at -0.026 BPP)**

### What already measured
- Standard eval: val_bpb = 1.1826
- Sliding window stride=256: val_loss = 1.9531 → ~1.157 BPP (-0.026)

### What to try
- Stride=64: may give additional ~0.002 BPP (competition data says marginal vs stride=256)
- Document-aware windowing: start each window at a `<s>` boundary

### Result from quick test
Document-isolated eval showed no improvement over standard eval (+0.0025, noise). Cross-document context isn't hurting — the model ignores irrelevant prior-document context.

---

## Execution Order

1. **Addendum 1 (int5+int6)** — Must do first. No submission without it.
2. **Addendum 2 (TTT)** — Run on the quantized model to measure actual submission BPP.
3. **Addendum 3 (sliding window)** — Already measured, apply at final eval.

## Expected Final Submission

| Component | BPP |
|-----------|-----|
| Exp 17 pre-quant | 1.1826 |
| + Int5+Int6 quant penalty | +0.015 to +0.020 |
| + Sliding window (stride=256) | -0.026 |
| + TTT (if it works) | -0.001 to -0.033 |
| **Estimated submission** | **1.17 to 1.20** |

Competition baseline: 1.2244. Our submission beats it regardless of TTT outcome.
