# Parameter Golf — Experiment Log & Plan

## Goal
Train the best LM in 16MB. Scored on val_bpb (lower = better). Baseline: **1.2244 BPB** (8xH100 SXM, 9L ReLU², 600s).

## Current Best
**val_bpb = 1.2019** (Exp 14 on 8xH100 SXM, int8+zlib roundtrip). Not yet a valid submission — needs int5/int6 quantization to fit 16MB.

## Current Architecture
11L, 512d, SwiGLU MLP 3x, 8 heads, 4 KV heads, SmearGate, BigramHash, OrthoInit, Muon WD=0.04, SWA during warmdown, warmdown=3000. 27M params.

---

## Experiment Results

| Exp | Config | GPU | Steps | val_bpb | Key Finding |
|-----|--------|-----|-------|---------|-------------|
| 1 | Baseline, batch=524K | 1xRTX 5090 | 94 | 3.10 | Batch too large for 1 GPU |
| 2 | batch=65K | 1xRTX 5090 | 1,785 | 1.46 | 12x speedup from smaller batch |
| 3 | seq_len=512 | 1xRTX 5090 | 1,925 | 1.49 | Shorter sequences worse |
| 4 | 600s training | 1xRTX 5090 | 5,964 | 1.354 | More training helps (log curve) |
| 5a | 6L/624d arch change | 1xRTX 5090 | — | 3.57 | LR doesn't transfer across architectures |
| 5 | batch=32K | 1xRTX 5090 | 8,533 | 1.38 | Smaller batch worse |
| 6 | 1200s, warmdown=600 | 1xRTX 5090 | 11,909 | 1.312 | Warmdown gives 2.6x per-step efficiency |
| 7 | warmdown=2000 | 1xRTX 5090 | 11,597 | 1.306 | Layer 3 activated, position degradation worsened |
| 8a | warmdown=3000 | 1xRTX 5090 | 11,603 | **1.299** | Best single-GPU baseline |
| 8b | warmdown=3000 + register | 1xRTX 5090 | 11,200 | 1.301 | Register token dead |
| 9 | 6L/608d loop to 9 | 1xRTX 5090 | 8,892 | 1.333 | Layer looping + wider = net negative |
| 10 | + SmearGate BigramHash OrthoInit MLP3x WD | 1xH100 PCIe | 12,596 | 1.283 | Competition stack works (+0.016) |
| 11a-d | Data sampling variants | 1xH100 SXM | ~1,500 | — | Dead end (capacity-limited) |
| 12 | + SWA (300s, too short) | 1xH100 SXM | ~1,530 | — | Inconclusive (SWA needs long runs) |
| 13 | SwiGLU replaces ReLU² | 1xH100 SXM | ~1,550 | — | SwiGLU > ReLU² (+0.004), activates L8 |
| 14 | 11L SwiGLU SWA | 1xH100 SXM | 11,248 | 1.278 | No dead layers, +0.005 vs Exp 10 |
| **14-8x** | **Same, competition HW** | **8xH100 SXM** | **8,583** | **1.202** | **Beats baseline by 0.023** |

## Confirmed Lessons

**Training regime:**
1. Warmdown=3000 is optimal — 2.6-3.0x per-step efficiency during cosine decay
2. Batch=65K optimal for 1xGPU, 524K for 8xGPU
3. More training always helps but with logarithmic returns
4. Data sampling/curriculum does not help — diminishing returns are capacity-limited
5. Post-hoc fine-tuning destroys Muon weight geometry

**Architecture:**
6. 11L > 9L by ~0.009 BPB (1xGPU), eliminates dead layers
7. SwiGLU > ReLU² by ~0.004 BPP at matched params, activates output layer
8. SmearGate + BigramHash + OrthoInit add ~0.010-0.015 BPB (must use OrthoInit)
9. MLP 3x adds ~0.008 BPP via increased capacity
10. Layer looping + wider dim = net negative (step time overhead kills it)
11. Register token doesn't help at this scale
12. Architecture changes NEED LR sweeps (Exp 5a proved this)

**Quantization & eval:**
13. Int6+zstd fits 22M params in 14.6MB (1.4MB headroom)
14. Muon WD=0.04 reduces int8 quant penalty from 0.0042 to 0.0037
15. SWA smooths weights for better compression (14.4MB vs 15.4MB in short run)
16. Sliding window eval (stride=256) adds ~0.033 BPB at eval time
17. BPB conversion: `val_loss / ln(2) * tokens_per_byte` where tokens_per_byte=0.4104

**Hardware:**
18. Batch size is the single biggest factor: 524K vs 65K = 0.076 BPB difference
19. 8xH100 SXM: 70ms/step with 11L SwiGLU MLP3x. 1xH100: 213ms/step.

---

## Detailed Analysis Documents

- [Experiment 10 Analysis](experiment10_analysis.md) — Competition stack (SmearGate + Int6 + MLP 3x)
- [Experiment 11 Analysis](exp11_hardtoken_sampling.md) — Hard token data sampling (negative result)
- [Experiments 12-14 Analysis](exp12_13_14_analysis.md) — SWA + SwiGLU + 11L + 8xH100 result

---

## Next: Experiment 15 — Submission-Ready Run

### Goal
Produce a valid 16MB submission that beats the baseline (1.2244). Current best pre-quant is 1.2019 (int8+zlib, 24.7MB — over limit). Need proper quantization + eval improvements.

### Changes from Exp 14

| Parameter | Exp 14 (8xH100) | Exp 15 | Why |
|-----------|-----------------|--------|-----|
| TRAIN_SEQ_LEN | 1024 | **2048** | Longer context → better sliding window eval, ~0.01 BPB |
| MATRIX_LR | 0.04 | **0.02** | Competition consensus for 8xH100 + 524K batch |
| Quantization | int8+zlib | **int5-MLP + int6-attn + zstd-22** | Must fit 27M params in 16MB |
| Eval | standard 1024 | **sliding window stride=64, seq=2048** | ~0.03 BPP free |
| Wallclock | 600s | 600s | Competition standard |

### Why these specific changes

**Seq 2048:** Every top-5 submission uses seq 2048. The model trains on longer sequences, learning to use more context. Sliding window eval with 2048 windows gives each scored token ~2000 tokens of context. Costs ~2x step time but 524K batch compensates. Expected ~4,000 steps at ~150ms/step.

**LR=0.02:** Competition consensus for 8xH100 with 524K batch. Our LR=0.04 was tuned on 1xGPU with 65K batch. Larger batch → smoother gradients → lower LR optimal. PR #198 (1.1326) uses matrix_lr=0.04 but PR #162 (1.1483) uses 0.02. Quick sweep recommended.

**Int5-MLP + Int6-attn:** 27M params. MLP weights are ~60% of total and tolerate lower precision. Attention weights (Q, K, V, O) need int6 for quality. Per PR #180 (1.1428, 3rd place). Estimated size: ~14.8MB.

### Param budget (int5-MLP + int6-attn + zstd-22)

| Component | Params | Quant | Est. bytes |
|-----------|--------|-------|-----------|
| tok_emb (fp16 passthrough) | 524K | fp16 | 1.0 MB |
| 11 blocks attn (Q,K,V,O) | ~9.4M | int6 | ~6.2 MB |
| 11 blocks MLP (gate,up,proj) | ~16.9M | int5 | ~6.6 MB |
| BigramHash + SmearGate | ~0.6M | int6 | ~0.4 MB |
| Scales + control tensors | ~0.1M | fp16/fp32 | ~0.2 MB |
| Code | — | — | ~0.06 MB |
| **Total** | **~27.5M** | mixed | **~14.5 MB** |

### Implementation needed

1. **Int5 quantization function:** per-row scale = abs_max / 15, quantize to [-16, 15] in int8 container
2. **Mixed quantization:** classify tensors as MLP (int5) or attn (int6) by name pattern
3. **Seq 2048:** set TRAIN_SEQ_LEN=2048 (env var)
4. **Sliding window eval:** stride=64, seq=2048, batch=4-8 windows

### LR sweep plan
3 short 180s runs on 8xH100: LR=0.015, 0.02, 0.03. Pick best by val_bpb at step 500.

### Expected result
- Pre-quant: ~1.18-1.20
- After int5+int6+zstd quant: ~1.20-1.22
- With sliding window eval: **~1.17-1.19**
- Competition placement: **top 10** (between #5 at 1.2060 and pending #1 at 1.1326)

### Cost
- LR sweep: 3 × 3 min × $21.52/hr = ~$3.20
- Full run: 10 min × $21.52/hr = ~$3.60
- Total: ~$7

### Risk
- Int5 quant penalty higher than expected → fall back to smaller MLP or fewer layers
- Seq 2048 slows step time too much → fewer total steps, might not converge enough
- LR=0.02 is wrong for this architecture → sweep catches this
