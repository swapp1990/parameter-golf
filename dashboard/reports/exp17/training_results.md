# Experiment 17 — XSA (Cross-token Self-Attention) Full Run

## Summary

**val_bpb = 1.1826 (int8+zlib SWA-averaged)** — New best. Beats Exp 15 (1.1874) by **0.005 BPB**.

XSA removes the self-value projection from attention output in the last 4 layers, forcing attention to contribute only contextual information. Zero new parameters. ~7% step time overhead.

## What XSA Does

Standard attention output for each token is a weighted sum of all value vectors. But in practice, this output has high cosine similarity with the token's own value vector — attention is mostly just passing through self-information rather than gathering context.

XSA fixes this with two lines of math:
```
dot_yv = (y * v).sum(dim=-1, keepdim=True)
y = y - (dot_yv / dot(v,v)) * v
```

This subtracts the self-value component, leaving only what other tokens contributed.

Applied to layers 7-10 (last 4 of 11), where the self-attention bias is strongest.

## Config

Same as Exp 15 + XSA on last 4 layers. Trained on 1xH100 SXM with gradient accumulation (batch=524K, seq=2048, 4850s wallclock = equivalent to 8xH100 600s).

| Setting | Value |
|---------|-------|
| Architecture | 11L SwiGLU + XSA (last 4) |
| GPU | 1xH100 SXM (grad accum 8) |
| Batch | 524,288 |
| Seq | 2048 |
| Steps | 7,926 at 612ms/step |
| Wallclock | 4,850s |
| Cost | ~$3.60 |

## Step-by-Step Comparison vs Exp 15

| Step | Exp 15 (no XSA) | Exp 17 (XSA) | Delta |
|------|-----------------|-------------|-------|
| 1,000 | 1.3411 | 1.3348 | **-0.006** |
| 2,000 | 1.2761 | 1.2719 | **-0.004** |
| 3,000 | 1.2497 | 1.2453 | **-0.004** |
| 4,000 | 1.2332 | 1.2294 | **-0.004** |
| 5,000 | 1.2207 | 1.2179 | **-0.003** |
| 6,000 | 1.2034 | 1.2006 | **-0.003** |
| 7,000 | 1.1872 | 1.1840 | **-0.003** |
| **Final** | **1.1874** | **1.1826** | **-0.005** |

XSA provides a consistent 0.003-0.006 BPP improvement at every stage of training. The gap is larger early (0.006) and slightly narrows during warmdown (0.003), settling at 0.005 final.

## Deep Eval Comparison

| Metric | Exp 15 | Exp 17 (XSA) | Delta |
|--------|--------|-------------|-------|
| val_bpb (int8+zlib) | 1.1874 | **1.1826** | **-0.005** |
| Sliding window (stride=256) val_loss | 1.9622 | **1.9531** | **-0.009** |
| easy (<1) | 46.2% | **46.3%** | +0.1% |
| medium (1-3) | 24.5% | **24.6%** | +0.1% |
| hard (3-5) | 18.9% | **18.8%** | -0.1% |
| very_hard (>5) | 10.4% | **10.3%** | -0.1% |
| first_64 loss | 2.382 | **2.369** | -0.013 |
| last_64 loss | 1.946 | **1.935** | -0.011 |
| context_benefit | 0.436 | **0.434** | -0.002 |

Loss distribution barely changed — XSA improves uniformly across all difficulty levels. Position degradation slightly improved (first_64 and last_64 both better).

## Layer Ablation

| Layer | Role | Exp 15 | Exp 17 (XSA) | Change | XSA? |
|-------|------|--------|-------------|--------|------|
| L0 | encoder | +6.131 | **+6.207** | +0.08 | No |
| L1 | encoder | +1.070 | +0.954 | -0.12 | No |
| L2 | encoder | +0.486 | +0.464 | -0.02 | No |
| L3 | encoder | +0.282 | +0.267 | -0.02 | No |
| L4 | encoder | +0.207 | +0.222 | +0.01 | No |
| L5 | bottleneck | +0.191 | +0.214 | +0.02 | No |
| L6 | decoder | +0.212 | +0.237 | +0.02 | No |
| L7 | decoder | +0.217 | +0.208 | -0.01 | **Yes** |
| L8 | decoder | +0.222 | +0.169 | -0.05 | **Yes** |
| L9 | decoder | +0.190 | +0.169 | -0.02 | **Yes** |
| L10 | decoder | +2.920 | **+3.052** | +0.13 | **Yes** |

### Key observation: XSA redistributes work from middle XSA layers to L10

The XSA layers (L7-L9) lost 0.01-0.05 impact each. L10 gained +0.13. XSA forces L7-L9 to stop passing through self-information, which makes them slightly less important individually — but L10 becomes MORE important because it now receives better contextual features from the preceding XSA layers.

L0 also slightly strengthened (+0.08), suggesting the embedding processing benefits from XSA layers downstream using context more effectively.

## Sliding Window Result

| Eval | val_loss |
|------|---------|
| Standard (2048, non-overlapping) | ~1.997 |
| Sliding window (stride=256) | **1.9531** |

Sliding window delta: -0.044 nats (~0.026 BPP improvement).

## Why XSA Works Here

1. **The last 4 layers were wasting attention on self-information.** By removing this redundancy, attention is forced to focus on genuine cross-token context.

2. **L10 becomes more effective.** The output layer (already the 2nd most important at 2.92) gained +0.13 impact. It receives purer contextual features instead of redundant self-information from L7-L9.

3. **Position-independent improvement.** Both first_64 and last_64 loss improved equally, meaning XSA helps everywhere, not just at long-range positions.

## Why the 300s Short Run Missed This

The 300s short run showed only -0.0007 (noise). At 490 steps, the model hadn't trained long enough for the XSA benefit to differentiate from noise. At 7,926 steps, the 0.003-0.005 signal is clear and consistent.

**Lesson: short runs can miss small but real effects.** The 300s A/B test is reliable for large effects (>0.01) but misses small ones (0.003-0.005). For techniques with theoretical backing, a full run is needed.

## Final Architecture (New Best)

```
11 layers, 512 dim, SwiGLU MLP 3x, 8 heads (4 KV)
SmearGate + BigramHash* + OrthoInit
Muon WD=0.04, warmdown=3000, SWA every 200 steps
XSA on layers 7-10 (last 4)
seq_len=2048, batch=524288

* BigramHash contributes zero (Exp 16 finding) — can be removed
```

**val_bpb = 1.1826** (int8+zlib, 27M params, 24.7MB — needs int5+int6 for 16MB)
