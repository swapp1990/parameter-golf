# Exp 20: N-gram Eval-Time Cache

## Hypothesis

Word-initial tokens are hard because the model lacks context. An eval-time n-gram frequency cache can interpolate model predictions with n-gram statistics, improving BPB without any training or parameter changes.

**Key pivot:** Research into leaderboard entries (PR #724, #740) revealed that the top n-gram implementations are **eval-time frequency caches**, not trained embeddings. They use XOR hashing with prime multipliers, 4M buckets, and fixed or entropy-adaptive interpolation alpha.

**Key difference from exp19 (retokenization):** Zero model changes. The n-gram cache is a pure eval-time addition that interpolates model probabilities with observed n-gram frequencies.

## Plan

### Original Plan: Training-Time N-gram Embedding

Added `NgramHashEmbedding` (7-gram, 8192 buckets, dim=64) to the model architecture, injected before RMSNorm+SmearGate.

### Revised Plan: Eval-Time N-gram Cache

After Phase 1 failed for training embeddings, pivoted to eval-time cache approach matching leaderboard implementations:
- XOR hash with primes [36313, 27191, 51647, 81929, 131071, 65537]
- Orders 2-7, highest-order-first backoff
- Score-first protocol (cache updates after scoring)
- Fixed alpha=0.40 or entropy-adaptive mixing

## Phase 1: Training-Time Embedding (300s)

### Config

| Parameter | Value |
|---|---|
| GPU | 1xH100 SXM |
| Variant | 7-gram, 8192 buckets, dim=64 |
| Wallclock | 300s, warmdown=400 |

### Results

- **val_bpb = 1.526 at step 1000** (baseline: 1.335)
- **+0.19 BPB worse than baseline**

### Go/No-Go

**NO-GO.** Training-time n-gram embedding hurts. Zero-init projection means slow ramp-up. Additive prime hash is inferior to XOR hash. SmearGate already covers the local context signal.

## Phase 2: Eval-Time N-gram Cache (local, 2000 docs)

Pivoted to eval-time frequency cache. Pre-computed model log-probs on pod, ran n-gram cache locally on CPU.

### Config

- Model: exp17 unquantized checkpoint (1024 vocab)
- Cache: 4M buckets, orders 2-7, entropy-adaptive alpha
- 2000 docs, 1.79M tokens

### Results

| Method | BPB | Delta |
|---|---|---|
| Model only | 1.7115 | — |
| **Model + N-gram cache** | **1.6669** | **-0.0446** |
| N-gram hit rate | 99.9% | |

**-0.045 BPB improvement.** Ran in 46 seconds on local CPU.

Note: absolute BPB is higher than submission (1.17) because per-document eval without sliding window overlap. Relative delta is the meaningful metric.

## Phase 3: TTT + N-gram Stacking (pod, 500 docs)

Tested whether LoRA TTT and n-gram cache stack.

### Config

- Model: exp17 mixed-quantized checkpoint
- TTT: rank-8, LR=0.05, 256-token chunks
- N-gram: entropy-adaptive alpha, orders 2-7
- 500 docs

### Results

| Method | BPB | Delta from base |
|---|---|---|
| Base model | 1.7065 | — |
| TTT only | 1.7054 | -0.0012 |
| N-gram only | 1.6772 | -0.0293 |
| **TTT + N-gram** | **1.6757** | **-0.0308** |

**They stack.** Combined improvement is -0.031 BPB.

TTT shows only -0.001 on unquantized model (vs -0.034 on quantized in submission). This is expected — TTT helps more on quantized models where there's more error to recover.

## Phase 4: Full Eval (pod, 5000 docs)

Ran TTT + n-gram on 5000 docs with quantized model.

### Results

- **val_bpb = 1.537** (5000 docs, 6.2M tokens, 991s)
- BPB progression: 1.477 (500 docs) → 1.529 (2500) → 1.537 (5000)
- BPB stabilized around 1.53 after 2500 docs

Absolute BPB not comparable to submission numbers (different eval path — per-document vs sliding window, different byte counting). The relative improvements from Phase 2 and 3 are the actionable metrics.

## Verdict

**N-gram eval-time cache works. Training-time embedding does not.**

### What works
- **Eval-time frequency cache:** -0.045 BPB on unquantized, -0.029 on quantized (2000/500 doc samples)
- **Stacks with LoRA TTT:** additional -0.002 on top of n-gram
- **Zero parameters added to model** — pure eval-time computation
- **99.9% hit rate** — almost every token gets an n-gram prediction

### What doesn't work
- **Training-time n-gram embedding:** +0.19 BPB worse. SmearGate already covers local context. Additive prime hash is inferior to XOR. Zero-init projection wastes early training steps.

### Estimated submission impact
- Current submission: 1.1573 BPB (mixed quant + LoRA TTT)
- Expected with n-gram cache: ~1.11-1.12 BPB (extrapolating -0.04 to -0.05 improvement)
- Would move from ~#10 to ~#5-6 on leaderboard

### Next steps
1. Integrate n-gram cache into `train_gpt.py` eval pipeline (done, needs testing with proper byte counting)
2. Run full 50K doc eval with official eval path to get exact submission BPB
3. Tune alpha and bucket count for optimal performance
4. Consider entropy-adaptive alpha (PR #740 approach)

### Cost
- Pod time: ~$12 across all runs (exp20)
- Local compute: ~2 min CPU time
