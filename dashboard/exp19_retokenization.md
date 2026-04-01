# Exp 19: Retokenization (2048 Vocab)

## Hypothesis

Word-initial single-character tokens (`▁s`, `▁c`, `▁f`) account for 32.5% of hard tokens with avg loss 3.4 nats. Merging these into bigrams (`▁st`, `▁co`, `▁fi`) via a 2048-token vocabulary should reduce ambiguity and lower word-initial prediction loss.

## Plan

- Retokenize fineweb dataset from 1024 → 2048 vocab (already done, 16% fewer tokens)
- Phase 1: 300s smoke test on 1xH100, compare val_bpb
- Phase 2: Per-token category analysis (word-initial loss comparison)
- Phase 3: Full 80-min training if Phase 1 passes
- **Go/no-go:** val_bpb within 0.05 of 1024 baseline at step 1000

## Phase 1: Quick Signal (300s)

### Config

| Parameter | Value |
|---|---|
| GPU | 1xH100 SXM |
| Vocab | 2048 |
| Dataset | fineweb10B_sp2048 (11 train shards + 1 val) |
| Architecture | 11L, 512d, SwiGLU, XSA, SmearGate |
| Wallclock | 300s |
| Warmdown | 400 iters |
| Batch | 65,536 tokens |

### Results

- **val_bpb = 1.4448** (int8+zlib), 1.4471 (mixed quant)
- Mixed quant size: **17.1 MB (OVER 16MB limit)**
- 1,383 steps completed
- zstandard was missing from pod image — had to install mid-run

### Go/No-Go

**NO-GO on size.** Mixed quant already over 16 MB due to 2x embedding table. But continued to Phase 2/3 anyway to test the core thesis.

## Phase 2: Per-Token Category Analysis

Ran `vocab_compare_eval.py` on 500K tokens from each model (1024 model = exp17 80-min checkpoint, 2048 model = 300s checkpoint).

### Results

| Category | 1024 avg loss | 2048 avg loss | Delta |
|---|---|---|---|
| word_initial_1char | **1.964** | 2.902 | +0.938 |
| word_initial_2char | 2.010 | 2.934 | +0.924 |
| word_initial_long | 2.006 | 2.940 | +0.934 |
| continuation | 1.936 | 2.855 | +0.919 |
| Overall | **1.958** | 2.895 | +0.937 |

**Count change:** word_initial_1char: 64,898 → 52,372 (19% fewer). But remaining 1char tokens are *harder* and new 2char tokens are *equally hard*.

**NOTE:** Models trained for different durations (80 min vs 300s). Gap partially due to undertrained 2048 model. Continued to Phase 3 to control for this.

## Phase 3: Full Training (killed early)

Started full 80-min training to get apples-to-apples comparison. Set `VAL_LOSS_EVERY=500` to monitor convergence.

### Results

| Step | 2048 vocab | 1024 vocab (exp17) | Gap |
|---|---|---|---|
| 500 | 1.638 | ~1.38 | +0.26 |
| 1000 | 1.511 | 1.335 | +0.18 |
| 2000 | 1.418 | 1.272 | +0.15 |
| 3000 | 1.377 | 1.245 | +0.13 |

**Killed at step 3200** (~15 min, ~$1.10). Gap not closing — consistently ~0.13 BPB behind. Saving remaining ~$2.50.

## Verdict

**FAILED.** The core thesis is disproven.

1. **Ambiguity shifted, didn't shrink.** `▁st` is just as ambiguous as `▁s` + `t` separately. Word-initial-2char avg loss (2.934) ≈ word-initial-1char (2.902).
2. **Softmax over 2048 is uniformly harder.** All categories ~0.93 nats worse — not just word-initial.
3. **Embedding table doubled.** Pushes artifact over 16 MB.
4. **Fewer gradient steps.** 16% fewer tokens = fewer updates per epoch.

**Lesson:** The word-initial problem is better addressed by adding *context* at the embedding layer (n-gram hash) rather than changing the *vocabulary*. Same task + more information > different task.

**Next:** [Exp 20: N-gram Hash Embeddings](exp20_ngram_plan.md)

**Total cost:** ~$3.50
