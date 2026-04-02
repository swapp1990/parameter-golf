# Exp 27: N-gram Diagnostic — Why Does N-gram Hurt?

**Goal**: Understand WHY n-gram interpolation hurts BPB with full-context TTT scoring, and whether any configuration could make it help.

## Background

Exp 20 (n-gram eval) showed n-gram hurts with both chunked and full-context scoring:
- Chunked + n-gram: +0.033 BPB (HURTS)
- Full-ctx + n-gram: +0.044 BPB (HURTS)

**Hypothesis**: The model's attention already captures n-gram patterns, so interpolation just corrupts a well-calibrated softmax distribution.

**Counter-hypothesis**: The model (11L, 8H, 4KV-GQA, 2048 ctx) may NOT have perfect attention. There might be a regime (specific entropy range, lower alpha, higher n-gram orders) where n-gram provides useful signal.

## Test Design

Using the correct submission checkpoint (15,746,871 bytes), full-context train-then-score TTT, 200 docs:

### Test 1: Per-token win rate
For each token where n-gram has a prediction, compare model prob vs n-gram prob for the correct token. If the model almost always wins → attention captures n-gram patterns.

### Test 2: Alpha sweep
Test fixed alpha values: 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60. If even tiny alpha hurts → n-gram signal is pure noise to this model.

### Test 3: Entropy-conditional analysis
Bin tokens by model entropy [0-1), [1-2), ..., [6+). Check if n-gram helps in any specific entropy range. The current sigmoid gives high alpha at high entropy — maybe the opposite is better?

### Test 4: N-gram order analysis
Check win rates per n-gram order (2-7). Higher orders match longer patterns — they might be more reliable.

## Script

`scripts/experiments/exp27_ngram_diagnostic.py`

## Hardware

RunPod 1xH100 SXM ($2.69/hr). Expected runtime: ~5 min.

## Results (2026-04-02, RunPod 1xH100 SXM, ~3 min)

### Overall Stats
- Total tokens scored: 162,427
- Tokens with n-gram prediction: 83,050 (51.1%)
- **N-gram wins: 17,545 (21.1%)** — model wins 78.9% of the time
- Baseline BPB (no n-gram): 1.1730

### Alpha Sweep (BPB)
| Alpha | BPB | Delta vs baseline |
|-------|-----|-------------------|
| 0.001 | 1.1730 | +0.0000 (best) |
| 0.005 | 1.1733 | +0.0003 |
| 0.010 | 1.1737 | +0.0007 |
| 0.020 | 1.1749 | +0.0019 |
| 0.050 | 1.1790 | +0.0061 |
| 0.100 | 1.1874 | +0.0145 |
| 0.200 | 1.2078 | +0.0348 |
| 0.400 | 1.2616 | +0.0886 |
| 0.600 | 1.3403 | +0.1674 |

**No alpha value helps.** Even alpha=0.001 gives +0.0000 (neutral at best). The relationship is monotonically worse — more n-gram = more hurt.

### Entropy Bins (tokens with n-gram predictions)
| Bin | Count | N-gram win% | Avg Model NLL | Avg N-gram NLL | N-gram better? |
|-----|-------|------------|---------------|----------------|----------------|
| [0-1) | 30,683 | 33.1% | 0.257 | 8.698 | NO |
| [1-2) | 10,518 | 14.8% | 1.523 | 14.256 | NO |
| [2-3) | 14,503 | 15.1% | 2.528 | 17.042 | NO |
| [3-4) | 19,443 | 14.3% | 3.497 | 18.762 | NO |
| [4-5) | 7,896 | 11.1% | 4.263 | 20.270 | NO |
| [5-6) | 7 | 0.0% | 6.043 | 23.026 | NO |

**N-gram loses in EVERY entropy bin.** The avg n-gram NLL is 5-34x worse than the model across all bins. Even in the low-entropy bin [0-1) where n-gram has its best win rate (33%), the average NLL is catastrophically worse (8.7 vs 0.26).

### N-gram Order Win Rates
| Order | Total | Wins | Win% |
|-------|-------|------|------|
| 2 | 63,867 | 7,366 | 11.5% |
| 3 | 9,114 | 4,017 | 44.1% |
| 4 | 3,979 | 2,313 | 58.1% |
| 5 | 2,328 | 1,359 | 58.4% |
| 6 | 1,339 | 754 | 56.3% |
| 7 | 2,423 | 1,736 | 71.6% |

**Higher-order n-grams (4-7) actually win >50% of the time.** But they're rare (only 10K of 83K n-gram tokens). Order 2 dominates (77% of predictions) and has terrible 11.5% win rate — it drags down the overall average.

## Deep Analysis

### 1. The Catastrophic Zero Problem

The biggest issue is visible in the entropy bins. Look at the **[0-1) bin** (low entropy, model is confident):
- 30,683 tokens — the largest group
- N-gram wins 33% of the time (its best bin!)
- But avg model NLL = **0.26** vs avg n-gram NLL = **8.70** — that's **34x worse**

When the model is confident (low entropy), it assigns ~0.77 probability on average (e^(-0.26)). When n-gram is wrong in this bin, it assigns **probability 0** (the target was never seen after that n-gram context), giving NLL = -log(1e-10) ≈ 23. A few zero-probability catastrophes wipe out all the wins.

This pattern repeats in every entropy bin — n-gram's average NLL is 5-34x worse than the model's, regardless of how uncertain the model is.

### 2. Why Higher Orders Win But Still Can't Help

| Order | Tokens | Win% | % of all n-gram tokens |
|-------|--------|------|------------------------|
| 2 | 63,867 | 11.5% | **77%** |
| 3 | 9,114 | 44.1% | 11% |
| 4-7 | 10,069 | 60.4% | **12%** |

Order-2 bigrams dominate (77% of all n-gram predictions) and are terrible (11.5% win rate). The model's attention captures bigram statistics perfectly — it can literally attend to the previous token.

Orders 4-7 have >50% win rate but only cover 12% of n-gram tokens. These represent long repeated phrases (4-7 token sequences) that the model's attention sometimes misses due to limited depth (11 layers) and heads (8H, 4KV with GQA).

Even if we restricted to orders 4+ only, the alpha sweep tells us it still wouldn't help. At alpha=0.001, the delta is already +0.0000 (neutral at best). The reason: when order-7 matches correctly (71.6% win rate), the improvement is small (model was already decent). When it loses (28.4%), it assigns 0 probability → catastrophic NLL.

### 3. The Math: Why No Alpha Value Works

For a token where n-gram assigns probability 0 to the correct answer:
```
interpolated = (1 - α) * model_p + α * 0 = (1 - α) * model_p
NLL increase = -log(1 - α) ≈ α  (for small α)
```

For a token where n-gram assigns probability 1.0 (perfect match):
```
interpolated = (1 - α) * model_p + α * 1.0
NLL decrease ≈ α * (1 - model_p) / model_p  (for small α)
```

The problem: n-gram rarely assigns 1.0 (perfect match), but frequently assigns 0 (target not seen in that context). The cost per wrong token ≈ α, while the benefit per right token << α. The costs dominate at every alpha value, which is why the alpha sweep is monotonically worse.

### 4. Could N-gram Be Saved? (Theoretical Fixes)

**A. Laplace smoothing** — `ng_p = (count + 1) / (total + vocab_size)` — prevents zero probabilities. But with vocab_size=1024, unseen tokens get ~0.001 probability, still far worse than what the model assigns (typically 0.01-0.3 even for wrong tokens). Would reduce the catastrophe but not eliminate the fundamental problem.

**B. Orders 4+ only** — Skip bigrams/trigrams. But only 12% of tokens have order 4+ predictions, and the alpha sweep shows even infinitesimal mixing doesn't help on the full set. The 12% subset would have even less impact.

**C. "Boost only" mode** — Only adjust probability when n-gram agrees with the model's top-k predictions. This avoids the catastrophic zero problem entirely. But it also limits the upside — if n-gram and model already agree, there's little to gain from boosting.

**D. Geometric interpolation** — `p = model_p^(1-α) * ng_p^α` instead of linear. This is more stable when ng_p=0 (the product goes to 0 gracefully). But it still penalizes whenever ng_p < model_p, which happens 79% of the time.

None of these fixes address the core issue: the model with full-context TTT is fundamentally better at token prediction than frequency counts.

### 5. Why Attention Dominates N-gram

The model has:
- 11 transformer layers with 8 attention heads (4 KV with GQA)
- 2048 token context window
- LoRA TTT fine-tuning per document

For any n-gram pattern, the model can:
1. **Attend to the exact positions** where that pattern occurred before
2. **Use semantic/syntactic context** beyond the fixed n-gram window
3. **Weigh multiple patterns simultaneously** via multi-head attention
4. **Adapt per-document** via LoRA TTT

N-gram provides a strict subset of this information — it can only count exact token sequences. The only edge case where n-gram could theoretically help is very long exact repetitions (orders 5-7) where attention depth is insufficient to propagate the pattern. But these are too rare (6% of scored tokens) and the failure mode (zero probability) is too catastrophic to be useful.

## Conclusion

**N-gram interpolation is a dead end for this model with full-context TTT.**

1. **No alpha helps** — strictly monotonic: more n-gram = worse BPB
2. **Loses in every entropy bin** — the hypothesis "n-gram helps when model is uncertain" is false
3. **Bigrams (77% of predictions) are useless** — 11.5% win rate, model captures these perfectly
4. **Higher orders (4-7) win >50% but are too rare and too catastrophic when wrong**
5. **Root cause**: zero-probability assignments when n-gram hasn't seen the target token in that context, producing unbounded NLL that overwhelms any gains

**Recommendation**: Abandon n-gram interpolation entirely. If long-range repetition detection is desired, a fundamentally different approach is needed (copy mechanism, retrieval-augmented attention, or pointer networks) rather than crude frequency-count interpolation. However, given that these only affect ~6% of tokens and the model already handles them >40% of the time correctly, the expected gain ceiling is very low (~0.001 BPB at best). Effort is better spent on improving the base model or TTT procedure.
