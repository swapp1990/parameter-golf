# Base Model Gap Analysis

**Goal**: Understand where our BPB is lost and which levers have the most potential.

**Current best**: val_bpb = ~1.142 (exp26 + SGD TTT, full eval running)
**SOTA**: 1.1086-1.1147
**Gap**: ~0.03 BPB

## Data Source

Analysis of 79,579 tokens across 100 validation documents using the exp26 (XSA-all + EMA) model's per-token log-probabilities from `xsa_comparison.json`.

## 1. Loss Distribution: The Long Tail

| Loss bin | Tokens | % of tokens | % of total loss |
|----------|--------|-------------|-----------------|
| <1 nat | 37,119 | 46.6% | 5.0% |
| 1-2 nats | 10,077 | 12.7% | 9.5% |
| 2-3 nats | 9,329 | 11.7% | 14.9% |
| 3-4 nats | 8,397 | 10.6% | 18.7% |
| 4-5 nats | 6,541 | 8.2% | 18.6% |
| 5-7 nats | 6,116 | 7.7% | 22.7% |
| **7+ nats** | **2,000** | **2.5%** | **10.5%** |

**Key finding**: The model gets nearly half of all tokens right (loss <1). The problem is the long tail:
- The hardest 20% of tokens (loss >4 nats) contribute **52% of total loss**
- The hardest 2.5% of tokens (loss >7 nats) contribute **10.5% of total loss**
- If we could halve the loss on the top-20% hardest tokens, BPB would drop by ~0.026

This means small improvements on easy tokens are nearly worthless. All the gains come from fixing the hard tail.

### What are the hardest tokens?

Looking at the worst 20 predictions (loss >13 nats):
- **Subword fragments**: "ible", "ong", "ink", "ence", "ur" — the model committed to the wrong word and the next subword piece is completely unexpected
- **Common words in unexpected contexts**: "The", "in", "what" — the model was very confident about something else
- **Low entropy failures**: Many have entropy <1.0, meaning the model was VERY confident but WRONG. These are the most costly — high confidence on the wrong answer produces huge loss.

## 2. Quantization Gap: 0.0195 BPB

| Stage | val_bpb | Delta from unquantized |
|-------|---------|----------------------|
| Unquantized (float32) | 1.1682 | — |
| Int8 + zlib | 1.1731 | +0.0049 |
| Mixed (int5/int6/int8) + zstd | 1.1877 | +0.0195 |

**The mixed quantization costs 1.7% of model performance.**

Breakdown:
- Int8 quantization: +0.0049 BPB (near-lossless)
- Going from int8 → int5/int6: +0.0146 BPB (this is where it hurts)

The int5 MLP weights and int6 attention weights lose significant precision. Better quantization (like GPTQ's optimal clipping or higher-bit mixed schemes) could recover much of this.

**If we had perfect quantization**: Pre-TTT would be ~1.168 instead of 1.188. With SGD TTT (~-0.05), final BPB would be ~**1.118** — essentially matching SOTA.

## 3. Training Convergence: Not Fully Converged

| Step range | BPB/1000 steps improvement |
|------------|---------------------------|
| 7000-7500 | 0.0150 |
| 7500-8000 | 0.0122 |
| 8000-8165 | 0.0061 |

The model was still improving when the 80-minute wallclock hit. The rate is slowing (diminishing returns) but not zero. Extrapolating:
- 2x wallclock (~160 min): ~0.005-0.008 BPB improvement
- The non-record track has no time limit, so we could train longer

## 4. The Three Improvement Levers

| Lever | Potential BPB gain | Requires retraining? | Cost | Confidence |
|-------|-------------------|---------------------|------|------------|
| Better quantization (GPTQ-style) | 0.005-0.015 | No | ~$1 (eval only) | High |
| Longer training (2x wallclock) | 0.005-0.008 | Yes | ~$5 (H100 time) | High |
| Fixing hard tokens (architecture) | 0.01-0.03 | Yes | ~$10+ | Speculative |

### Lever 1: Better Quantization (Exp 33)

**No retraining needed.** Apply GPTQ-style optimal clipping to the existing unquantized checkpoint.

Current approach: fixed row-max clipping for int5/int6. Each weight row is scaled by its maximum absolute value.

GPTQ-lite approach: try multiple clip percentiles (0.999, 0.9995, 0.9999, 0.99999, 1.0) per row, pick the one minimizing reconstruction MSE.

This was planned as Exp 21 but never run. Expected gain: 0.005-0.015 BPB. Cost: ~$1 for eval-only on a GPU.

### Lever 2: Longer Training

Our training was capped at 80 min (4850s wallclock). The non-record track has no time limit. Training for 160 min would give ~8000 more steps, potentially -0.005 to -0.008 BPB.

Cost: ~$5-7 on H100. This stacks with everything else.

### Lever 3: Fixing Hard Tokens

The hardest 20% of tokens cause 52% of total loss. These are mostly:
- Wrong subword completions (model committed to wrong word)
- Confident wrong predictions (low entropy, high loss)
- Rare/domain-specific tokens

Potential approaches (need evaluation):
- More model capacity (wider dim, more layers) — but must fit in 16MB quantized
- Better tokenizer (fewer subword ambiguities)
- Curriculum training (focus more on hard examples)
- Architecture changes targeting confident wrong predictions

These need more investigation. See brainstorming document for ideas.

## Summary

The ~0.03 BPB gap to SOTA can be decomposed as:

```
Quantization loss:      ~0.015 BPB  (recoverable with better quant)
Training convergence:   ~0.005 BPB  (train longer)
Hard token tail:        ~0.010 BPB  (architecture/capacity)
                        ─────────
Total recoverable:      ~0.030 BPB  → would reach ~1.112 BPB
SOTA:                   ~1.109 BPB
```

**The biggest single lever is quantization (0.015 BPB), and it requires no retraining.**
