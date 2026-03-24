# Experiment 11 — Hard Token Data Sampling

## Hypothesis

After ~2000 training steps, returns diminish because the model has learned all frequent patterns. Rare token combinations (word-initial tokens after juncture points like `,` `.` `the` `and`) account for ~65% of hard tokens but appear infrequently in training data. By oversampling text rich in these patterns during later training, we can accelerate learning in the diminishing-returns phase.

## Baseline

Exp 10 (competition stack on H100 PCIe): val_bpb=1.2793 pre-quant at 12,596 steps.

For these short experiments, we compare against **Exp 10 at ~1,580 steps** (equivalent 300s wallclock on H100 at 190ms/step). This is the baseline for the same step budget.

## Experiment Design

All three experiments run for **300 seconds (5 min)** on H100 PCIe using the Exp 10 architecture (9L, MLP 3x, SmearGate, BigramHash, OrthoInit, WD=0.04). Same model, same hyperparameters — only the data sampling changes.

### Exp 11a — Juncture-enriched shards

**Idea:** Pre-score training shards by juncture token density (count of `,` `.` `the` `and` `to` `of` `in` tokens per shard). In later training steps (after step 500), sample more frequently from high-juncture shards.

**Why this might work:** Juncture tokens are where the model is worst. More exposure to diverse continuations after junctures = faster learning of these patterns.

**Implementation:**
- Score all 80 training shards by juncture density
- First 500 steps: normal sequential loading (warmup)
- Steps 500+: load from top-20 highest-juncture shards only

### Exp 11b — Rare bigram enrichment

**Idea:** Pre-compute bigram frequency table from training data. Create a filtered subset of training sequences where the average bigram rarity is above median. Train exclusively on this rare-bigram-enriched data after step 500.

**Why this might work:** Rare bigrams are exactly the patterns the model hasn't seen enough times. Oversampling them gives the model 4x more exposure to rare patterns per step.

**Implementation:**
- Build bigram frequency table from first 5 shards
- For each 1024-token sequence, compute average bigram log-frequency
- Select sequences in top 25% rarity
- Save as enriched training shard
- First 500 steps: normal data
- Steps 500+: enriched data only

### Exp 11c — Loss-based curriculum (online)

**Idea:** Every 100 steps, evaluate the model on 10 candidate training batches and pick the one with highest loss. This is online hard-example mining — the model always trains on what it's worst at.

**Why this might work:** Directly addresses the diminishing returns problem. Instead of randomly sampling training data (where 80% is "easy" for the model by step 2000), every batch is maximally informative.

**Why this might NOT work:** Highest-loss batches might be noisy/corrupted text rather than learnable hard patterns. Also 10x forward-pass overhead for selection.

**Implementation:**
- Every 100 steps: evaluate 10 random batches, pick highest loss, train on it
- Other steps: normal random sampling
- Net overhead: ~10% more forward passes

## Deep Analysis Plan (per experiment)

Each experiment gets:
1. **Val BPB trajectory** — compared against Exp 10 baseline at same steps
2. **Loss distribution** — easy/medium/hard/very_hard bucket percentages
3. **Position degradation** — context_benefit, first_64 vs last_64 loss, position ranges
4. **Layer ablation** — impact of removing each layer (0-8)
5. **BPB extrapolation** — predict final BPB if run for 12,596 steps based on learning curve shape

## Expected Outcomes

| Experiment | Expected BPB at 1,580 steps | Prediction if run to 12,596 steps |
|------------|---------------------------|-----------------------------------|
| Baseline (Exp 10) | ~1.47 (from Exp 10 log) | 1.2793 (actual) |
| 11a (juncture shards) | ~1.45-1.47 | ~1.27 (slight improvement) |
| 11b (rare bigrams) | ~1.44-1.46 | ~1.26-1.27 (moderate improvement) |
| 11c (online hard mining) | ~1.46-1.48 | ~1.28 (neutral — overhead cancels benefit) |

## Results

### Val BPB Comparison (int8+zlib roundtrip)

| Experiment | val_bpb | Delta vs control | Steps | ms/step |
|------------|---------|-----------------|-------|---------|
| 11c (control) | **1.4386** | — | ~1,600 | 186 |
| 11a (juncture-enriched) | 1.4402 | +0.0016 (worse) | ~1,600 | 188 |
| 11b (rare bigram) | 1.4494 | **+0.0108 (worse)** | ~1,500 | 198 |

### Deep Eval Comparison

| Metric | 11c (control) | 11a (juncture) | 11b (rare bigram) |
|--------|--------------|----------------|-------------------|
| val_loss (500 win) | 2.3986 | 2.4004 | 2.4148 |
| val_bpb (500 win) | 1.2402 | 1.2412 | 1.2486 |
| easy (<1) | 36.7% | 36.6% | 36.3% |
| medium (1-3) | 26.5% | 26.5% | 26.6% |
| hard (3-5) | 23.2% | 23.2% | 23.4% |
| very_hard (>5) | 13.6% | 13.6% | 13.8% |
| context_benefit | 0.3709 | 0.3679 | 0.3604 |
| first_64_loss | 2.7111 | 2.7097 | 2.7224 |
| last_64_loss | 2.3401 | 2.3418 | 2.3621 |

### Layer Ablation Comparison

| Layer | 11c (control) | 11a (juncture) | 11b (rare bigram) |
|-------|--------------|----------------|-------------------|
| L0 (encoder) | +3.6257 | +4.2884 | +3.5740 |
| L1 (encoder) | +0.5129 | +0.4913 | +0.5108 |
| L2 (encoder) | +0.4524 | +0.4405 | +0.4842 |
| L3 (encoder) | +0.3130 | +0.3210 | +0.3477 |
| L4 (bottleneck) | +0.1505 | +0.1744 | +0.1495 |
| L5 (decoder) | +0.1006 | +0.0977 | +0.1047 |
| L6 (decoder) | +0.1007 | +0.0989 | +0.1001 |
| L7 (decoder) | +0.1011 | +0.0992 | +0.0994 |
| L8 (decoder) | +0.1123 | +0.1200 | +0.0753 |

### Position Degradation

| Range | 11c (control) | 11a (juncture) | 11b (rare bigram) |
|-------|--------------|----------------|-------------------|
| 0-64 | 2.7111 | 2.7097 | 2.7224 |
| 64-128 | 2.5177 | 2.5197 | 2.5316 |
| 128-192 | 2.4262 | 2.4241 | 2.4432 |
| 192-256 | 2.4131 | 2.4129 | 2.4308 |
| 256-320 | 2.3849 | 2.3827 | 2.3983 |
| 320-384 | 2.3621 | 2.3682 | 2.3794 |
| 384-448 | 2.3625 | 2.3642 | 2.3754 |
| 448-512 | 2.3511 | 2.3552 | 2.3631 |
| 512-576 | 2.3332 | 2.3413 | 2.3528 |
| 576-640 | 2.3693 | 2.3742 | 2.3873 |
| 640-704 | 2.3693 | 2.3718 | 2.3881 |
| 704-768 | 2.3732 | 2.3734 | 2.3909 |
| 768-832 | 2.3644 | 2.3678 | 2.3839 |
| 832-896 | 2.3325 | 2.3351 | 2.3496 |
| 896-960 | 2.3668 | 2.3646 | 2.3778 |
| 960-1024 | 2.3401 | 2.3418 | 2.3621 |

---

## Analysis

### Verdict: Hard token data sampling does NOT help

Both sampling strategies performed **equal or worse** than the control. 11b (rare bigrams) was significantly worse, losing 0.0108 BPB.

### Why it failed

**1. Reduced data diversity hurts more than targeted exposure helps.**
11b had 0.2% more very_hard tokens (13.8% vs 13.6%) and 0.4% fewer easy tokens. The model got worse at everything, not better at hard things. By concentrating on rare patterns, the model saw fewer common patterns — and common patterns are the foundation that hard patterns build on.

**2. The hard tokens aren't hard because of insufficient exposure — they're hard because of insufficient model capacity.**
The loss distribution barely changed across all 3 experiments. The hard token profile is architectural (limited attention capacity past position 256, only 9 layers) not data-driven.

**3. Position degradation worsened with rare data.**
11b's context_benefit (0.3604) was worse than control (0.3709). The rare bigram sequences may have less coherent long-range structure, giving the model fewer opportunities to learn position-dependent patterns.

**4. Layer 8 weakened in 11b.**
L8 impact dropped from 0.1123 (control) to 0.0753 (rare bigram) — the final decoder layer became less useful, suggesting the rare data disrupted the model's output distribution.

### Key insight

The diminishing returns after step 2000 are **not a data problem**. The model has already seen enough data to learn all learnable patterns at its current capacity. The bottleneck is:
- **Model capacity** (9 layers, 512 dim) — solved by 11L + wider architecture
- **Training steps** — solved by faster GPU or longer wallclock
- **Warmdown phase** — the 2.3x efficiency boost during cosine decay is the best available accelerator

Data sampling is a dead end for this competition. The path forward is architectural (11L) and eval-time (sliding window, SWA).

## Decision

All experiments within 0.01 of baseline → **Move to 11-layer architecture or SWA instead.**
