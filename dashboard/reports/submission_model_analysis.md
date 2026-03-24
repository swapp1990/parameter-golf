# Submission Model — Comprehensive Analysis

**Model**: 11L SwiGLU + XSA + SmearGate + OrthoInit + Muon WD + SWA
**Params**: 26,502,232 (quantized to 15.02 MB with int5-MLP + int6-attn + int8-embed + zstd)
**Best BPB**: 1.1573 (LoRA TTT) / 1.1914 (standard eval)

---

## 1. Journey Summary

From val_bpb=3.10 (Exp 1) to 1.1573 (Exp 17 + LoRA TTT) in 18 experiments over 5 days.

| Experiment | val_bpb | What changed | Cost |
|-----------|---------|-------------|------|
| 1 (baseline) | 3.10 | Wrong batch size | — |
| 2 | 1.46 | Fixed batch to 65K | — |
| 6 | 1.312 | 1200s training, warmdown=600 | — |
| 8a | 1.299 | Warmdown=3000 | — |
| 10 | 1.283 | + SmearGate, BigramHash, OrthoInit, MLP 3x, WD | ~$4 |
| 13 | — | SwiGLU > ReLU² (+0.004) | ~$0.50 |
| 14 | 1.278 | + 11 layers + SWA | ~$3.60 |
| 14-8x | 1.202 | Same on 8xH100 (batch=524K) | ~$3.60 |
| 15 | 1.187 | + seq_len=2048 | ~$3.60 |
| 17 | 1.183 | + XSA (last 4 layers) | ~$3.60 |
| Quantized | 1.191 | int5+int6+int8+zstd (15 MB) | local |
| + LoRA TTT | **1.157** | Per-document adaptation at eval | ~$2.90 |

Total compute cost: ~$50 across all experiments.

### What the model looks like

```
Input tokens
    ↓
Token Embedding (1024 vocab, 512 dim) → int8 quantized
    ↓
SmearGate (blend with previous token embedding)
    ↓
RMSNorm
    ↓
┌──────────────────────────────────────────┐
│ 11 Transformer Blocks (U-Net)            │
│                                          │
│ Encoder (L0-L4):     save skip outputs   │
│ Bottleneck (L5):     U-Net crossover     │
│ Decoder (L6-L10):    add skip inputs     │
│                                          │
│ Each block:                              │
│   resid_mix → Attn → scale → MLP → scale│
│                                          │
│ Attention: 8 heads, 4 KV (GQA)          │
│   L7-L10: XSA (remove self-value proj)  │
│                                          │
│ MLP: SwiGLU 3x (gate·up→proj)           │
│   int5 quantized                         │
│                                          │
│ OrthoInit: all weights orthogonal        │
│ Muon optimizer with WD=0.04             │
└──────────────────────────────────────────┘
    ↓
RMSNorm → Tied embedding projection → logit softcap
    ↓
Output probabilities
```

---

## 2. Architecture Decisions

Each technique, why we added it, and its measured impact:

| Technique | What it does | Why we added it | Measured BPB gain |
|-----------|-------------|-----------------|-------------------|
| **SmearGate** | Blends previous token embedding into current | Gives bigram context at embedding layer — frees attention from local lookups | ~0.012 (critical: +1.80 loss without it) |
| **SwiGLU** | Gated MLP: `swish(gate(x)) * up(x)` | More parameter-efficient than ReLU². Activates output layer. | +0.004 |
| **11 Layers** | 2 extra transformer blocks | Eliminates dead middle layers. More capacity. | +0.009 |
| **XSA** | Removes self-value projection from attention output in L7-L10 | Forces attention to contribute context, not redundant self-info | +0.005 |
| **OrthoInit** | Orthogonal weight initialization | Required for SmearGate. Creates independent channels. | included in SmearGate |
| **Muon WD 0.04** | Decoupled weight decay in optimizer | Shrinks weights → better quantization + generalization | +0.003 |
| **SWA** | Average 15 checkpoints during warmdown | Smoother weights → better quantization | +0.003 |
| **seq_len 2048** | Train on longer sequences | Model learns long-range context. Context benefit 0.38→0.69. | +0.015 |
| **BigramHash removed** | Removed hash table for token pairs | Exp 16 proved zero contribution. Saves 590K params. | 0 (free params) |

### What we removed

**BigramHash**: A 4096-bucket hash table mapping token pairs to embeddings. Exp 16 ablation showed exactly zero contribution — SmearGate captures all bigram information. Removing it saved 590K params, helping the model fit in 16MB.

---

## 3. Where the 1.19 BPB Goes (Bits Budget)

Analyzed 1,024,000 tokens from the quantized submission model. Average loss: 2.06 nats.

### Token Difficulty Categories

| Category | % of Tokens | % of Bits | Avg Loss | What's in it |
|----------|------------|-----------|----------|--------------|
| Easy (<1 nat) | **44.0%** | ~5% | 0.23 | Word completions, common patterns |
| Medium (1-3) | 25.4% | ~24% | 1.96 | Common words in context |
| Hard (3-5) | 19.8% | ~37% | 3.90 | Word-initial predictions, content after boundaries |
| Very Hard (5+) | **10.8%** | ~34% | 6.39 | Names, numbers, rare patterns |

**44% of tokens are easy** — nearly half the text is trivially predictable. But the **very hard 10.8%** eat 34% of all bits. The hard + very hard combined (30.6% of tokens) account for over 70% of the model's total bit budget.

### What makes tokens hard

The hard tokens (loss 3-5) break down as:

| Sub-category | Count | % of Hard | Example |
|-------------|-------|-----------|---------|
| Word-initial letters | 65,895 | **32.5%** | `the ▁s___` — is it "store", "street", "school"? |
| Function words | 11,368 | 5.6% | `worked ▁___` — "in"? "for"? "on"? |
| After period | 9,458 | 4.7% | `. ▁___` — which word starts the next sentence? |
| After "the" | 8,742 | 4.3% | `the ▁c___` — "city"? "car"? "country"? |

Word-initial single letters dominate because the 1024-token vocabulary forces the model to predict words letter by letter. `▁s` could be the start of 200+ different words.

### Juncture analysis

| Context | Avg Loss | Ratio |
|---------|----------|-------|
| After juncture (`, . the and`) | **3.37** | 1.78x harder |
| Not after juncture | 1.89 | baseline |
| Word-initial token | **3.41** | 2.82x harder |
| Word-middle/end token | 1.21 | easy |

SmearGate reduced this gap significantly (without it, the gap would be even larger — removal causes +1.80 loss). But word-initial prediction remains the model's biggest challenge.

### Document cold-start

The validation set has **50,000 documents** (median length 733 tokens). The first tokens of each document are "cold" — the model has no knowledge of the document's topic.

This is where **LoRA TTT helps**: it adapts the model to each document before scoring, reducing cold-start loss. Our TTT result (-0.034 BPB) directly targets this.

---

## 4. Inside the Model

### Layer Utilization

Every layer contributes. No dead layers (unlike our earlier 9L model where layers 4-6 were dead).

| Layer | Role | Ablation Impact | Interpretation |
|-------|------|----------------|----------------|
| **L0** | encoder | **+5.35** | Critical: processes embeddings + SmearGate signal |
| L1 | encoder | +0.92 | Strong: second-pass encoding |
| L2 | encoder | +0.52 | |
| L3 | encoder | +0.41 | |
| L4 | encoder | +0.24 | |
| L5 | bottleneck | +0.24 | U-Net crossover |
| L6 | decoder | +0.29 | |
| L7 | decoder [XSA] | +0.24 | |
| L8 | decoder [XSA] | +0.19 | Weakest, but still active |
| L9 | decoder [XSA] | +0.19 | |
| **L10** | decoder [XSA] | **+3.15** | Critical: output predictions |

L0 and L10 are by far the most important — the input and output endpoints. The middle layers (L4-L9) contribute 0.19-0.29 each, remarkably uniform. This even distribution is a sign of a healthy architecture.

### Head Importance

Two heads dominate: **L0H6** (impact +0.77) and **L0H0** (impact +0.58). These are the first layer's primary heads — they handle the initial token-level processing. All other heads have impact < 0.12. No dead heads (all > 0.005).

### MLP Importance

MLP layers follow the same pattern as full-layer ablation:

| Layer | MLP Impact | Observation |
|-------|-----------|-------------|
| L0 | +5.60 | MLP is MORE important than attention at L0 |
| L10 | +3.13 | Nearly all of L10's contribution is MLP |
| L1-L9 | 0.18-1.00 | Steady decline from L1 to L9 |

**L0 and L10 are MLP-dominated.** The SwiGLU gating mechanism makes these endpoint MLPs extremely powerful — they do the heavy lifting of embedding processing (L0) and output prediction (L10). This is why SwiGLU was better than ReLU² — the gating lets these critical MLPs specialize.

### SmearGate

Removing SmearGate causes **+1.80 loss** — catastrophic. The model has completely internalized the previous-token blending signal. Every layer downstream depends on it.

### Position Utilization

| Position Range | Avg Loss | What's happening |
|---------------|----------|------------------|
| 0-128 | 2.47 | High — minimal context |
| 128-256 | 2.12 | Dropping fast — context building |
| 256-512 | 2.07 | Good context |
| 512-1024 | 2.02 | Settled |
| 1024-2048 | 2.02 | Stable — full context utilized |

Context benefit: **0.69** (first_64 loss 2.70 vs last_64 loss 2.01). Training on seq_len=2048 taught the model to use long-range context effectively. Loss stabilizes around position 512 — beyond that, additional context provides diminishing returns.

---

## 5. Confidence Analysis

| Quadrant | % | What it means |
|----------|---|---------------|
| **Confident Right** | **44.2%** | Model knows and is correct. Ideal. |
| Uncertain Right | 13.2% | Model unsure but happens to be right. |
| Confident Wrong | **5.8%** | Model sure but wrong. Calibration target. |
| **Uncertain Wrong** | **36.8%** | Model doesn't know, gets it wrong. Capacity limit. |

**Good news**: confident-wrong is only 5.8% — the model is well-calibrated. It rarely makes overconfident mistakes.

**Challenge**: 36.8% uncertain-wrong. These tokens are at the model's capacity limit. Improving them requires either more parameters (bigger model) or test-time adaptation (TTT).

---

## 6. What Didn't Work

| Technique | Experiment | Result | Why it failed |
|-----------|-----------|--------|---------------|
| Register token | Exp 7-8b | +0.002 worse | Step overhead > marginal juncture benefit |
| Layer looping + wider | Exp 9 | +0.034 worse | Step time overhead from wider dim |
| Data sampling (juncture) | Exp 11a | +0.002 worse | Shard-level too coarse, data is homogeneous |
| Data sampling (rare bigram) | Exp 11b | +0.011 worse | Reduced diversity hurts |
| Hard example mining | Exp 11d | +0.040 worse | Post-hoc fine-tuning destroys Muon weight geometry |
| Partial RoPE (16/64) | Exp 18 | +0.015 worse | 64 head_dim too small to sacrifice 75% |
| EMA (replacing SWA) | Exp 18 | +0.015 worse | Over-smoothed warmdown weights |
| LN Scale | Exp 16 | +0.003 (noise) | Redundant with existing learned scales |
| Heterogeneous MLP | Exp 16 | +0.011 worse | Param reduction in middle > concentration benefit |
| XSA (300s test) | Exp 16 | -0.001 (noise) | Too few steps to detect. Full run gave -0.005. |
| BigramHash | Exp 16 | 0.000 | SmearGate makes it redundant |
| SGD TTT | Exp 17 addendum | +0.018 worse | Modifying dequantized weights directly breaks them |

**Key lesson**: At this model size and training regime, most "clever" techniques don't help. The gains came from proven competition techniques (SmearGate, SwiGLU, int6 quantization, 11L) and disciplined hyperparameter tuning (warmdown=3000, WD=0.04).

---

## 7. Evaluation Methods

### Three approaches tested

| Method | val_bpb | Delta | How it works |
|--------|---------|-------|-------------|
| Standard | 1.1914 | — | Non-overlapping 2048 windows |
| Sliding window (s256) | ~1.170 | -0.022 | Overlapping windows, score last 256 tokens each |
| **LoRA TTT** | **1.1573** | **-0.034** | Adapt LoRA per document, then score |

### Why they can't combine

Sliding window operates over the **full token stream** (continuous, overlapping windows across document boundaries). TTT operates **per document** (adapt weights, score, reset). We tested per-document sliding window and it gave **worse** results (+0.021) because:
- Short docs (median 733 tokens < 2048 seq_len) get no sliding window benefit
- The scoring covers different token subsets, breaking BPB comparability

### LoRA TTT wins

LoRA TTT (-0.034) beats sliding window (-0.022) because it directly addresses the document cold-start problem. The model adapts to each document's topic and style before scoring, reducing uncertainty on the 36.8% of tokens that are "uncertain-wrong."

**Note**: SGD TTT (modifying base weights directly) failed (+0.018 worse). LoRA works because it adds a small perturbation without disturbing the quantized base weights.

---

## 8. Next Steps

### Remaining gap to competition leader

| | BPB |
|---|-----|
| Our submission (LoRA TTT) | 1.1573 |
| Competition pending #1 (#198) | 1.1326 |
| Gap | **0.025** |

### Where the 0.025 gap comes from (estimated)

| Source | Est. BPP | How to close |
|--------|---------|-------------|
| FA3 (Flash Attention 3) | ~0.005 | Use FA3 kernel on H100 — more steps in same wallclock |
| Better LR schedule (0.025 + momentum warmup) | ~0.005 | #198 uses LR=0.025, momentum 0.99 with 1500-step warmup |
| Sliding window on 8xH100 (in-training eval) | ~0.005 | Run sliding window as part of training eval loop |
| 3-seed averaging | ~0.003 | Statistical noise reduction |
| Larger BigramHash (2048 buckets) | ~0.002 | #198 kept BigramHash — may help with their int6 budget |
| Adam WD + Muon WD combined | ~0.002 | #198 uses WD on both optimizers |
| Remaining | ~0.003 | Hyperparameter tuning, unknown |

### Most promising techniques not yet tried

1. **FA3**: Zero-effort speed boost on H100. ~5% more training steps.
2. **LR + momentum schedule from #198**: Their specific LR=0.025, momentum 0.99 with 1500-step warmup from 0.92 differs from ours.
3. **Vocabulary increase**: Our bits budget shows 32.5% of hard tokens are word-initial letters — a vocabulary problem. A 2048-vocab tokenizer would reduce this, but increases embedding table size.
4. **QAT (late)**: Quantization-aware training in the last 4% of steps. Reduces quant penalty from +0.009 to ~+0.003.
