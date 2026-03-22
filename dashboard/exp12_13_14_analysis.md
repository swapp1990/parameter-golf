# Experiments 12-14: SWA + SwiGLU + 11 Layers

## Baseline

Exp 10: val_bpb=1.2793 pre-quant, ~1.246 with sliding window (stride=256).
Architecture: 9L, 512d, MLP 3x ReLU², SmearGate, BigramHash, OrthoInit, WD=0.04.

## Experiment Summary

| Exp | Change | Steps | ms/step | val_bpb (pre-quant) | val_bpb (sliding) | Delta vs Exp 10 |
|-----|--------|-------|---------|---------------------|-------------------|-----------------|
| 10 (baseline) | Competition stack | 12,596 | 190 | 1.2793 | ~1.246 | — |
| 12 | + SWA | — | — | — | — | — |
| 13 | + SwiGLU | — | — | — | — | — |
| 14 | + 11L + Int5 MLP | — | — | — | — | — |

*(Results will be filled in as experiments complete)*

---

## Exp 12: SWA (Stochastic Weight Averaging)

### Config
Same as Exp 10 + SWA averaging every 100 steps during warmdown.

### Results
*(pending)*

### Deep Eval
*(pending)*

---

## Exp 13: SwiGLU

### Config
Same as Exp 10 but MLP uses SwiGLU activation with matched param count (hidden=1024).

### Results
*(pending)*

### Deep Eval
*(pending)*

---

## Exp 14: 11 Layers + Int5 MLP

### Config
11 layers, MLP 3x SwiGLU (if Exp 13 positive) or ReLU², int5 for MLP weights, int6 for attention.

### Results
*(pending)*

### Deep Eval
*(pending)*

---

## Comparative Deep Eval

### Loss Distribution
| Bucket | Exp 10 | Exp 12 | Exp 13 | Exp 14 |
|--------|--------|--------|--------|--------|
| easy (<1) | — | — | — | — |
| medium (1-3) | — | — | — | — |
| hard (3-5) | — | — | — | — |
| very_hard (>5) | — | — | — | — |

### Position Degradation
| Metric | Exp 10 | Exp 12 | Exp 13 | Exp 14 |
|--------|--------|--------|--------|--------|
| context_benefit | — | — | — | — |
| first_64_loss | — | — | — | — |
| last_64_loss | — | — | — | — |

### Layer Ablation
*(per experiment)*

### Key Findings
*(pending)*
