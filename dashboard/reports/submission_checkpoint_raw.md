# Checkpoint Analysis: submission

| Metric | Value |
|--------|-------|
| Parameters | 26,502,232 |
| Layers | 11 |
| Seq Length | 2048 |
| Val Tokens Analyzed | 62021846 |

## Loss Distribution

| Category | % of Tokens | Avg Loss |
|----------|------------|----------|
| easy(<1) | 44.0% ██████████████████████ | 0.23 |
| medium(1-3) | 25.4% ████████████ | 1.96 |
| hard(3-5) | 19.8% █████████ | 3.90 |
| very_hard(5+) | 10.8% █████ | 6.39 |

## Hard Token Breakdown (loss 3-5)

Total hard tokens: 202717

| Sub-category | Count | % of Hard |
|-------------|-------|-----------|
| Word-initial letters | 65895 | 32.5% |
| Function words | 11368 | 5.6% |
| After period | 9458 | 4.7% |
| After "the" | 8742 | 4.3% |

## Position Analysis

- First 64 tokens avg loss: **2.7034**
- Last 64 tokens avg loss: **2.0095**
- Context benefit: **0.6939**

| Range | Avg Loss |
|-------|----------|
| 0-128 | 2.4711 |
| 128-256 | 2.1212 |
| 256-384 | 2.0722 |
| 384-512 | 2.0595 |
| 512-640 | 2.0220 |
| 640-768 | 2.0203 |
| 768-896 | 2.0128 |
| 896-1024 | 2.0200 |
| 1024-1152 | 2.0082 |
| 1152-1280 | 2.0107 |
| 1280-1408 | 2.0326 |
| 1408-1536 | 2.0071 |
| 1536-1664 | 2.0165 |
| 1664-1792 | 2.0359 |
| 1792-1920 | 2.0336 |
| 1920-2048 | 2.0242 |

## Document Analysis

- Documents: 50000
- Mean length: 1240 tokens
- Median length: 733 tokens

## Layer Ablation

Base loss: 1.9767

| Layer | Impact | |
|-------|--------|---|
| L0 (encoder) | +5.3453 | `########################################` |
| L1 (encoder) | +0.9246 | `#########` |
| L2 (encoder) | +0.5229 | `#####` |
| L3 (encoder) | +0.4123 | `####` |
| L4 (encoder) | +0.2388 | `##` |
| L5 (bottleneck) | +0.2376 | `##` |
| L6 (decoder) | +0.2892 | `##` |
| L7 (decoder) | +0.2444 | `##` |
| L8 (decoder) | +0.1861 | `#` |
| L9 (decoder) | +0.1934 | `#` |
| L10 (decoder) | +3.1520 | `###############################` |

## Head Ablation

Notable heads (impact > 0.01): 88

| Head | Impact |
|------|--------|
| L0H6 | +0.7748 |
| L0H0 | +0.5832 |
| L1H7 | +0.1178 |
| L7H7 | +0.0817 |
| L6H6 | +0.0779 |
| L5H4 | +0.0686 |
| L3H6 | +0.0657 |
| L4H3 | +0.0571 |
| L2H4 | +0.0542 |
| L3H1 | +0.0489 |
| L5H3 | +0.0489 |
| L6H7 | +0.0486 |
| L7H6 | +0.0468 |
| L5H1 | +0.0467 |
| L2H3 | +0.0455 |
| L0H2 | +0.0449 |
| L0H5 | +0.0444 |
| L7H2 | +0.0440 |
| L0H7 | +0.0434 |
| L4H7 | +0.0427 |

## Component Ablation

- SmearGate removal: **+1.8002**

### MLP Ablation (per layer)

| Layer | MLP Impact |
|-------|-----------|
| L0 | +5.6028 |
| L1 | +0.9974 |
| L2 | +0.6210 |
| L3 | +0.4625 |
| L4 | +0.2538 |
| L5 | +0.1877 |
| L6 | +0.1843 |
| L7 | +0.1893 |
| L8 | +0.1774 |
| L9 | +0.1786 |
| L10 | +3.1325 |

## Entropy Quadrants

```
  Confident Right:  44.2%   Uncertain Right: 13.2%
  Confident Wrong:  5.8%   Uncertain Wrong: 36.8%
```

## Juncture Analysis

- After juncture: **3.368** (11.6% of tokens)
- Not after juncture: **1.888**
- Word-initial: **3.405**
- Not word-initial: **1.207**

## Most Expensive Bigrams

| Prev | Cur | Total Cost | Count | Avg |
|------|-----|-----------|-------|-----|
| . | ▁The | 4061 | 2017 | 2.01 |
| , | ▁and | 3314 | 1763 | 1.88 |
| . | ▁A | 2710 | 885 | 3.06 |
| ▁of | ▁the | 2653 | 2484 | 1.07 |
| ▁ | 1 | 2584 | 2597 | 0.99 |
| ▁in | ▁the | 2354 | 1754 | 1.34 |
| . | ▁S | 2324 | 861 | 2.70 |
| ▁ | 2 | 2234 | 2582 | 0.87 |
| , | ▁the | 2212 | 989 | 2.24 |
| ▁the | ▁c | 2152 | 696 | 3.09 |
| . | ▁I | 2143 | 926 | 2.31 |
| . | ▁B | 2066 | 650 | 3.18 |
| s | , | 1944 | 1227 | 1.58 |
| ▁the | ▁s | 1926 | 726 | 2.65 |
| ▁the | ▁f | 1926 | 633 | 3.04 |

## Most Expensive Tokens

| Token | Total Cost | Count | Avg |
|-------|-----------|-------|-----|
| , | 36943 | 22218 | 1.66 |
| ▁the | 33845 | 19276 | 1.76 |
| . | 30965 | 21768 | 1.42 |
| ▁and | 28149 | 11066 | 2.54 |
| ▁ | 25794 | 12829 | 2.01 |
| ▁a | 25446 | 9815 | 2.59 |
| ▁in | 22592 | 8173 | 2.76 |
| ▁c | 21794 | 6200 | 3.52 |
| ▁s | 21572 | 6836 | 3.16 |
| ▁p | 20731 | 6338 | 3.27 |