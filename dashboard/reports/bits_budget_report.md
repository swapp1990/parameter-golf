# Bits Budget Report — Where Does the Model Spend Its Bits?

**Model**: Exp 17 (11L SwiGLU + XSA, val_bpb=1.1826)
**Data**: 2,048,000 tokens from FineWeb validation set (1000 windows × 2048 tokens)
**Average loss**: 2.615 nats/token

---

## 1. The Big Picture — Where the Bits Go

The model spends bits (information) to predict each token. Easy tokens (like "the" after "of") cost almost nothing. Hard tokens (like which word starts a new sentence) cost a lot.

### Bits Budget by Category

```
                              % of Bits                    % of Tokens
Easy (<1 nat)       ██                    3.5%    ████████████████         31.3%
Medium (1-3 nats)   ██████████            21.6%   ██████████████           28.2%
Hard Learnable      ██████████████████████████████████  67.0%   ██████████████████   37.2%
Unpredictable (5+)  ████                  8.0%    ██                        3.4%
```

| Category | % of Tokens | % of Bits | Avg Loss | What it means |
|----------|------------|-----------|----------|---------------|
| Easy (<1 nat) | 31.3% | **3.5%** | 0.29 | Model nails these. Mostly word continuations, common patterns. |
| Medium (1-3 nats) | 28.2% | **21.6%** | 2.00 | Model gets close. Common words in somewhat predictable positions. |
| Hard Learnable (3-5 nats) | 37.2% | **67.0%** | 4.71 | Model struggles. Word-initial tokens, content words after boundaries. |
| Unpredictable (5+ nats) | 3.4% | **8.0%** | 6.26 | No model can predict these. Numbers, names, rare patterns. |

**The key insight: 37% of tokens eat 67% of the bits.** These "hard but learnable" tokens are where improvements pay off. The easy tokens (31% of tokens) contribute only 3.5% of the total cost — they're already optimized.

The "unpredictable" floor uses 8% of bits on only 3.4% of tokens. These are our theoretical minimum — no amount of model improvement can reduce them.

### What Are the Hard Learnable Tokens?

The 67% breaks down into 5 sub-categories:

| Sub-Category | % of Hard Bits | What's happening | Solution | Est. Gain |
|---------|---------------|-----------------|----------|-----------|
| **Word-initial letters** | 66% | 1024 vocab splits words into letters. Model guesses which of ~200 words start with that letter. | Larger vocab (4096) | -0.01 to -0.015 |
| | | `She walked into the` **`▁s`**`___` → "store"? "street"? "school"? "small"? "sun"? (loss=3.65) | | |
| **Function words** | 20% | Multiple prepositions are grammatically valid. Needs deep meaning understanding. | More capacity | -0.003 to -0.008 |
| | | `The team worked` **`▁___`** → "in" the office? "for" the company? "on" the project? (loss=3.24) | | |
| **Sentence starters** | 6% | After a period, almost any word can begin the next sentence. | TTT (adapt to doc) | -0.003 to -0.010 |
| | | `The weather was nice.` **`▁___`** → "The" next day? "She" walked? "It" was? "However"? (loss=3.50) | | |
| **Content after "the"** | 4% | Model knows a noun follows "the" but can't narrow down which noun without topic understanding. | Longer context | -0.002 to -0.005 |
| | | `He drove to the` **`▁c`**`___` → "city"? "coast"? "corner"? "church"? "clinic"? (loss=3.27) | | |
| **Multi-char starts** | 4% | Like word-initial but with 2-3 letter prefix. Less ambiguous, still many options. | Larger vocab | included above |
| | | `The results were` **`▁re`**`___` → "released"? "remarkable"? "recently"? "reported"? (loss=4.33) | | |

**The biggest insight: 66% of hard bits are a VOCABULARY problem.** The 1024-token vocabulary forces the model to predict words letter-by-letter. A 4096-token vocabulary would merge common words into single tokens, eliminating this ambiguity.

---

## 2. What the Losses Look Like — Distribution

```
Token Count
  │
  │ ████
  │ █████
  │ ██████
  │ ████████
  │ █████████
  │ ██████████
  │ ███████████░░░░░░░░
  │ ████████████░░░░░░░░░░░░░░░
  │ █████████████░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒
  ├────────┬─────────┬──────────┬───────────────────── Loss (nats)
  0        1         3          5                    10
       Easy     Medium       Hard        Unpredictable
```

The peak is around 0-1 nats (easy tokens), with a long tail stretching past 8 nats. The bump around 4-5 nats is the "hard learnable" peak — this is where the model's capacity limit shows up. A bigger or smarter model could push this bump to the left.

---

## 3. Position Matters — Later Tokens Are Surprisingly Worse

```
Loss (nats)
 2.80 │                                                          ██████
 2.75 │                                                    ██████
 2.70 │                                              ██████
 2.65 │                                        ██████
 2.60 │                          ████████████████
 2.55 │                ██████████
 2.50 │  ████████████
 2.45 │        ██
 2.40 │
      ├─────────────────────────────────────────────────────────────
      0       256      512      768     1024    1280    1536    2048
                        Position in 2048-token window
```

| Position Range | Avg Loss | Observation |
|---------------|----------|-------------|
| 0-128 | 2.525 | Moderate — fresh window start |
| 128-256 | 2.445 | **Best** — enough context, coherent text |
| 256-512 | 2.489 | Slightly worse |
| 512-1024 | 2.558 | Getting worse |
| 1024-1536 | 2.669 | Significantly worse |
| 1536-2048 | 2.763 | **Worst** — 0.32 nats above best |

**Surprising finding: later positions are WORSE, not better.** This is the opposite of what context theory predicts. The likely cause: the validation set is chunked into non-overlapping 2048-token windows. Late-window tokens may hit document boundaries or topic shifts within the window.

**This means sliding window evaluation helps most for late-position tokens** — giving them full 2048 context instead of partial window context.

---

## 4. The Most Expensive Bigrams

These token pairs cost the most total bits (frequency × average loss):

```
. → The       ████████████████████████████████████████████  8,982  (n=4052, avg=2.22)
, → and       █████████████████████████████████████████     7,833  (n=3541, avg=2.21)
_ → 1         █████████████████████████████████████         7,479  (n=5682, avg=1.32)
_ → 2         ██████████████████████████████                6,096  (n=5274, avg=1.16)
. → A          ██████████████████████████                   5,454  (n=1736, avg=3.14)
, → the        █████████████████████████                    5,315  (n=2006, avg=2.65)
. → S          █████████████████████████                    5,250  (n=1754, avg=2.99)
of → the       █████████████████████████                    5,155  (n=4806, avg=1.07)
. → I          ████████████████████████                     5,100  (n=1904, avg=2.68)
s → ,          ███████████████████████                      4,894  (n=2484, avg=1.97)
```

**Transitions after punctuation dominate.** `. → The` is the single most expensive bigram — after a period, which word starts the next sentence? The model knows a sentence is starting but can't predict WHICH word. This is the fundamental capacity limitation of a 27M-param model.

---

## 5. Confidence vs Correctness

```
         ┌─────────────────────┬─────────────────────┐
         │                     │                     │
         │    CONFIDENT RIGHT  │   UNCERTAIN RIGHT   │
  Right  │       38.0%         │       7.3%          │
         │     (ideal)         │     (lucky)         │
         │                     │                     │
         ├─────────────────────┼─────────────────────┤
         │                     │                     │
         │   CONFIDENT WRONG   │  UNCERTAIN WRONG    │
  Wrong  │      12.0%          │      42.7%          │
         │  (calibration bug)  │  (capacity limit)   │
         │                     │                     │
         └─────────────────────┴─────────────────────┘
              Confident              Uncertain
```

| Quadrant | % of Tokens | What it means | How to fix |
|----------|------------|---------------|------------|
| Confident Right | 38.0% | Ideal — model knows and is correct | Already good |
| Uncertain Right | 7.3% | Model guesses, happens to be right | More capacity helps |
| **Confident Wrong** | **12.0%** | Model is sure but wrong | TTT, calibration, label smoothing |
| **Uncertain Wrong** | **42.7%** | Model doesn't know, gets it wrong | More capacity, longer context |

**42.7% of tokens are uncertain-wrong** — the model honestly doesn't know. These need more capacity (bigger model, more context) to improve.

**12.0% are confident-wrong** — the model THINKS it knows but is wrong. These are the highest-leverage targets for test-time training (TTT), which adapts the model to each specific document.

---

## 6. Juncture Tokens — The Hardest Positions

```
Avg Loss (nats)
  4.0  ████████████████████            ████████████████████
       After Juncture                  Word-Initial
       3.95                            3.97

  2.5  ████████████████
       Not After Juncture
       2.44

  1.7                                  █████████████
                                       Word-Middle
                                       1.75
```

| Context | Avg Loss | % of Tokens | Observation |
|---------|----------|------------|-------------|
| After juncture (`, . the and`) | **3.95** | 11.6% | 62% harder than average |
| Not after juncture | 2.44 | 88.4% | Normal |
| Word-initial (starts new word) | **3.97** | ~50% | 127% harder than word-middle |
| Word-middle/end | 1.75 | ~50% | Easy — just completing a known word |

**Word-initial tokens after syntactic boundaries are where the model bleeds bits.** SmearGate provides bigram context (knowing the previous token) which helps, but the gap remains 1.5 nats. This is a vocabulary problem — with 1024 tokens, there are ~500 possible word-initial tokens, and the model can't narrow it down enough.

---

## 7. The Most Expensive Individual Tokens

| Rank | Token | Total Cost | Count | Avg Loss | Why expensive |
|------|-------|-----------|-------|----------|---------------|
| 1 | `,` | 93,399 | 44,106 | 2.12 | Very frequent, moderately hard |
| 2 | `.` | 88,994 | 43,533 | 2.04 | Very frequent, moderately hard |
| 3 | `the` | 71,629 | 38,830 | 1.84 | Frequent, context-dependent |
| 4 | `_` (space) | 63,647 | 26,230 | 2.43 | Precedes unknown words |
| 5 | `and` | 60,500 | 21,831 | 2.77 | Common but not always predictable |
| 6 | `a` | 57,893 | 20,150 | 2.87 | Which article? Context-dependent |
| 7 | `in` | 52,802 | 16,290 | 3.24 | Preposition, many contexts |
| 8 | `s` (word-initial) | 49,228 | 13,499 | 3.65 | Which s-word? Very ambiguous |
| 9 | `p` (word-initial) | 47,446 | 13,178 | 3.60 | Which p-word? Very ambiguous |
| 10 | `c` (word-initial) | 45,163 | 12,415 | 3.64 | Which c-word? Very ambiguous |

**Function words dominate by total cost** (frequency × loss). Word-initial letters (`s` `p` `c` `f` `m` `d`) are expensive per-token (3.6-4.0 avg loss) because predicting which word starts with a given letter is fundamentally ambiguous at 1024 vocabulary.

---

## 8. Improvement Potential — Can We Reach 1.08 BPP?

### What's recoverable

| Source | Current % of Bits | Floor | Recoverable |
|--------|------------------|-------|-------------|
| Unpredictable tokens | 8.0% | 8.0% (can't improve) | 0% |
| Hard learnable | 67.0% | ~40% (better model) | ~27% |
| Medium | 21.6% | ~15% (better model) | ~7% |
| Easy | 3.5% | ~3% (near floor) | ~0.5% |

### Realistic improvements from known techniques

| Technique | BPP Gain | Status | What it targets |
|-----------|---------|--------|-----------------|
| Sliding window eval (stride=256) | **-0.026** | **Measured** (1.1826 → ~1.157) | Position degradation (Section 3) |
| Int5+Int6 quantization + QAT | **-0.010** | Not yet implemented | Quantization penalty |
| Test-time training (TTT) | **-0.010 to -0.033** | Not yet implemented | Confident-wrong tokens (Section 5) |
| Larger vocabulary (2048+) | **-0.005 to -0.015** | Not yet implemented | Word-initial ambiguity (Section 1) |
| More parameters (12L or wider) | **-0.005 to -0.010** | Not yet implemented | Uncertain-wrong tokens (Section 5) |
| Remove BigramHash + reallocate | **-0.002** | Validated (Exp 16: zero loss) | Free params |
| **Total realistic** | **-0.06 to -0.10** | | |

### Verdict

**Current (standard eval)**: 1.1826 BPP
**With sliding window (already measured)**: ~1.157 BPP
**Realistic target after all techniques**: ~1.06-1.10 BPP
**Stretch target**: ~1.05 BPP (requires ALL techniques stacking perfectly)

The sliding window gain is already confirmed — that's 0.026 BPP for free at eval time. Starting from 1.157:
1. **Int5+Int6 quantization + QAT** (-0.010) → ~1.147
2. **TTT** (-0.010 to -0.033) → ~1.114-1.137
3. **Larger vocabulary** (-0.005 to -0.015) → ~1.10-1.13

Getting to 1.08 requires TTT delivering its full -0.033 AND a vocabulary increase working well. Achievable but aggressive.
