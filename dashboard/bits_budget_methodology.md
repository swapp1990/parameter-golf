# Bits Budget Analysis — Methodology

## Why This Analysis

Our model scores 1.1826 BPP on the FineWeb validation set. We want to reduce this by 0.1 to ~1.08. But before trying random techniques, we need to understand **where the 1.1826 bits per byte are being spent**.

Every token the model predicts costs some number of bits. Some tokens are inherently unpredictable (a person's name appearing for the first time), some are hard but learnable (the word after a comma in a specific grammatical context), and some are nearly free (the "e" in "the"). If most of our bits are going to unpredictable tokens, no amount of model improvement will help — we need to know our theoretical floor.

This analysis answers three questions:
1. **What is the theoretical minimum BPP for this validation set?** (How much of the text is inherently unpredictable?)
2. **Where is our model wasting bits?** (Which token categories have the most room for improvement?)
3. **What specific improvements would recover the most bits?** (Prioritized list of techniques with estimated BPP contribution)

---

## Step 1: Per-Token Loss Map

### What we do
Run the Exp 17 checkpoint on the entire validation set (~62M tokens) and record for every token:
- **loss**: cross-entropy loss in nats (how many bits this token costs)
- **token_id**: the target token ID (0-1023)
- **prev_token_id**: the token immediately before (for bigram analysis)
- **position**: position within the 2048-token window
- **top1_correct**: whether the model's highest-probability prediction matched the target
- **entropy**: Shannon entropy of the model's output distribution (how uncertain the model is)
- **top1_prob**: probability assigned to the most likely next token

### Why
The per-token loss tells us exactly where bits are being spent. Aggregating by different dimensions (token type, position, bigram, etc.) reveals which categories are the biggest bit sinks and which have the most room for improvement.

### What we learn
- The distribution of per-token losses (how many tokens are easy vs hard)
- Which specific tokens cost the most bits in aggregate
- Whether losses are concentrated in specific positions, contexts, or token types

---

## Step 2: Token Category Classification

### What we do
Classify every token in the validation set into categories based on predictability:

**Category A — Structurally unpredictable (floor tokens)**
- First token of a new document/paragraph (no context available)
- Proper nouns appearing for the first time in context
- Numbers, dates, URLs, email addresses
- Foreign language fragments
- Random identifiers and codes

These tokens are unpredictable by ANY model — they're essentially random bits that must be stored, not predicted. The bits spent on these are our **floor**.

**Category B — Hard but learnable (high-value targets)**
- Word-initial tokens after syntactic boundaries (`,` `.` `the` → next word)
- Rare but valid English words
- Context-dependent word choices (requires understanding meaning)
- Tokens requiring long-range context (references to earlier text)

These are tokens where a better model COULD do better. Our earlier hard token analysis showed ~65% of very-hard tokens fall here. These represent our **opportunity**.

**Category C — Medium difficulty (incremental gains)**
- Common word completions (word-middle tokens)
- Grammatical function words in predictable positions
- Repeated patterns within a document

A better model would get marginal improvements here. Moderate opportunity.

**Category D — Easy (already optimized)**
- High-frequency tokens in predictable contexts
- Word-final tokens after unambiguous prefixes
- Punctuation following standard patterns

The model already handles these well. Little room for improvement.

### Why
This classification tells us the **theoretical ceiling** for improvement. If Category A accounts for X% of total bits, then even a perfect model can only improve by (100-X)%. It also tells us which categories to target — Category B tokens are where architectural and training improvements pay off.

### How we classify
- **A (unpredictable)**: Tokens where even a very large LM (GPT-4 class) would have high loss. We approximate this by looking at tokens with loss > 8 nats AND low model entropy (the model is confident but wrong — typically noise) OR tokens in specific pattern categories (numbers, URLs, etc.)
- **B (hard but learnable)**: Loss > 3 nats, follows a juncture token, valid English word-initial
- **C (medium)**: Loss 1-3 nats
- **D (easy)**: Loss < 1 nat

---

## Step 3: Positional Analysis

### What we do
Compute average loss as a function of position within the 2048-token window, broken into:
- Fine-grained (per-position, 0-2048)
- Position ranges (0-64, 64-128, ..., 1984-2048)
- Context distance analysis: for each token, how far back does the model actually look?

### Why
Our earlier analysis showed position degradation — tokens in early positions (less context) have higher loss. But we also saw that seq_len=2048 improved context_benefit from 0.379 to 0.436. The question is: **how much BPP improvement is left from even better position utilization?**

If the loss at position 0-64 averages 2.37 nats and position 1984-2048 averages 1.93 nats, the gap is 0.44 nats. A model with no position degradation would save those 0.44 nats on the early tokens — but those are only 3% of all tokens. The actual BPP impact depends on how many tokens are in each position bucket.

### What we learn
- How many bits are lost to position degradation
- Whether longer sequence length (4096?) would help
- Whether position-specific techniques (ALiBi, different RoPE) would help

---

## Step 4: Bigram Analysis

### What we do
For every bigram (prev_token → current_token) in the validation set:
- Compute average loss for that bigram
- Count frequency of occurrence
- Compute total bits spent on that bigram pattern (avg_loss × count)

Sort bigrams by total bits spent. The top bigrams by total cost are where the model is spending most of its budget.

### Why
SmearGate provides bigram context at the embedding level. But our Exp 16 analysis showed BigramHash contributes zero. This analysis tells us whether the model has fully exploited bigram information or whether there's still a bigram-related opportunity.

More importantly, it identifies the **specific transition patterns** that cost the most bits. If `,` → `▁the` costs 2.1 nats and appears 500K times, that's 1.05M nats — a significant chunk of the total. Improving prediction on just this one bigram would recover measurable BPP.

### What we learn
- The top-50 most expensive bigrams by total bits
- Whether SmearGate has fully captured bigram information
- Whether a larger vocabulary (merging frequent bigrams into single tokens) would help

---

## Step 5: Entropy Analysis

### What we do
For each token, record the model's output entropy (uncertainty). Plot loss vs entropy to create four quadrants:

| | Low entropy (confident) | High entropy (uncertain) |
|---|---|---|
| **Low loss (correct)** | Confident-right (ideal) | Uncertain-right (lucky) |
| **High loss (wrong)** | Confident-wrong (worst) | Uncertain-wrong (honest) |

### Why
**Confident-wrong tokens are the highest-leverage targets.** These are cases where the model thinks it knows the answer but is wrong — often caused by overfitting to common patterns or failing to use long-range context. Reducing confident-wrong predictions requires the model to be better calibrated, which training techniques like label smoothing, temperature scaling, or test-time training address.

**Uncertain-wrong tokens are opportunity tokens.** The model knows it doesn't know — if we can provide better context (longer sequences, better attention), these tokens could become uncertain-right or even confident-right.

### What we learn
- What fraction of bits go to confident-wrong (calibration problem) vs uncertain-wrong (capacity problem)
- Whether the model would benefit from calibration techniques
- Whether TTT (test-time training) could help by reducing uncertainty on document-specific patterns

---

## Step 6: Document-Level Analysis

### What we do
The FineWeb validation set contains multiple documents. For each document:
- Compute average loss for the first 100 tokens vs the rest
- Track how loss decreases as the model reads more of the document
- Identify documents with unusually high or low loss

### Why
The model sees each document from the beginning. Early tokens in a document have high loss because the model hasn't learned the document's topic, style, or vocabulary yet. As it reads more, it adapts (through its context window) and gets better.

This tells us how much BPP is lost to "cold start" on each document. If the average first-100-tokens loss is 0.5 nats higher than the rest, and documents are ~500 tokens on average, that's 20% of tokens paying a 0.5 nat penalty = 0.1 nats average = significant BPP impact.

**Test-time training (TTT)** directly addresses this by adapting the model to each document before scoring. If cold-start tokens account for X BPP, TTT could recover much of that.

### What we learn
- How much BPP is lost to document cold-start
- The potential ceiling for TTT improvement
- Whether document-level patterns (technical vs narrative, long vs short) affect loss

---

## Step 7: Quantization Impact Analysis

### What we do
Compare per-token losses between:
- fp32 model (raw checkpoint)
- int8+zlib roundtrip model
- int6+zstd roundtrip model (simulated)
- int5-MLP + int6-attn + zstd roundtrip (the submission format)

For each quantization level, identify which tokens are most affected — which token categories see the biggest loss increase from quantization.

### Why
Our current int8+zlib adds ~0.004 BPP. Moving to int5+int6 for submission will add more (~0.015-0.020). But this penalty isn't uniform — some tokens are much more sensitive to quantization than others. If we know WHICH tokens suffer most, we can:
- Keep those specific weight matrices in higher precision (selective fp16)
- Apply QAT training to make those weights quantization-robust
- Adjust the int5/int6 boundary to protect the most sensitive layers

### What we learn
- The exact BPP cost of each quantization strategy
- Which tokens/layers are most quantization-sensitive
- Whether QAT could reduce the penalty

---

## Step 8: Scaling Law Projection

### What we do
From all 18 experiments, extract (params, tokens_seen, val_loss) triples. Fit the Chinchilla scaling law:
```
L(N, D) = A/N^α + B/D^β + L_∞
```

where N = parameters, D = tokens seen, L_∞ = irreducible loss (the floor).

### Why
The scaling law tells us:
- **L_∞**: The absolute minimum loss achievable — no model of any size can beat this on this data
- **How much more capacity (N) we need** to reach our target
- **How much more training (D) we need** to reach our target
- **Whether we're compute-optimal** — are we spending params and training in the right ratio?

If L_∞ = 1.5 nats and we're at 2.0 nats, there's 0.5 nats to recover. If L_∞ = 1.9, there's only 0.1 nats available — and getting 0.1 BPP improvement might be impossible without fundamental changes.

### What we learn
- The irreducible loss for this validation set
- Whether our model is parameter-limited or data-limited
- The predicted loss at 50M params, 100M params (if we could fit them)

---

## Step 9: Competition Technique Gap Analysis

### What we do
For each technique used by the top-5 competition submissions but NOT in our model:
- Estimate the BPP contribution from competition ablation data
- Assess implementation difficulty
- Assess compatibility with our stack

### Techniques to analyze
| Technique | Used by | Est. BPP | In our model? |
|-----------|---------|----------|---------------|
| FA3 (Flash Attention 3) | #198, #173 | ~0.005 (more steps) | No |
| QAT with STE | #86, #150, #117 | ~0.010 (reduces quant penalty) | No |
| Late QAT (last 4% of training) | #315 | ~0.005-0.010 | No |
| Int5-MLP quantization | #180 | enables more params | No |
| Test-time training (TTT) | #77, #152 | 0.001-0.033 | No |
| Sliding window stride=64 | Nearly all | ~0.02 | Not in submission yet |
| NorMuon optimizer | #122, #89 | ~0.002 | No |
| RoPE base 50K | #206 | ~0.001-0.003 | No |

### Why
This is the "known techniques" budget. These are proven improvements that we simply haven't implemented yet. The sum of these represents the "free" improvement available without any novel research.

---

## Expected Output

A single comprehensive report with:

1. **Bits budget pie chart**: Where do the 1.1826 BPP go?
   - Floor (unpredictable tokens): X BPP
   - Position degradation: Y BPP
   - Document cold-start: Z BPP
   - Quantization penalty: W BPP
   - Bigram transition difficulty: V BPP
   - Remaining (improvable): R BPP

2. **Improvement roadmap**: Ordered list of techniques with:
   - Expected BPP contribution
   - Implementation cost (time, money)
   - Confidence level
   - Dependencies (what must come first)

3. **Feasibility assessment**: Can we actually reach 1.08? Or is the floor higher than that?

---

## Execution Plan

### Can run locally (no pod, free)
- Step 8: Scaling law (just math from experiment logs)
- Step 9: Competition gap (research from PR data)
- Parts of Step 2: Token classification logic

### Needs Exp 17 checkpoint + CPU (local machine, slow but free)
- Step 1: Per-token loss map (slow on CPU, ~30-60 min for 62M tokens)
- Step 4: Bigram analysis
- Step 5: Entropy analysis

### Needs GPU pod (~$2.69/hr for 1xH100, ~30 min = $1.35)
- Step 1: Per-token loss map (fast on GPU, ~5 min)
- Step 3: Positional analysis
- Step 6: Document-level analysis
- Step 7: Quantization impact analysis

### Recommended order
1. Scaling law (free, immediate insight)
2. Per-token loss map on GPU (most important, enables all other analyses)
3. Token classification + bigram + entropy + position (all derive from Step 1 data)
4. Document analysis (requires knowing document boundaries in val set)
5. Quantization analysis (requires int5/int6 implementation)
6. Competition gap (research, informs final roadmap)
7. Synthesize into final report
