# XSA Impact Report: What We Learned

## What We Did

We compared two models token-by-token on 100 validation documents (79,579 tokens):
- **Submission model**: XSA on last 4 layers (layers 7-10)
- **Exp26 model**: XSA on all 11 layers

Both models are the same architecture, same training setup — the only difference is how many layers use XSA. We looked at every single token prediction and measured which model assigned higher probability to the correct next token.

## The Surprising Result

**XSA-all is only better on 49% of tokens.** It's essentially a coin flip at the token level. Yet it achieves a measurably better BPB (1.1877 vs 1.1914 pre-TTT). How?

The answer: **when XSA-all wins, it wins bigger than when it loses.** The average improvement per token is +0.0013 nats. That's tiny per token, but across 46 million scored tokens in the full eval, it adds up to the 0.0037 BPB improvement we measured.

This tells us something important: XSA-all isn't a dramatic change for most tokens. It's a subtle shift in the probability distribution that slightly favors the correct answer in aggregate.

## Where XSA-all Helps

### 1. Tokens the model has never seen in the document (+0.0073 avg delta)

This was the biggest surprise. The largest improvement bucket is "token never seen before in this doc" — 25,451 tokens with the best average delta. This means XSA-all doesn't just help with recognizing repeated words. It helps the model make better **first-time predictions** of words based on the document's topic and context.

Why this matters: XSA forces every layer to build representations from OTHER tokens' information, not its own. This means the model builds richer contextual features throughout all 11 layers. When predicting a new word, it has better "understanding" of the surrounding topic because every layer contributed cross-token context.

### 2. Medium entropy tokens (+0.0087 avg delta)

When the model is moderately uncertain (entropy 2.0-4.0), XSA-all helps the most. These are tokens where the model has maybe 5-15 reasonable candidates. XSA-all shifts the probability distribution slightly toward the correct one.

Low entropy tokens (model is confident) show almost no difference — the model already knows the answer regardless of XSA. High entropy tokens (model is very uncertain) show moderate improvement but less than medium entropy.

### 3. Top individual improvements are subword completions

Looking at the top improvements (+6 to +12 nats), they're mostly **subword tokens** — completing words like "Crutch**low**", "Griffin **I**II", "Winf**ost**er", "g**aw**ky". These are cases where the model needs to recognize a rare name or word from context and complete the next piece. XSA-all's richer cross-token features help it "commit" to the right word once the first subword token appears.

## Where XSA-all Hurts

### 1. Mid-range repetitions, 50-200 tokens back (-0.006 avg delta)

This was unexpected. For tokens that appeared 50-200 tokens back, XSA-last4 is actually better. A possible explanation: XSA-last4 lets the early layers (0-6) use self-value bias, which acts like a "copy mechanism" — the token can reference its own embedding to help recognize a repeat. XSA-all removes this self-copy ability from every layer.

### 2. Low entropy tokens (-0.005 avg delta)

When the model is already confident, XSA-all slightly hurts. This makes sense — removing self-value information from attention removes a useful signal when the answer is obvious. The model doesn't need cross-token reasoning for simple predictions like "the" after a period.

### 3. Top individual regressions are formatting tokens

The worst regressions are tokens like "||" (table separators), apostrophes in predictable contexts, and structural tokens. These are patterns where self-attention (copying what came before) is the right strategy, and XSA's forced cross-token attention gets in the way.

## Key Insight

XSA-all is not a pure win. It's a **tradeoff**: better cross-token reasoning at the cost of worse self-referencing. The net effect is positive (+0.0013 nats/token on average) because novel predictions and moderate-uncertainty situations are more common and more impactful than the cases where self-copying helps.

The improvement is genuinely diffuse — it's not concentrated in any one pattern or position. It's a small improvement spread across many tokens, which is characteristic of a fundamental architectural change rather than a targeted optimization.

## What This Suggests for Next Experiments

### High-confidence ideas (directly supported by the data)

**1. Selective XSA: Use XSA on some layers, not all**

The data shows XSA-all hurts on confident predictions and mid-range repetitions. What if we use XSA on layers 0-4 (early, for building cross-token features) and layers 8-10 (late, for final reasoning), but NOT on layers 5-7 (middle, where copy/repeat patterns might be most useful)?

This hybrid approach could get the benefit of cross-token feature building in early layers without destroying the copy mechanism in middle layers. Quick experiment: train 3 variants with different XSA patterns (300s smoke test each).

**2. Scaled XSA: Partial self-value removal instead of full removal**

Instead of fully projecting out the self-value component, remove only a fraction:
```python
# Current: full removal
y = y - (dot_yv / dot_vv) * v

# New: partial removal with learnable scale
y = y - alpha * (dot_yv / dot_vv) * v  # alpha = 0.5 or learnable
```
This lets the model keep some self-value information while still encouraging cross-token attention. The optimal alpha might differ by layer.

**3. Better TTT: SGD with momentum instead of Adam**

Our TTT improvement was -0.0374 BPB. The top submission gets -0.0025 BPB from TTT but starts from a much lower pre-TTT baseline (1.1218). Their TTT recipe is fundamentally different:
- SGD(lr=0.002, momentum=0.9) instead of Adam(lr=0.05)
- ALL weights unfrozen, not just LoRA on Q/V
- 3 epochs per chunk instead of 1
- Cosine LR decay with grad clipping

Testing SGD TTT with our exp26 checkpoint is a no-retrain experiment — just change the eval-time TTT procedure. If it works, it could give another -0.003 to -0.01 BPB.

### Speculative ideas (less directly supported)

**4. Attention to attention patterns**

The fact that XSA-all helps most on "never seen" tokens suggests the model's contextual understanding is the bottleneck, not its memory. We could investigate this by extracting attention patterns from both models and comparing where they attend differently. Specifically:
- For the top-improved tokens, which earlier positions does XSA-all attend to that XSA-last4 doesn't?
- Do XSA-all's early layers (0-3) attend to semantically relevant tokens, or just nearby tokens?
- Is there a pattern in attention entropy — does XSA-all produce sharper or more diffuse attention?

This is a diagnostic experiment (no training), and would tell us whether the improvement comes from better attention routing or from the forced feature mixing. If attention patterns look similar but predictions differ, the improvement is in the learned features, not the attention itself. If attention patterns are very different, we could potentially design better attention mechanisms.

**5. Position-aware TTT**

The data shows XSA improvement is roughly flat across document positions. But TTT trains on the full document equally. What if TTT used higher learning rate for later chunks (where the model has more context to learn from) and lower for early chunks?

**6. Document-adaptive XSA**

Some documents show strong positive delta (+0.11 nats/token), others strong negative (-0.11). If we could predict which documents benefit from XSA-all vs XSA-last4 and switch at eval time, we'd get the best of both. This is complex but the ceiling is high.

---

Next experiments are documented in `dashboard/exp28_32_plan.md`.
