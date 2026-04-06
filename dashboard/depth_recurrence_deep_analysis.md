# Depth Recurrence: Deep Analysis

**Model**: 7 unique blocks, dim=624, layer schedule `[0,1,2,3,4,3,4,3,4,5,6,5,6]`
**Result**: val_bpb = 1.1172 (with GPTQ + SGD TTT)

---

## 1. The Simple Explanation

### What is depth recurrence?

A normal transformer has N unique layers stacked on top of each other. Each layer has its own weights. A 13-layer transformer needs 13 sets of weights.

Depth recurrence means **reusing some layers multiple times**. Instead of 13 unique layers, we have 7 unique layers, but we pass the data through some of them more than once:

```
Normal 13-layer:    L0 → L1 → L2 → L3 → L4 → L5 → L6 → L7 → L8 → L9 → L10 → L11 → L12
                    (13 unique sets of weights = 13 × 3.8M = 49.4M params)

Our model:          B0 → B1 → B2 → B3 → B4 → B3 → B4 → B3 → B4 → B5 → B6 → B5 → B6
                    (7 unique sets of weights = 7 × 3.7M = 25.6M params)
```

The data passes through blocks 3 and 4 three times each, and blocks 5 and 6 twice each. The model gets 13 layers of processing depth but only pays for 7 layers of parameters.

### Why would this work?

Think of it like re-reading a paragraph. The first time you read it, you get the gist. The second time, you catch details you missed. The third time, you notice connections between sentences. Each pass through the same "processing unit" (your reading comprehension) extracts more from the same input.

Similarly, blocks 3 and 4 see the same data three times. Each pass refines the representation further. The block doesn't "know" which pass it's on — it applies the same function each time — but the *input* is different because previous passes have already transformed it.

### Why not just use 7 unique layers?

Because depth matters — and the model is catastrophically dependent on its recurrence passes.

We ran a full ablation study (see [exp47_recurrence_ablation.md](exp47_recurrence_ablation.md)) evaluating the trained exp40-D checkpoint with different layer schedules. The model was trained with `[0,1,2,3,4,3,4,3,4,5,6,5,6]` (13 effective layers, blocks 3-4 used 3x, blocks 5-6 used 2x). We call this **schedule A**. At inference time, we kept the same 7 trained blocks but varied how many times each is called:

| Schedule | Eff Layers | val_bpb | Delta vs trained schedule |
|----------|-----------|---------|--------------------------|
| **A: Full 3x `[0,1,2,3,4,3,4,3,4,5,6,5,6]` (as trained)** | **13** | **1.1493** | **baseline** |
| B: 2x recurrence | 11 | 1.6919 | +0.543 (broken) |
| C: No recurrence | 7 | 2.7739 | +1.625 (destroyed) |
| D: 3x middle only | 11 | 2.6173 | +1.468 (destroyed) |
| E: 2x end only | 9 | 1.8744 | +0.725 (broken) |
| F: 4x recurrence | 17 | 3.8153 | +2.666 (worst) |

Removing even a single recurrence pass (B) increases BPB by **+0.54** — the model doesn't degrade gracefully, it breaks. The weights are fundamentally trained to expect data flowing through the exact sequence in the exact order. Running just 7 layers (C) produces BPB of 2.77, worse than an untrained model.

Adding extra passes (F: 4x) is even worse than removing them — the downstream blocks have never seen representations from a 4th pass.

**Key finding from the ablation**: end recurrence (blocks 5-6) matters 2x more than middle recurrence (blocks 3-4). Removing the end pass (D) is +1.47 worse; removing the middle pass (E) is +0.73 worse. The decoder blocks are more critical because they consume U-Net skip connections and sit closest to the output.

**TTT cannot rescue broken schedules**: SGD all-weights TTT recovers only 6% of the gap for B (2x) and 0.2% for C (no recurrence). The problem isn't wrong weights — it's a wrong computational graph.

### Why not just use 13 unique layers?

Because parameters cost storage. Our model must fit in 16MB after quantization. 13 unique blocks at dim=624 would need ~48M parameters — that's 192MB in float32, far too large even with aggressive quantization. By reusing blocks, we get deep processing while staying within budget.

---

## 2. The Architecture in Detail

### Layer Schedule

```
Position:   0   1   2   3   4   5   6   7   8   9  10  11  12
Block:      B0  B1  B2  B3  B4  B3  B4  B3  B4  B5  B6  B5  B6
            ├─ unique ─┤  ├──── 3x recurrence ────┤  ├─ 2x rec ─┤
Role:       ├── encoder (positions 0-5) ──┤├──── decoder (positions 6-12) ────┤
```

The model uses a **U-Net structure**: the first half is the "encoder" and the second half is the "decoder." Skip connections link encoder layers to decoder layers (position 0 → 12, position 1 → 11, etc.), letting the decoder access early representations directly.

### What each block does

Every block applies the same two operations:

```
1. Blend:     x = resid_mix[0] * x + resid_mix[1] * x0     (mix running state with original embedding)
2. Attention: x = x + attn_scale * Attention(RMSNorm(x))    (attend to other positions)
3. MLP:       x = x + mlp_scale * MLP(RMSNorm(x))           (per-token nonlinear transform)
```

Key detail: `x0` is the **original token embedding** (after SmearGate). It's passed to every block as a residual anchor. This means block 3 on its third pass still has access to the raw token information, not just the heavily-processed running state.

### Per-block parameter breakdown

Each block (dim=624, 8 heads, 4 KV heads, SwiGLU 3x) contains:

| Component | Parameters | Notes |
|-----------|-----------|-------|
| `attn.c_q` (query projection) | 624 × 624 = 389,376 | Full rank |
| `attn.c_k` (key projection) | 624 × 312 = 194,688 | GQA: 4 KV heads |
| `attn.c_v` (value projection) | 624 × 312 = 194,688 | GQA: 4 KV heads |
| `attn.c_proj` (output projection) | 624 × 624 = 389,376 | Projects back |
| `mlp.gate` + `mlp.up` (SwiGLU) | 2 × 624 × 1872 = 2,335,296 | 3x expansion |
| `mlp.down` | 1872 × 624 = 1,167,744 | Projects back |
| Norms, scales, resid_mix | ~2,500 | Small |
| **Total per block** | **~3,673,668** | |

With 7 unique blocks: **25.7M params** (+ embeddings, skip weights, etc. ≈ 26.5M total).

### How recurrence changes the gradient flow

In a normal transformer, block 3 receives gradient from one path: the loss backpropagates through layers 12 → 11 → ... → 4 → **3** → 2 → 1 → 0.

With recurrence, block 3 appears at positions 3, 5, and 7. It receives gradients from **three different paths**:

```
Loss
 ↓
Position 12 (B6) → 11 (B5) → 10 (B6) → 9 (B5) → 8 (B4) → 7 (B3)  ← gradient path 1
                                                      ↓
                                                    6 (B4) → 5 (B3)  ← gradient path 2
                                                               ↓
                                                             4 (B4) → 3 (B3)  ← gradient path 3
```

Block 3's weights receive the **sum of three gradients**, one from each position where it appears. This is effectively 3x the gradient signal per training step — like a form of implicit data augmentation.

---

## 3. The Evidence: What Experiments Tell Us

### Experiment 36: First depth recurrence (2x)

Config: 9 unique blocks, dim=560, schedule `[0,1,2,3,4,5,3,4,5,6,7,8]` → 12 effective layers.

| Observation | Data | Interpretation |
|-------------|------|----------------|
| Advantage grew during active training | -0.006 at step 1000 → -0.008 at step 5000 | More gradient signal = faster learning |
| Advantage froze during warmdown | -0.006 at step 6000-8000 | When lr decays, extra signal doesn't help |
| TTT delta was larger (-0.050 vs -0.045) | Recurrent model + TTT vs non-recurrent + TTT | Richer features for TTT to exploit |

**Key finding**: Depth recurrence helps the model **learn faster per step**, not converge to a fundamentally different solution. Given enough training time, a non-recurrent model might catch up (but we'd exceed our time budget).

### Experiment 40: Architecture exploration

Tested 4 variants at 500 steps (smoke test):

| Variant | Change | Result | Why |
|---------|--------|--------|-----|
| B: 3x recurrence, same width | 9 blocks, dim=560, 15 eff layers | **+0.017 worse** | More depth without wider dim = wasted |
| C: wider, fewer layers | 7 blocks, dim=624, 9 eff layers | **+0.064 worse** | Wider without depth = underprocessed |
| **D: wider + 3x recurrence** | **7 blocks, dim=624, 13 eff layers** | **-0.009 better** | Width + depth complement each other |
| E: 4x MLP | 9 blocks, dim=528, 12 eff layers | Over 16MB | Too many params |

The critical result: **B and C both fail individually but D (their combination) succeeds.**

This tells us depth recurrence is not just "free layers." It's specifically valuable when the model has wider representations (more information per token) that benefit from multiple processing passes. Narrow representations (dim=560) don't have enough information for a third pass to extract — the block just rehashes the same features.

### Full training: Matched step comparison

At step 8366 (where exp36 finished):

| Model | Raw val_bpb at step 8366 |
|-------|-------------------------|
| Exp36 (9 blocks, dim=560, 2x) | 1.1594 |
| Exp40-D (7 blocks, dim=624, 3x) | ~1.158 (interpolated) |

**Only -0.001 difference at matched steps.** The remaining -0.012 of exp40-D's final -0.013 improvement came from training longer (240 vs 120 min, 14777 vs 8366 steps).

This confirms: the architecture advantage is **per-step efficiency**, which compounds over more training steps. The recurrent model isn't learning a different function — it's learning the same function faster.

---

## 4. Deep Dive: What Each Recurrence Pass Does

### The `resid_mix` mechanism

Each block blends its input `x` (running hidden state) with `x0` (original embedding):

```python
x = resid_mix[0] * x + resid_mix[1] * x0
```

On block 3's first pass (position 3), `x` is shallow — it's only been through blocks 0, 1, 2. The blend with `x0` adds back raw token information.

On block 3's third pass (position 7), `x` is deep — it's been through B0, B1, B2, B3, B4, B3, B4. The blend with `x0` still anchors the representation to the original tokens, preventing the signal from drifting too far.

**The problem**: `resid_mix` is a learned parameter of block 3, fixed across all three passes. It can't adapt its blending ratio based on which pass it's on. On pass 1, the optimal blend might be 70% `x` + 30% `x0` (raw tokens still important). On pass 3, it might be 95% `x` + 5% `x0` (deep features dominate). But the block uses the same ratio every time.

This is a **real information bottleneck** — the block can't distinguish which pass it's executing.

### What the block "sees" each pass — and what it does to the loss

We can go beyond theory and measure this directly. **Exp47b probed the loss at every position in the schedule** — at each point, project the hidden state to logits and measure how good the predictions are. This reveals how each layer contributes to prediction quality:

```
Position  Block        Loss     Delta     Role
────────────────────────────────────────────────────────────
0         B0           7.858    —         Almost random predictions
1         B1           6.494    -1.364    ▼▼▼▼ Biggest single-layer drop
2         B2           6.366    -0.128    ▼
3         B3-pass1     6.071    -0.295    ▼▼ First pass helps predictions
4         B4-pass1     5.876    -0.195    ▼▼
5         B3-pass2     5.936    +0.060    ▲ LOSS GOES UP
6         B4-pass2     6.003    +0.066    ▲ LOSS GOES UP
7         B3-pass3     6.242    +0.240    ▲▲ LOSS GOES UP MORE
8         B4-pass3     5.672    -0.570    ▼▼▼▼ B4 pass 3 recovers everything
9         B5-pass1     5.627    -0.046    ▼
10        B6-pass1     4.361    -1.266    ▼▼▼▼ Second biggest drop
11        B5-pass2     4.381    +0.020    ~ flat
12        B6-pass2     1.926    -2.455    ▼▼▼▼▼ LARGEST — 41% of total reduction
```

**The critical finding: middle recurrence passes (positions 5-7) make predictions WORSE, not better.** If you projected to logits after block 3's second pass, the loss is higher than after its first pass. After block 3's third pass, it's higher still.

This doesn't mean these passes are useless. It means they're **transforming representations for downstream layers, not improving predictions directly.** Like an assembly line: the middle station might make the product look worse (disassembling components) because it's rearranging parts for the final assembly.

The proof: block 4 pass 3 (position 8) immediately follows with **the biggest improvement in the recurrence section** (-0.570), and the final layer (B6 pass 2) does **41% of all loss reduction** (-2.455). The middle passes build the features that make these final layers effective.

### Block 3 and Block 4 play complementary roles

| Pass | Block 3 (delta) | Block 4 (delta) | Pattern |
|------|----------------|----------------|---------|
| 1 | -0.295 (reduces loss) | -0.195 (reduces loss) | Both help on first pass |
| 2 | +0.060 (reorganizes) | +0.066 (reorganizes) | Both sacrifice predictions for features |
| 3 | +0.240 (destabilizes) | **-0.570 (reconstructs)** | B3 tears down, B4 builds back up |

Block 3 appears to destabilize representations — making them less directly useful for prediction but richer in the features that block 4 and the decoder need. Block 4 then compresses these features into a form the decoder can use. They work as a **deconstruct/reconstruct pair**.

### Where predictions actually happen

The model builds confidence in two sharp jumps, not gradually:

```
Position  Block        Top-1 Prob   What's happening
────────────────────────────────────────────────────
0-8       B0 to B4p3   0.3-0.7%    Near-uniform — no opinion yet
9         B5-pass1     0.4%        Still no opinion
10        B6-pass1     5.8%        ▼ First real confidence jump
11        B5-pass2     6.1%        Holds
12        B6-pass2     53.0%       ▼▼▼ Commits to prediction
```

The entire recurrence section (positions 3-8) doesn't change confidence at all. Block 6 is where features get converted into predictions — first tentatively (5.8% top-1), then decisively (53% top-1).

### Which tokens benefit from recurrence?

We split tokens by difficulty (based on final loss) and tracked how they evolve:

| Difficulty | After B3-pass1 | After B3-pass3 | Change | Interpretation |
|-----------|---------------|---------------|--------|----------------|
| Easy (0-25%) | 5.46 | 6.23 | **+0.76 worse** | Recurrence hurts easy tokens! |
| Medium (25-50%) | 5.86 | 5.69 | -0.17 better | Medium tokens benefit most |
| Hard (50-75%) | 6.07 | 5.89 | -0.18 better | Hard tokens also benefit |
| Hardest (90-100%) | 7.74 | 8.47 | **+0.73 worse** | Hardest tokens also hurt |

The model **sacrifices easy and very-hard tokens during recurrence** to improve medium and hard tokens. This makes sense: easy tokens are already predictable (final loss 0.01), so the model doesn't need recurrence for them. Very hard tokens are unpredictable regardless (final loss 6.37), so the model gives up on them. Recurrence focuses its effort on the tokens where additional processing actually helps — the middle difficulty range.

### The iterated function perspective — and what the data actually shows

Depth recurrence looks like a **fixed-point iteration**: apply the same function repeatedly until the output stabilizes. If that were true, we'd expect:
- Cosine similarity increasing with each pass (convergence)
- Representation norms stabilizing (reaching a fixed point)
- Later passes making smaller changes (diminishing returns)

**Exp47 measured this directly.** We hooked block 3's output at each of its 3 appearances and computed activation statistics across 64 sequences (131K tokens):

| Comparison | Cosine Similarity | Interpretation |
|------------|------------------|----------------|
| Pass 1 → Pass 2 | 0.787 | Substantial transformation |
| Pass 2 → Pass 3 | 0.859 | Smaller but still significant |
| Pass 1 → Pass 3 | **0.595** | **Least similar** (not converging!) |

| Pass | Mean Norm | Relative Change from Prior |
|------|-----------|---------------------------|
| 1 (position 3) | 188,481 | — |
| 2 (position 5) | 221,943 | 0.741 (74% of input norm) |
| 3 (position 7) | 245,646 | 0.579 (58% of input norm) |

**The model is NOT doing fixed-point iteration.** Three pieces of evidence:

1. **Pass 1 and Pass 3 are the LEAST similar** (cosine 0.595). A contractive mapping would make them the most similar (closest to the fixed point). Instead, each pass moves the representation further from where it started.

2. **Norms grow monotonically** (188K → 222K → 246K). A fixed-point attractor would stabilize the norm. Instead, the block pumps energy into the representation — amplifying features that downstream blocks need.

3. **Changes diminish but don't vanish.** Pass 2→3 relative change (0.579) is still large. If we were near a fixed point, this would be close to zero.

**What it's actually doing**: Each pass applies the same weights to a *different input distribution*. Pass 1 sees shallow features (from blocks 0-2). Pass 2 sees medium-depth features (after one round of blocks 3-4). Pass 3 sees deep features (after two rounds). The block has learned weights that are useful across all three distributions — a compromise that creates a directional trajectory rather than convergence.

**This is closer to an iterative refinement process than fixed-point convergence** — like running multiple passes of a denoising algorithm, where each pass removes different types of noise rather than converging to a clean image.

---

## 5. Interaction with Other Components

### Depth recurrence × GPTQ

Two effects compound to make quantization especially damaging in recurrent models:

**1. Shared block errors appear multiple times.** Block 3's quantization error shows up at 3 positions in the forward pass. And because norms grow across passes (188K → 222K → 246K, from exp47b), the same rounding error gets amplified on each subsequent pass.

**2. Block 6 does 63% of loss reduction — and it's quantized like every other block.** The loss probing data (exp47b) shows B6 is responsible for 41% (pass 2) + 22% (pass 1) = 63% of total loss reduction. Yet we quantize B6's attention weights at int6, the same as every other block. Quantization errors in B6 directly corrupt the final prediction more than errors in any other block.

This explains why GPTQ was so effective (-0.009 BPB): it uses Hessian information to minimize output-level error, which naturally prioritizes the weights that matter most for predictions — and those weights are concentrated in B6 and the recurrent blocks.

**Actionable**: Per-block bit allocation could help. Giving B6 int8 (or even keeping it in float16) while using int5 for middle blocks would preferentially protect the most critical layer. This trades ~100KB of artifact size for potentially significant BPB improvement.

### Depth recurrence × SGD TTT

During TTT, SGD adapts ALL 26.5M weights to a specific document. The shared blocks 3-4 get adapted once but affect 3 positions in the forward pass. A small weight change in block 3 has **3x the leverage** of a change in block 0.

This may explain why TTT is slightly more effective on recurrent models (-0.050 vs -0.045): the shared blocks amplify the effect of weight adaptation.

### Depth recurrence × U-Net skip connections

Our model has skip connections linking encoder layers to decoder layers:

```
Encoder:   B0 → B1 → B2 → B3 → B4 → B3      (saves skips at each step)
                ↓     ↓     ↓     ↓    ↓   ↓
Decoder:   B6 ← B5 ← B6 ← B5 ← B4 ← B3 ← B4   (consumes skips in reverse)
```

The skip connections bridge the "same" block at different depths. Position 3 (block B3, shallow) connects to position 9 (block B5, deep) via skip. Position 5 (block B3 again, medium depth) connects to position 7 (block B4, medium depth).

This means block B3's output at different recurrence passes gets bridged to different decoder positions. The encoder's "shallow B3" output feeds directly to a deep decoder layer, providing a gradient shortcut that helps train the shared block.

---

## 6. Open Questions and Diagnostic Experiments

### ANSWERED: Q1 — How do recurrence passes contribute to loss reduction?

**Answer (from exp47b loss probing)**: Middle recurrence passes (B3/B4 passes 2-3) don't reduce probed loss — they **increase** it. They build abstract features that only become useful when the decoder (B6) processes them. The actual loss reduction is concentrated:
- B1 (early): -1.36 (25%)
- B6 pass 1: -1.27 (22%)
- B6 pass 2: -2.46 (41%)
- B4 pass 3: -0.57 (10%, the "exit" from recurrence)

Block 3 destabilizes representations while block 4 reconstructs them — they work as a pair.

### ANSWERED: Q2 — Are activations converging across passes?

**Answer (from exp47)**: No. Block 3 does NOT converge to a fixed point. Cosine similarity between pass 1 and pass 3 is only 0.595 (the lowest pair). Norms grow monotonically (188K → 222K → 246K). Each pass moves the representation in a consistent direction with diminishing step size, but without converging back toward the start. This is iterative refinement, not fixed-point convergence.

### ANSWERED: Q4 — Is 3x optimal?

**Answer (from exp47)**: 4x is catastrophically worse (+2.67 BPB) when applied to a model trained with 3x. This cannot tell us whether 4x would be better if trained from scratch, but it confirms the recurrence count is baked into the weights and cannot be changed at inference time.

### ANSWERED: Q6 — Which tokens benefit from recurrence?

**Answer (from exp47b)**: Medium-difficulty tokens benefit most. Easy tokens (final loss ~0.01) get worse through recurrence — the model sacrifices their intermediate representations because they'll be correctly predicted regardless. The hardest tokens (final loss ~6.4) also get worse — the model gives up on them. Recurrence focuses effort on the middle difficulty range where extra processing actually helps.

### OPEN: Q3 — Does `resid_mix` create a bottleneck?

**Experiment**: Replace the single `resid_mix` per block with pass-dependent mixing. Give block 3 three different `resid_mix` values — one for each of its three passes. This adds only ~3,744 parameters (3 passes × 2 × 624 dim) but lets each pass blend `x` and `x0` differently.

**What we'd learn**: Whether the fixed blending ratio is a real bottleneck. If pass-dependent mixing improves the loss, it confirms the block needs to distinguish which pass it's on.

**Connection**: This is the minimal version of exp44 (Relaxed Recurrence LoRA). The exp47 activation data (cosine sim 0.60-0.86 across passes) strongly suggests each pass is doing different work. Giving each pass its own adapter should help.

### OPEN: Q5 — Would different blocks benefit from sharing?

**What we'd learn**: Whether we've hit diminishing returns on recurrence, or whether 4x would give further improvement.

### Question 5: Would different blocks benefit from sharing?

We currently share the "middle" blocks. What about sharing the early or late blocks instead?

| Schedule | Description |
|----------|-------------|
| `[0,1,0,1,0,1,2,3,4,5,6]` | 3x early blocks |
| `[0,1,2,3,4,5,6,5,6,5,6]` | 3x late blocks |
| `[0,1,2,3,4,3,4,3,4,5,6,5,6]` | Current (3x middle) |

**Intuition**: Early blocks handle low-level features (token identity, position). Late blocks handle high-level features (semantics, prediction). Middle blocks are the "bottleneck" where low-level features get transformed into high-level ones. Recurrence in the middle gives more iterations on the hardest transformation.

---

## 7. Connection to Exp44 (Relaxed Recurrence LoRA)

The biggest limitation of depth recurrence is that **the block can't distinguish which pass it's on**. Same weights, same `resid_mix`, same everything. The only signal is the input `x`, which changes between passes.

Exp44 proposes adding small per-pass LoRA adapters to the shared blocks. This gives each pass its own learned adjustment while keeping the bulk of parameters shared:

```
Pass 1: output = Block3(x) + LoRA_3_pass1(x)    (32 params × rank)
Pass 2: output = Block3(x) + LoRA_3_pass2(x)    (32 params × rank)
Pass 3: output = Block3(x) + LoRA_3_pass3(x)    (32 params × rank)
```

At rank 4, this adds only ~60K params total (across all shared blocks and passes) — well within our 16MB budget. But it lets:
- Pass 1 specialize in reducing loss directly (it's the only pass that does)
- Pass 2 specialize in feature deconstruction for the decoder
- Pass 3 specialize in producing features that B4 pass 3 can reconstruct from

The exp47b loss probing data makes the strongest case for this: passes 1, 2, and 3 of block 3 have **qualitatively different roles** (loss-reducing vs feature-building vs destabilizing). A single set of weights must compromise across all three. Per-pass LoRA resolves this compromise by letting each pass adapt the shared weights to its specific role.

---

## 8. Summary

| Aspect | Finding | Confidence |
|--------|---------|------------|
| Why it works | More processing depth within fixed param budget | High |
| Primary benefit | Faster learning per step (3x gradient signal) | High |
| Width + depth interaction | Both needed; neither alone helps | High (exp40 A-D) |
| Middle passes increase probed loss | Passes 2-3 build features for decoder, don't reduce loss directly | **High (exp47b measured)** |
| B6 does 63% of loss reduction | Final layer is where predictions happen (41% + 22%) | **High (exp47b measured)** |
| B3 destabilizes, B4 reconstructs | Complementary roles within recurrence pair | **High (exp47b measured)** |
| Token difficulty triage | Recurrence helps medium tokens, sacrifices easy and hardest | **High (exp47b measured)** |
| Confidence builds in 2 jumps | At B6-pass1 (5.8%) and B6-pass2 (53%), not during recurrence | **High (exp47b measured)** |
| Convergence behavior | NOT fixed-point — iterative directional refinement | **High (exp47 measured)** |
| Quantization priority | B6 should get higher precision than other blocks | **High (exp47b implies)** |
| `resid_mix` bottleneck | Can't distinguish passes (potential info loss) | Medium (untested) |
| Per-pass specialization need | Passes have qualitatively different roles (loss-reducing vs feature-building) | **High (exp47b measured)** |

**The core insight**: Depth recurrence creates a **pipeline with division of labor**. Early blocks reduce loss. Middle recurrence passes sacrifice prediction quality to build abstract features. The final decoder layer (B6) converts those features into predictions, doing 41% of all loss reduction in a single layer. This means:

1. **The recurrence section is a feature factory**, not a prediction improver. Per-pass adapters (exp44) should help because each pass has a distinct role.
2. **B6 is the most important block to protect** during quantization — errors there directly corrupt 63% of prediction capability.
3. **Medium-difficulty tokens are the recurrence beneficiaries** — the model strategically allocates recurrence effort to tokens where extra processing helps.

**Detailed diagnostic data**: [exp47_recurrence_ablation.md](exp47_recurrence_ablation.md)
