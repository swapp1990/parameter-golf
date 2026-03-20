# Parameter Golf — Experiment Framework

## Goal
Train the best language model that fits in 16MB, scored on val_bpb (lower = better).
Target: **1.22 bpb**. Current best: **1.2987 bpb** (Experiment 8a, 11603 steps, 1200s).

## Constraints
- Single RTX 5090 GPU (~24GB VRAM)
- 16MB budget = ~17M params in int8+zlib compression
- Vocab locked at 1024 (must match tokenizer)
- grad_accum_steps is hardcoded to 8 on single GPU
- Each experiment: up to 20 minutes wallclock (relaxed from 10 min)

## Process
```
1. Read experiments.json (learn from all past experiments)
2. Update this plan (experiments_plan.md) for next experiment only
3. Build/modify any tools needed
4. Run experiment
5. Run evaluation (deep_eval.py)
6. Append post-mortem to experiments.json
7. Go to 1
```

---

## Lessons Learned (Experiments 1–6)

### Exp 1 (94 steps, val_bpb=3.10)
- batch=524288 → 1286ms/step. Catastrophically undertrained.

### Exp 2 (1785 steps, val_bpb=1.46)
- batch=65536 → 100ms/step. Transformative speedup.
- Deep eval: position degradation past ~256, bathtub layer pattern, tail-heavy loss.

### Exp 3 (1925 steps, val_bpb=1.49) — DISPROVED
- seq_len=512 is WORSE than 1024 despite more steps and better layer utilization.
- Position degradation was from undertraining, not architectural limitation.
- **Lesson**: Deep eval metrics can mislead if used prescriptively. val_bpb is what matters.

### Exp 4 (5964 steps, val_bpb=1.354)
- 600s training. Confirmed more steps helps, logarithmic curve.
- Position degradation worsened to -1.13. Middle layers still dead.
- Uncertain-wrong tokens increased (32.2% → 35.0%).

### Exp 5a (FAILED: val_bpb=3.57)
- 6L/624d architecture change. Optimizer hyperparams don't transfer across architectures. Killed early.

### Exp 5 (8533 steps, val_bpb=1.38) — DISPROVED
- batch=32768. More steps but worse val_bpb than Exp 4. batch=65536 is optimal.

### Exp 6 (11909 steps, val_bpb=1.312) — BEST RESULT
- 1200s training. Only 0.09 from target.
- **Warmdown discovery**: Last 600 steps gave 2.6x per-step efficiency (0.000040 bpb/step vs 0.000015).
- Three learning phases: rapid (0–2k), slow grind (2k–11k), warmdown boost (11k–12k).
- Compression at 15.75 MB — tight but stable.
- Position degradation stable at -1.08 (architectural, not training-dependent).
- Loss distribution barely shifts with more training. Hard tokens remain ~53%.

### Exp 7 (11597 steps, val_bpb=1.3062) — NEW BEST at the time
- warmdown_iters=2000. Confirmed warmdown as highest-leverage parameter.
- **Layer 3 activated** (impact 0.1→0.58) — longer warmdown revived a dead middle layer.
- Position degradation worsened (-1.08→-1.20). Warmdown reinforces short-range bias.
- Confident-wrong tokens reduced (1.7%→1.4%). Better calibrated predictions.
- Register token was NOT implemented in this run (oversight).

### Exp 7 Register Token Tests (A/B/C at 180s and 600s)
- **Register at position 0 (causal)**: Useless — causal masking means position 0 only sees itself. It's a learned constant, not a document summary.
- **Register appended at end + per-layer gate (mode 1)**: Slight juncture penalty reduction (-0.063) but ~14% slower per step. Net negative at 600s.
- **Register appended at end + final gate (mode 2)**: Marginal juncture penalty reduction. Minimal overhead.
- **Context injection (mean-pool)**: No improvement. Scale parameters learned to stay near zero.
- **Bidirectional register mask**: Incompatible with torch.compile + SDPA backend. Not testable.
- **Conclusion**: Register token helps juncture tokens slightly but costs steps. Not worth it at this scale.

### Exp 8a (11603 steps, val_bpb=1.2987) — CURRENT BEST
- warmdown_iters=3000, no register token. Best result by 0.0075 over Exp 7.
- Layer 3 stayed active (0.38 impact). Layers 4-6 still dead (0.10-0.14).
- warmdown=3000 did NOT activate more dead layers — diminishing returns on warmdown for layer activation.
- Position degradation stable at -1.19.
- Very hard tokens down to 19.8% (from 20.0%).

### Exp 8b (11200 steps, val_bpb=1.3009) — Register token + warmdown=3000
- Register mode 2 + warmdown=3000. Worse than 8a by 0.0022 bpb.
- Register token reduced juncture penalty (+0.912 vs +0.944) but hurt overall BPB due to step overhead.
- **Register token is conclusively dead** at this model scale.

### Exp 8 Large Batch Accident
- Ran warmdown=3000 + register with batch=524288 (forgot TRAIN_BATCH_TOKENS=65536).
- Got val_bpb=1.3034 with only 1912 steps (entire run was in warmdown, i.e. pure cosine schedule).
- Insight: large batch + full cosine decay is competitive. Each step sees 8x more tokens = better gradients.

### Confirmed facts after 8 experiments
1. **9L/512d/8h/4kv, batch=65536, seq_len=1024** is optimal within tested configs
2. **More training always helps** but with severe diminishing returns (logarithmic)
3. **Warmdown is the highest-leverage training parameter** — warmdown=3000 > 2000 > 600
4. **Position degradation (~-1.20) is architectural** — stable across all experiments
5. **Layers 4-6 are dead weight** (~0.10-0.14 impact) — not fixable by warmdown alone
6. **Layer 3 activates with warmdown ≥ 2000** (0.1→0.58 impact)
7. **Hard tokens are word-initial at juncture points** — invariant across all approaches
8. **Register token doesn't help overall BPB** — marginal juncture benefit eaten by step overhead
9. **Compression is not the bottleneck** (15.75-15.83 MB, grows slowly)
10. **Architecture changes need LR sweeps** (Exp 5a failed without one)
11. **Batch=65536 is optimal for single GPU** — smaller batch worse (Exp 5), larger batch needs grad_accum
12. **Warmdown has diminishing returns for layer activation** — 3000 didn't activate layers 4-6

### Key discoveries about train_gpt.py
- `warmup_steps=20` is torch.compile warmup (not LR warmup), happens BEFORE timer
- `grad_accum_steps = 8` hardcoded on single GPU
- All hyperparameters configurable via environment variables
- Warmdown uses cosine LR decay to zero over final N steps

---

## Next Experiment: Experiment 9 — Layer Looping + Wider Model

### Rationale

We've exhausted training regime improvements (warmdown=3000 is optimal, register token doesn't help). The remaining 0.08 bpb gap to target requires an **architectural change**. The core problem: layers 4-6 are dead weight (~0.10-0.14 impact each), wasting ~5M parameters that could be spent on width.

**Layer looping** solves this by using fewer unique layer weights reused multiple times:
- Fewer unique layers → fewer stored parameters → room for wider model
- Every unique layer gets gradient signal from multiple forward uses → no dead layers
- Wider model → more attention capacity per head → better long-range patterns

### The problem with architecture changes

Exp 5a proved that changing architecture without re-tuning LR causes catastrophic failure (3.57 bpb). We must either:
- Do a quick LR sweep (3-4 short runs at different MATRIX_LR values)
- Or make a conservative change that stays close enough to current dims

### Param budget analysis

| Config | Unique Params | Est Compressed | Headroom | Effective Depth |
|--------|--------------|----------------|----------|-----------------|
| Current 9L/512d | 17.06M | 14.97MB | +0.27MB | 9 |
| 5L/672d loop to 9 | 16.51M | 14.48MB | +0.76MB | 9 |
| 6L/608d loop to 9 | 16.16M | 14.18MB | +1.06MB | 9 |
| 5L/640d loop to 9 | 15.01M | 13.17MB | +2.07MB | 9 |

### Recommended config: 6 unique layers / 608 dim, looped to 9 effective layers

Why this config:
- **608 dim** = 19% wider than 512. Each attention head gets 76 dims (vs 64). Meaningful capacity increase.
- **6 unique layers** looped to 9 effective = same depth as current model, no risk of underfitting
- **+1.06MB headroom** = safe margin for compression variance
- **608 is divisible by 8** (num_heads) → head_dim=76, works with GQA
- Close enough to 512 that MATRIX_LR=0.04 should transfer (but we'll do a quick sweep)

### Implementation: Layer Looping

```python
# New env vars:
# NUM_UNIQUE_LAYERS=6 (physical layers stored in memory)
# NUM_EFFECTIVE_LAYERS=9 (how many times layers are applied in forward)

# In GPT.__init__:
self.num_unique_layers = int(os.environ.get("NUM_UNIQUE_LAYERS", num_layers))
self.num_effective_layers = num_layers  # num_layers becomes effective depth
self.blocks = nn.ModuleList([Block(...) for _ in range(self.num_unique_layers)])

# Layer mapping: which unique layer to use at each effective position
# For 6→9: [0, 1, 2, 3, 4, 5, 2, 3, 4] (loop middle layers)
# Encoder: positions 0-3 (unique 0,1,2,3)
# Decoder: positions 4-8 (unique 4,5,2,3,4) — middle layers reused

# In forward:
self.layer_map = [i % self.num_unique_layers for i in range(self.num_effective_layers)]
# Or a smarter mapping that loops only the middle layers

# Skip weights remain unique per effective position (not shared):
self.skip_weights = nn.Parameter(torch.ones(num_skip_weights, model_dim))
```

### What to change from Exp 8a

| Parameter | Exp 8a | Exp 9 | Why |
|-----------|--------|-------|-----|
| NUM_LAYERS | 9 | 9 (effective) | Same depth for fair comparison |
| NUM_UNIQUE_LAYERS | 9 (all unique) | 6 | Fewer stored params, each layer gets more gradient |
| MODEL_DIM | 512 | 608 | Use freed params for width |
| NUM_HEADS | 8 | 8 | Keep same head count, head_dim increases 64→76 |
| NUM_KV_HEADS | 4 | 4 | Same GQA ratio |
| warmdown_iters | 3000 | 3000 | Keep validated setting |
| MATRIX_LR | 0.04 | sweep 0.02-0.06 | Architecture change requires LR validation |

### Code changes needed

1. **Remove register token code** from train_gpt.py:
   - Delete REGISTER_MODE env var, register_token param, reg_gate, reg_final_gate
   - Delete all register-related forward logic and attn_mask handling
   - Revert Block.forward and CausalSelfAttention.forward to not accept attn_mask
   - Revert torch.compile to always use fullgraph=True

2. **Add layer looping** to GPT class:
   - New env var: `NUM_UNIQUE_LAYERS` (default = NUM_LAYERS, i.e. no looping)
   - Create `self.blocks` with `num_unique_layers` blocks instead of `num_layers`
   - Add `self.layer_map` list mapping effective position → unique layer index
   - Modify forward encoder/decoder loops to use `self.blocks[self.layer_map[i]]`
   - Skip weights stay unique per effective position (no change to skip_weights)

3. **LR sweep** (3 short 180s runs):
   - MATRIX_LR=0.02, 0.04, 0.06 with the new architecture
   - Pick the best, then do the full 1200s run

### Risks

1. **LR mismatch**: The bigger risk. Wider model may need lower LR. Mitigated by sweep.
2. **Shared gradients**: Looped layers accumulate gradients from multiple positions per step. Muon optimizer handles this automatically (it normalizes updates), but the effective LR per unique layer is higher. May need LR reduction.
3. **Skip connection mismatch**: Skip connections pair encoder position i with decoder position (n-1-i). If both use the same unique layer, the skip might be less useful (carrying redundant info). Mitigated by keeping skip_weights unique.
4. **torch.compile compatibility**: Layer reuse via indexing should work with fullgraph=True since the computation graph is static.

### Hypothesis

- **LR sweep** will find a working LR in 0.02-0.06 range (high confidence based on competition findings)
- **Layer looping + wider** should give 0.01-0.03 bpb improvement over 8a (medium confidence)
- Expected: val_bpb ~1.27-1.29
- Optimistic: val_bpb ~1.25

### Decision rules after Exp 9

**If val_bpb < 1.27 (big improvement):**
- Architecture change is working. Try even wider (5L/672d looped to 9) or longer training (2400s).

**If val_bpb 1.27-1.30 (modest improvement):**
- Width helps but not enough alone. Try combining with longer training or different loop pattern.

**If val_bpb >= 1.30 (no improvement):**
- Layer looping doesn't help at this scale, or LR is still wrong. Try different loop mapping (only loop dead layers 4-6, keep 0,1,2,8 unique).

---

## Evaluation Plan for Experiment 9

Run `deep_eval.py` after training. Compare against Exp 8a.

### Key metrics to track

1. **val_bpb**: Must beat 1.2987. Target: 0.01-0.03 improvement.
2. **Layer utilization**: Are ALL layers now contributing? No more dead layers?
3. **Position degradation**: Does wider model improve long-range context use?
4. **Loss distribution**: Does wider model shift the hard token distribution?
5. **Compression**: Must stay under 16 MB. Expect ~14.2MB.
6. **ms/step**: Should be similar or faster (wider but fewer unique params to store).
