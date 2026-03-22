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

## Experiment 9 Results — Layer Looping + Wider Model (FAILED)

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

### Exp 9 Result: val_bpb = 1.3327 (WORSE than 8a by 0.034)

- 8,892 steps at 134.96ms/step (vs 11,603 at 103.4ms for Exp 8a)
- Model: 16,165,552 params, compressed 14,877,389 bytes
- Layer map: [0,1,2,3,4,5,0,1,2] — 6 unique, looped to 9
- LR sweep found MATRIX_LR=0.06 optimal (0.02=1.4955, 0.04=1.4732, 0.06=1.4701, 0.08=1.4699)
- **Failed because:** 23% fewer steps (135ms vs 103ms/step due to wider dim), fewer total params (16.2M vs 17.1M)

### Exp 9 Lesson
Layer looping with width increase is net negative at this scale. The step time overhead from wider dim (608 vs 512) costs more training steps than the parameter savings provide. The approach might work on 8xH100 where batch parallelism hides the latency.

---

## Experiment 10 Results — Competition Stack (SmearGate + Int6 + MLP 3x)

**val_bpb = 1.2793 pre-quant / 1.2830 int8+zlib / ~1.246 with sliding window**

Adopted the proven competition stack: SmearGate + BigramHash + OrthoInit + MLP 3x + Muon WD 0.04. Trained on H100 PCIe for 2400s (12,596 steps at 190ms/step). Model: 22.4M params, fits in 14.6MB with int6+zstd.

Key findings:
- Beats Exp 8a by 0.0157 BPB (1.2830 vs 1.2987)
- Competition stack techniques are additive: MLP 3x (~0.008), SmearGate+BigramHash (~0.004), OrthoInit (~0.002), WD (~0.001)
- Warmdown acceleration confirmed at 2.3x (consistent across all experiments)
- WD=0.04 reduced int8 quant penalty from 0.0042 to 0.0037
- Sliding window eval (stride=256) adds ~0.033 BPB improvement at eval time
- BPB conversion for sliding window: use `val_loss / ln(2) * tokens_per_byte` where tokens_per_byte=0.4104 for this tokenizer

Full analysis: [experiment10_analysis.md](experiment10_analysis.md)

## Experiment 11 Results — Hard Token Data Sampling (FAILED)

Tested 4 variants of training data manipulation to address diminishing returns after step 2000:
- 11a: Juncture-enriched shards → +0.0016 worse (shard-level too coarse)
- 11b: Rare bigram enriched → +0.0108 worse (reduced diversity hurts)
- 11c: Control baseline → 1.4386 (reference)
- 11d: Loss-based hard mining (one-time scoring at 80%, train on top 5%/25% hard) → catastrophic forgetting (wrong optimizer) then +0.0396 worse (corrected version)

**Conclusion: Data sampling is a dead end.** Diminishing returns are capacity-limited, not data-limited. The model needs more layers/capacity, not different data. Post-hoc fine-tuning on filtered data disrupts Muon-optimized weight geometry regardless of LR, optimizer, or mix ratio.

Full analysis: [exp11_hardtoken_sampling.md](exp11_hardtoken_sampling.md)

### Updated confirmed facts after 11 experiments
13. **Sliding window eval adds ~0.033 BPB** at eval time with stride=256, seq_len=1024
14. **Data sampling/curriculum does not help** — all variants equal or worse than random
15. **Post-hoc fine-tuning destroys Muon weight geometry** — even low-LR SGD/AdamW hurts
16. **Int6+zstd fits 22.4M params in 14.6MB** — 1.4MB headroom under 16MB limit

---

## Next Experiments: 12, 13, 14

### Experiment 12 — SWA (Stochastic Weight Averaging)

**Rationale:** Average model weights across multiple checkpoints during warmdown phase. Produces smoother weight landscape that quantizes better. Used by 8/19 validated submissions. Nearly free — just accumulate and average, no extra training.

**Implementation:**
- During warmdown (last 3000 steps), save model state every 100 steps
- After training, average all saved states
- Use the averaged model for final eval and compression

**Expected gain:** -0.005 to -0.01 BPB
**Risk:** None — can always fall back to non-SWA checkpoint
**Cost:** Same training run as Exp 10, just adds weight accumulation

### Experiment 13 — SwiGLU (replace ReLU-squared)

**Rationale:** SwiGLU is the standard MLP activation in modern LLMs (LLaMA, Mistral). It uses a gating mechanism that lets the model learn which features to pass through, generally more parameter-efficient than ReLU-squared.

```python
# Current (ReLU-squared):
MLP(x) = proj(relu(fc(x))²)          # 2 matrices: fc, proj

# SwiGLU:
MLP(x) = proj(swish(gate(x)) * up(x))  # 3 matrices: gate, up, proj
```

SwiGLU needs 3 matrices vs 2, so at matched param count, hidden dim reduces from 1536 to ~1024. But quality-per-param is typically better.

**Implementation:** Replace MLP class with SwiGLU variant. Keep total MLP params constant by adjusting hidden dim.

**Expected gain:** -0.005 to -0.01 BPB
**Risk:** Low — simple code change, no LR sweep needed (activation change is usually LR-stable)

### Experiment 14 — 11 Layers + Int5 MLP

**Rationale:** The single biggest architectural gain available. Competition data: 11L consistently beats 9L by 0.03-0.04 BPB across 5+ independent submissions. The param budget problem: 11L+MLP3x exceeds 16MB with int6. Solution: use int5 for MLP weights (range [-16,15]) and int6 for attention, matching PR #180 (3rd place, 1.1428 BPB).

**Param budget:**
- 11L × MLP 3x at int5-MLP/int6-attn + zstd-22 → ~15.5MB (fits)
- Needs LR sweep (3 short runs at 0.02, 0.03, 0.04)

**Implementation:**
- Add 2 more layers (NUM_LAYERS=11)
- Add int5 quantization for MLP weights
- LR sweep then full 2400s run

**Expected gain:** -0.03 to -0.04 BPB
**Risk:** Medium — needs LR sweep, int5 quant penalty (~0.029) is higher than int6 (~0.010)

### Combined projection

| State | val_bpb (sliding window) |
|-------|-------------------------|
| Exp 10 (current) | ~1.246 |
| + Exp 12 (SWA) | ~1.236-1.241 |
| + Exp 13 (SwiGLU) | ~1.231-1.236 |
| + Exp 14 (11L) | ~1.19-1.21 |

### What changes from Exp 8a

| Parameter | Exp 8a | Exp 10 | Why |
|-----------|--------|--------|-----|
| NUM_LAYERS | 9 | 9 | Keep same depth |
| MODEL_DIM | 512 | 512 | Keep same width |
| MLP_MULT | 2 | 3 | Int6 frees space for wider MLP |
| Quantization | int8+zlib | int6+zstd-22 | ~25% better compression |
| SmearGate | no | yes | Bigram context at embedding |
| BigramHash | no | yes (4096 buckets, dim=128) | Token-pair embeddings |
| OrthoInit | no (normal+zero) | yes | Required for SmearGate |
| Weight Decay | 0 | 0.04 (Muon decoupled) | Quantization-friendly weights |
| Eval | standard | sliding window stride=64 | ~0.03 BPB free improvement |
| FP16 embed | no | yes | Protect tied embedding from quant |
| TRAIN_SEQ_LEN | 1024 | 1024 | Keep same (2048 costs step time) |

### Param budget with int6+zstd

| Component | Params | Int6 bytes | Notes |
|-----------|--------|-----------|-------|
| tok_emb (fp16) | 524K | 1.0 MB | Passthrough, not quantized |
| 9 blocks (Q,K,V,O,MLP) | ~20.5M | ~13.5 MB | Int6 per-row + zstd-22 |
| Scales + control tensors | ~50K | ~0.1 MB | fp16/fp32 |
| **Total** | **~21M** | **~14.6 MB** | Under 16MB with room |

### Implementation

**SmearGate** (~512 params):
```python
class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev
```

**BigramHash** (~4096 * 128 + 128 * 512 = 590K params):
```python
class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size, bigram_dim, model_dim):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def bigram_hash(self, tokens):
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, token_ids):
        h = self.embed(self.bigram_hash(token_ids))
        h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
```

**OrthoInit**: orthogonal_(gain=1.0) for all Linear with dims >= 64, proj layers scaled by 1/sqrt(2*num_layers).

**Int6 quantization**: per-row scale = abs_max / 31, quantize to [-32, 31] in int8 container.

**Muon WD**: `p.data.mul_(1 - lr * wd)` before update in optimizer step.

### Risks

1. **torch.compile compatibility** — SmearGate uses cat+slice which may break fullgraph. Mitigated by testing with fullgraph=False first.
2. **OrthoInit + existing init conflict** — The current code has both zero-init and normal-init paths. Need to carefully layer orthogonal on top.
3. **Int6 quality loss** — 0.010 BPB quant penalty (vs 0.004 for int8). Mitigated by WD=0.04 shrinking weight magnitudes.
4. **Step time increase** — MLP 3x adds FLOPs. Expect ~120-130ms/step vs 103ms. Offset by better per-step quality.
5. **zstd dependency** — Need to pip install zstandard on the pod.

### Hypothesis

- **Expected: val_bpb ~1.22-1.25** on single GPU with sliding window eval
- **Optimistic: val_bpb ~1.20** if all techniques compose well
- **Pessimistic: val_bpb ~1.27** if int6 quality loss is worse than expected

### Decision rules after Exp 10

**If val_bpb < 1.22 (target reached):**
- Success. Submit and iterate. Try SWA, longer training, or seq_len=2048.

**If val_bpb 1.22-1.25 (close to target):**
- Add SWA (stochastic weight averaging) during warmdown. Try Muon WD sweep.

**If val_bpb >= 1.25 (disappointing):**
- Debug: check int6 quant penalty, check SmearGate is actually helping via ablation.

---

## Evaluation Plan for Experiment 10

Run deep_eval.py + sliding window eval after training.

### Key metrics to track

1. **val_bpb** (standard eval + sliding window eval): Target < 1.25
2. **Int6 quant penalty**: Should be < 0.015 BPB
3. **Layer utilization**: SmearGate should free up layers from bigram work
4. **Hard token analysis**: Expect improvement on word-initial juncture tokens
5. **Compression**: Must stay under 16 MB with int6+zstd
