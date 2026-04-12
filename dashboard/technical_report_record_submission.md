# Technical Report: From 1.1172 to 1.0825 — Record Submission Journey

**Author**: swapp1990
**Date**: April 11, 2026
**Final Result**: val_bpb = 1.0825 (3-seed mean, std 0.0012), 10-min track, 8xH100 SXM

---

## Part I — The Final Winning Stack

### 1. Architecture

| Component | Choice |
|-----------|--------|
| Tokenizer | SP8192 (8192 vocab, seq_len 2048) |
| Layers | 11 unique, no depth recurrence |
| Dim | 512 |
| Heads | 8 query / 4 KV |
| MLP | 4x LeakyReLU² (2048 hidden, 2 matmuls) |
| QK-Gain init | 5.25 (learnable per-head) |
| XSA | All 11 layers |
| Extras | SmearGate, U-Net skips, tied embeddings, logit softcap 30 |
| Params | ~36M |

### 2. Training (600s on 8xH100 SXM)

- Muon (matrix_lr=0.022, WD=0.095, momentum=0.99, row-normalized NS-5)
- AdamW embeddings (lr=0.03, WD=0.085), Adam scalars (lr=0.02)
- EMA decay 0.9965 (replaces SWA)
- Warmup 20 steps, warmdown 72% (~6500 iters)
- 524K tokens/step, ~7360 steps, 81.5 ms/step

### 3. Quantization (GPTQ + SDClip)

Pipeline runs on a **fresh uncompiled** GPT (torch.compile breaks forward hooks):

1. Collect Hessians from 32 calibration batches
2. GPTQ column-ordered block error compensation (block_size=128) with **SDClip** per-row scale = k × std(row) / clip_range
   - int6 (k=12.85) for attention + MLP
   - int8 (k=20.0) plain SDClip for embeddings (Hessian dim mismatch — skip GPTQ)
3. Byte-shuffle (stride=2)
4. Brotli-11

Result: **~0.45 bytes/param**, ~15.6 MB artifact.

### 4. Test-Time Training (SGD all-weights, ~355s)

On a fresh uncompiled GPT loaded with dequantized weights:
- Per document: SGD (lr=0.005, momentum=0.9) on 2048-token chunks, score first 2048 tokens, restore
- Docs <32 tokens scored without adaptation
- Sharded across 8 GPUs

### 5. Pipeline Flow

```
Training (600s)            → raw val_bpb ~1.12
GPTQ + SDClip (uncompiled) → post-quant ~1.14
SGD TTT (uncompiled)       → post-TTT  ~1.08
Byte-shuffle + Brotli-11   → ~15.6 MB
```

### 6. Why Each Piece Matters

- **SP8192** — each token covers 3–4x more bytes; every top entry uses it.
- **LeakyReLU²** — 2 matmuls vs SwiGLU's 3, 50% wider hidden at same params.
- **WD=0.095** — the single most important hyperparameter. Makes weights near-Gaussian so SDClip clips cleanly (37x lower quant MSE vs WD=0.04).
- **SDClip** — `k × std` scale → low-entropy quantized values that Brotli compresses to ~0.45 B/param (vs 0.69 for min/max).
- **GPTQ** — Hessian-weighted column ordering + block error compensation recovers most quantization loss.
- **Fresh uncompiled models for GPTQ/TTT** — torch.compile silently breaks both hooks and per-doc SGD.
- **SGD all-weights TTT** — adapts all 36M weights per document; ~0.05 BPB gain.

### 7. Reproducibility

```bash
pip install brotli sentencepiece
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
SEED=314 torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=42  torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=999 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Expected: val_bpb ~1.08 ± 0.001 on 8xH100 SXM (RunPod parameter-golf image).

### 8. 3-Seed Results

| Seed | Steps | Raw BPB | TTT BPB | Artifact |
|------|-------|---------|---------|----------|
| 314 | 7,358 | 1.1211 | 1.0836 | 15.60 MB |
| 42  | 7,364 | 1.1197 | 1.0825 | 15.57 MB |
| 999 | 7,361 | 1.1197 | 1.0813 | 15.64 MB |
| **Mean** | | **1.1202** | **1.0825** | |
| **Std**  | | | **0.0012** | |

---

## Part II — The Journey

### Starting Point: 1.1172 (non-record, 1xH100, 80 min)

Our best pre-record config:
- 7 unique blocks, dim=624, depth recurrence [0,1,2,3,4,3,4,3,4,5,6,5,6] = 13 effective layers
- SwiGLU 3x MLP, XSA, SmearGate, U-Net skips, QK-Gain 1.5
- SP1024 tokenizer, SWA averaging
- GPTQ int5/int6/int8 + zstd-22, SGD all-weights TTT
- ~25.6M params, 15.2 MB

Merged SOTA: **1.0810**. Gap to close: **0.036 BPB**.

### Step 1 — 10-Minute Track Shock (Exp 53, 58)

Direct port of the 1xH100 architecture to 8xH100/600s: **val_bpb 1.1456** — catastrophically worse. Depth recurrence cost 126 ms/step → only 4,747 steps vs 14,654 on 1xH100.

Benchmarked 6 configs:

| Config | ms/step | Steps@600s | val_bpb (60s smoke) |
|--------|---------|------------|----------|
| 7L dim=624, 13 eff (original) | 127 | 4,747 | 1.4296 |
| 7L dim=624, 11 eff | 108 | 5,560 | 1.4025 |
| **11L dim=512, no recurrence** | **71** | **8,450** | **1.3516** |

**Insight**: on the 10-min track, more unique layers at smaller dim beats depth recurrence — 78% more steps with better quality per step. Confirmed by LoopLM research: loops help reasoning, not perplexity.

**Bug**: first TTT attempt gave val_bpb=1.82. Root cause: `torch.compile` mutates the model in-place, breaking load-train-score-restore. Fix: fresh uncompiled `ttt_model = GPT(...)`.

### Step 2 — LeakyReLU² (Exp 59)

Replaced SwiGLU with `leaky_relu(up(x), 0.5) ** 2`:
- 2 matrices instead of 3 → 50% wider hidden at same param count
- 64 ms/step vs 66 ms/step
- **-0.006 to -0.009 BPB** across metrics

### Step 3 — SP8192 Tokenizer (Exp 60)

Switched SP1024 → SP8192. Each token covers ~3–4x more bytes. Bigger embedding (4.2M vs 0.5M params) forced dropping from 10 to 7 layers under our old 0.66 B/param compression. Raw BPB matched SP1024 despite fewer layers.

### Step 4 — The Compression Revolution (Exp 61–62)

We wanted 11L + MLP 4x + SP8192 (~36M params), but 0.66 B/param = 24 MB. SOTA fits it at **0.44 B/param**. How?

**SDClip**: `scale = k × std(row) / clip_range` instead of min/max. Produces lower-entropy values that compress much better:

| Method | bytes/param | Size |
|--------|-------------|------|
| Min/max int6 + brotli | 0.689 | 24.8 MB (over) |
| SDClip int6 + brotli | 0.455 | 16.0 MB (fits) |

**But SDClip destroyed quality with WD=0.04**:

| Quantization | val_bpb | Δ |
|--------------|---------|----|
| Raw float | 1.1287 | — |
| Min/max int6 | 1.1347 | +0.006 |
| SDClip k=12.85 (WD=0.04) | 1.2953 | **+0.167** |

The weights had heavy tails — outliers beyond 12.85σ got clipped.

### Step 5 — The WD=0.095 Unlock (Exp 62b)

SOTA uses **WD=0.095**. Retrained same architecture:
- SDClip MSE: **0.003230 → 0.000087** (37x)
- Quant cost: **0.167 → 0.014 BPB**
- Raw val_bpb actually improved: 1.1306 → 1.1213

This one hyperparameter unlocked the entire SOTA-class stack.

### Step 6 — GPTQ Integration

Two integration bugs cost real money:
- **torch.compile vs hooks**: compiled model's `named_modules()` renames break forward hooks — Hessians silently skipped. Fix: fresh uncompiled `gptq_model = GPT(...)`.
- **Embedding Hessian mismatch**: weight (8192, 512) but hook sees integer token IDs. Fix: skip GPTQ for embeddings, use plain SDClip int8.

### What Failed

- **Depth recurrence on 10-min track** — compute/step cost outweighs param efficiency at 600s.
- **Mixed int5/int6/int8** — different entropy per layer type hurts compression uniformity.
- **SDClip with WD=0.04** — single biggest blocker; weeks of "fix compression" experiments when the real fix was training HP.
- **torch.compile + post-training pipeline** — 2 wasted 8xH100 runs (~$40) before diagnosing root cause.

### Progression Table

| Exp | val_bpb | Key change | Cost |
|-----|---------|-----------|------|
| exp43 (1xH100, non-record) | 1.1172 | GPTQ + SGD TTT | ~$11 |
| exp53 (1xH100) | 1.1159 | QK-Gain 4.0 | ~$11 |
| exp58 (8xH100, 10 min) | 1.1456 | 10L dim=512, no recurrence | ~$6 |
| exp59 | ~1.14 | LeakyReLU² | ~$6 |
| exp60 | 1.1897 | SP8192, 7L | ~$6 |
| exp61b | 1.1882 | 8L + 2x recurrence | ~$6 |
| exp62 | 1.1768 post-TTT | 11L MLP4x, SDClip (broken) | ~$9 |
| **exp62b** | **1.0825** | **WD=0.095 + full pipeline** | ~$9 |

Final compute: ~$30. Total exploration: ~$150.

### Gap Analysis vs SOTA (1.0810)

| | Merged SOTA | Ours |
|---|---|---|
| val_bpb | **1.0810** | 1.0825 (+0.0015) |
| Depth recurrence | 3-layer (17 eff) | None (11 eff) |
| Parallel residuals | L7+ (GPT-J style) | None |
| Score-first TTT | 3 epochs, 32K chunks, cosine LR | 1 pass, 2K chunks |
| Partial RoPE | 16/64 dims | Full RoPE |
| Train batch | 786K tok/step | 524K tok/step |

Any of depth recurrence, parallel residuals, or score-first TTT could close (or exceed) the remaining 0.0015 BPB.
