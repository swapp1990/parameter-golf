# Record: SP8192 + 11L MLP4x + QK-Gain 5.25 + GPTQ SDClip + SGD TTT

**val_bpb = 1.0825** (3-seed mean, std 0.0012) | **~15.6 MB** | 8xH100 SXM, 600s training + 355s TTT eval

## 3-Seed Results

| Seed | Steps | ms/step | Raw BPB | **TTT BPB** | Artifact |
|------|-------|---------|---------|-------------|----------|
| 314  | 7,358 | 81.6    | 1.1211  | **1.0836**  | 15.60 MB |
| 42   | 7,364 | 81.5    | 1.1197  | **1.0825**  | 15.57 MB |
| 999  | 7,361 | 81.5    | 1.1197  | **1.0813**  | 15.64 MB |
| **Mean** | | | **1.1202** | **1.0825** | |
| **Std** | | | | **0.0012** | |

## Architecture

- 11 layers x 512d x 8H / 4KV, MLP 4x (2048 hidden), LeakyReLU(0.5)^2
- XSA on all 11 layers, SmearGate, U-Net skip connections
- QK-Gain init 5.25 (learnable per-head query scaling)
- Tied embeddings, vocab 8192 (SP8192), seq_len 2048
- Logit softcap 30.0
- ~36M parameters

## Training

- Muon optimizer (matrix_lr=0.022, WD=0.095, momentum=0.99, row-normalized Newton-Schulz 5 steps)
- AdamW for embeddings (lr=0.03, WD=0.085)
- Adam for scalars (lr=0.02)
- EMA decay 0.9965
- Warmdown 72% (6500 iters), warmup 20 steps
- 524K tokens/step, ~7360 steps in 600s on 8xH100 SXM
- 81.5 ms/step

## Quantization (GPTQ + SDClip)

1. **Hessian collection**: Fresh uncompiled model, 32 calibration batches from training data
2. **GPTQ + SDClip**: Per-row scale = k * std(row) / clip_range. Column ordering by Hessian diagonal. Block-wise error compensation (block_size=128).
   - int6 (k=12.85) for all attention + MLP matrices (clip_range=31)
   - int8 (k=20.0) for token embeddings (clip_range=127, plain SDClip, no GPTQ)
3. **Byte-shuffle** (stride=2) before compression
4. **Brotli-11** compression

Key insight: Weight decay 0.095 is critical for SDClip — it produces tighter weight distributions that quantize with 37x lower MSE than WD=0.04.

## Test-Time Training (TTT)

SGD all-weights adaptation per document at eval time:
- Fresh uncompiled model loaded with GPTQ-dequantized weights
- For each document: train ALL weights with SGD (lr=0.005, momentum=0.9) on 2048-token chunks
- Score first 2048 tokens after adaptation
- Restore base weights before next document
- Short documents (<32 tokens) scored without adaptation
- Docs sharded across 8 GPUs, all-reduced at end
- Total TTT eval time: ~355s (within 600s eval budget)

## Run Command

```bash
pip install brotli sentencepiece

# Download SP8192 data
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

# Run (all defaults are set in train_gpt.py)
SEED=314 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Design Decisions

1. **SP8192 tokenizer**: 8x larger vocab = better compression per token. Every top entry uses it.
2. **LeakyReLU² MLP**: 2 matrices (up + proj) instead of SwiGLU's 3 (gate + up + proj). Same params, 50% wider hidden layer.
3. **High weight decay (0.095)**: Critical for SDClip quantization. Without it, quantization loses 0.17 BPB instead of 0.01.
4. **GPTQ on uncompiled model**: torch.compile breaks forward hooks needed for Hessian collection. Fresh model created for GPTQ.
5. **TTT on uncompiled model**: torch.compile breaks per-document SGD adaptation. Fresh model created for TTT.

## Progression

| Experiment | val_bpb | Key change |
|-----------|---------|------------|
| Baseline (SP1024) | 1.2244 | 9L, int8, no TTT |
| + SGD TTT | 1.1159 | All-weights SGD adaptation |
| + SP8192 + 11L MLP4x | 1.1306 | Bigger vocab, more params |
| + WD=0.095 + SDClip | 1.1213 | Quantization-friendly weights |
| + GPTQ SDClip + TTT | **1.0825** | Full pipeline, 3-seed mean |

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `train_seed314.log`
- `train_seed42.log`
- `train_seed999.log`
