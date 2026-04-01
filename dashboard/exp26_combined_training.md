# Exp 26: Combined Training Run (XSA-all + EMA)

**Goal**: Train a new model combining the best techniques and beat current val_bpb=1.1573.

## Architecture Changes from Current Submission

| Component | Current (exp17) | New (exp26) | Source |
|-----------|----------------|-------------|--------|
| XSA | Last 4 layers | **All 11 layers** | Top submission #1019 (1.1147 BPB) |
| Weight Averaging | SWA only | **EMA(0.997) + Tight SWA(50)** | PR #414 (proven -0.0006 BPB) |
| MLP | SwiGLU 3x | SwiGLU 3x (keep) | LeakyReLU² didn't show advantage |
| Everything else | Same | Same | — |

## Training Config

```bash
# Environment variables for train_gpt.py
NUM_LAYERS=11
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=3
VOCAB_SIZE=1024
TRAIN_SEQ_LEN=2048
TIED_EMBED_INIT_STD=0.02
LOGIT_SOFTCAP=30.0
ROPE_BASE=10000.0
QK_GAIN_INIT=1.5

# XSA on ALL layers (the key change)
XSA_LAST_N=11   # or modify use_xsa=(True) in code

# Optimizer
MATRIX_LR=0.04
SCALAR_LR=0.04
EMBED_LR=0.05
MUON_WD=0.04

# Training
TRAIN_BATCH_TOKENS=524288   # 8xH100: native; 1xGPU: use grad accum
MAX_WALLCLOCK_SECONDS=4850  # ~80 min for non-record, 600 for record
WARMDOWN_ITERS=3000

# EMA (new)
EMA_ENABLED=1
EMA_DECAY=0.997
SWA_EVERY=50               # Tight SWA alongside EMA
```

## Code Changes Required

### 1. XSA on all layers

In `train_gpt.py`, change the Block construction in GPT.__init__:

```python
# Current:
use_xsa=(i >= num_layers - 4)

# New:
use_xsa=True
```

### 2. EMA weight averaging

Add after each optimizer step in the training loop:

```python
# Initialize once before training loop:
ema_state = {n: p.detach().clone().float() for n, p in base_model.state_dict().items()}
ema_decay = 0.997

# After each optimizer.step():
with torch.no_grad():
    for n, p in base_model.state_dict().items():
        ema_state[n].lerp_(p.float(), 1 - ema_decay)

# Before quantization (end of training):
# Load EMA weights into model
base_model.load_state_dict({n: v.to(base_model.state_dict()[n].dtype)
                            for n, v in ema_state.items()})
```

### 3. Keep existing

- SwiGLU MLP (3x expansion)
- SmearGate
- OrthoInit
- GQA (8H/4KV)
- U-Net skip connections
- Tied embeddings
- Logit softcap=30.0
- Mixed quantization (int5-MLP + int6-attn + int8-embed + zstd-22)

## Running on Different Hardware

### Option A: RunPod 1xH100 SXM ($2.69/hr)

Matches original setup exactly. No grad accumulation needed.

```bash
# On RunPod with network volume
cd /runpod-volume/parameter-golf

# Training
RUN_ID=exp26_xsa_all_ema \
TRAIN_BATCH_TOKENS=524288 \
MAX_WALLCLOCK_SECONDS=4850 \
WARMDOWN_ITERS=3000 \
XSA_LAST_N=11 \
EMA_ENABLED=1 EMA_DECAY=0.997 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# Cost: ~$3.60 for ~80 min
```

### Option B: Vast.ai RTX 4090 ($0.27/hr)

Needs gradient accumulation. Training is in bf16 (fresh model, not dequantized).

```bash
# Grad accum: 524288 / 16384 = 32
TRAIN_BATCH_TOKENS=16384 \
GRAD_ACCUM_STEPS=32 \
MAX_WALLCLOCK_SECONDS=25200 \  # ~7 hours
WARMDOWN_ITERS=3000 \
XSA_LAST_N=11 \
EMA_ENABLED=1 EMA_DECAY=0.997 \
python -u train_gpt.py

# Cost: ~$1.90 for ~7 hours
# Note: train_gpt.py needs modification to support GRAD_ACCUM_STEPS on single GPU
```

### Option C: Vast.ai A100 PCIe ($0.52/hr)

Faster than 4090, larger VRAM (80GB).

```bash
TRAIN_BATCH_TOKENS=65536 \
GRAD_ACCUM_STEPS=8 \
MAX_WALLCLOCK_SECONDS=14400 \  # ~4 hours
...

# Cost: ~$2.10 for ~4 hours
```

## Post-Training Pipeline

After training completes, the script automatically:

1. **SWA averaging** — averages checkpoints from warmdown phase
2. **Serialize unquantized model** — `final_model.pt` (~106MB)
3. **Int8+zlib quantization** — roundtrip eval
4. **Mixed quantization** — int5-MLP + int6-attn + int8-embed + zstd-22
5. **Roundtrip eval** — verify mixed quant BPB (should be ~1.19)
6. **LoRA TTT eval** — train-then-score, should give final BPB

## Expected Results

| Metric | Current (exp17) | Expected (exp26) | Source |
|--------|----------------|-------------------|--------|
| Pre-TTT BPB | 1.1914 | ~1.18 | XSA-all + EMA each ~-0.001 to -0.003 |
| Post-TTT BPB | 1.1573 | ~1.15 | Proportional improvement |
| Artifact size | 15.75 MB | ~15.75 MB | Same quantization |

Conservative estimate: **-0.003 to -0.007 BPB improvement** over current 1.1573.

## Important Notes

- **bfloat16 during training is fine** — model starts with random weights, trains natively in bf16
- **bfloat16 for eval of quantized model is NOT fine** — dequantized int5/int6 weights lose precision, must stay float32 with autocast
- **XSA_LAST_N env var** may not exist in current code — need to add it or hardcode `use_xsa=True`
- **EMA_ENABLED env var** may not exist — need to implement in training loop
- **Validate on H100 first** with a quick 300s run before committing to full 80 min

## Risks

1. XSA-all may increase step time (more XSA computation per layer)
2. XSA-all may not help if the model was already trained with XSA-last4 patterns
3. EMA + SWA interaction needs tuning (decay rate, SWA frequency)
4. Grad accumulation on single GPU may have numerical differences vs multi-GPU
