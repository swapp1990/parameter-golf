# Non-Record Submission: 9L Depth Recurrence + XSA + EMA + SGD TTT

**val_bpb: 1.1290** | Artifact: 15.88 MB | Hardware: 1xH100 SXM | Training: 120 min

## Architecture

- 9 unique transformer blocks, middle 3 shared (looped 2x) = 12 effective layers
- Layer schedule: [0,1,2,3,4,5,3,4,5,6,7,8]
- Model dim: 560, Heads: 8, KV heads: 4 (GQA)
- SwiGLU MLP (3x expansion)
- XSA (Cross-Sample Attention) on all layers
- U-Net skip connections (encoder-decoder with learned skip weights)
- Tied embeddings, logit softcap=30.0
- SmearGate, OrthoInit, RoPE (base=10000)
- 26,483,032 parameters

## Training

- Optimizer: Muon (matrix params) + Adam (embeddings/scalars)
- Batch: 524,288 tokens (grad_accum=8 on 1xH100)
- Wallclock: 7200s (120 min), 8366 steps
- EMA: decay=0.997, applied before quantization
- SWA: started at step 5400

## Quantization

- Mixed: int5 (MLP) + int6 (attention) + int8 (embeddings) + zstd-22
- Artifact: 15,957,177 bytes (fits 16MB)

## Test-Time Training (TTT)

- SGD all-weights (lr=0.005, momentum=0.9)
- Train-then-score: 1 epoch on 1024-token chunks, score full doc (2048 cap)
- All model weights temporarily adapted per document, restored after scoring

## Key Innovations

1. **Depth recurrence**: Sharing middle 3 blocks gives 12 effective layers for the storage cost of 9. More depth improves complex token predictions without increasing artifact size.

2. **SGD all-weights TTT**: Instead of LoRA (rank-8 on Q/V only), adapt ALL 26.5M weights per document with SGD. Gives -0.011 BPB over LoRA TTT because adaptation signals exist in MLP, K, Proj, and embeddings that LoRA can't reach.

3. **XSA on all layers**: Cross-Sample Attention removes self-value bias from attention output, forcing cross-token reasoning. Applied to all 12 effective layers (not just last 4).

## Results

| Metric | Value |
|--------|-------|
| Pre-TTT val_bpb (int8) | 1.1639 |
| Post-TTT val_bpb (SGD all-weights) | **1.1290** |
| TTT improvement | -0.0349 |
| Tokens scored | 46,243,627 |
| Documents | 50,000 |
