# Experiment 16 — Optimization Sweep

## Strategy

Maximize experiments per dollar. Split into 3 batches:

### Batch 1: Eval-Only (no training, load Exp 15 checkpoint)
These only need forward passes on the validation set. Run all on 1xH100 (~$2.69/hr).

| # | Experiment | What to measure | Time |
|---|-----------|----------------|------|
| 2 | Sliding window stride=64 (vs 256) | val_loss improvement | ~10 min |
| 7 | Head-level ablation (zero each of 88 heads) | Dead head identification | ~5 min |
| 11 | MLP removal from L5-L7 | Loss impact | ~3 min |
| 12 | SmearGate-only vs BigramHash-only vs both | Ablation | ~5 min |

**Est. time: ~25 min, cost: ~$1.10**

### Batch 2: Short Training Runs (300s each on 1xH100)
Compare against Exp 15's step-1000 baseline (val_bpb=1.5087 from 11c control).
All use Exp 15 architecture (11L SwiGLU) but on 1xH100 with batch=65K.

| # | Experiment | Change | Time |
|---|-----------|--------|------|
| 3 | Partial RoPE (16/64 dims) | Modify Rotary class | 8 min (compile+300s) |
| 4 | Heterogeneous MLP (4x endpoints, 2x middle) | Modify Block init | 8 min |
| 5 | LN Scale (1/√(layer+1)) | Add scale to RMSNorm | 8 min |
| 6 | EMA (decay=0.997) vs SWA | Replace SWA logic | 8 min |

**Est. time: ~35 min, cost: ~$1.60**

### Batch 3: Full Submission Run (8xH100, 600s)
Combine all winning techniques from Batch 1+2 with Int5+Int6+zstd + Late QAT.

| # | Experiment | Time |
|---|-----------|------|
| 1 | Full run with best config + Int5/Int6 quantization + Late QAT | ~15 min |

**Est. time: ~15 min, cost: ~$5.40**

### Skip (not worth the time)
- #8 (Larger vocab): Requires dataset regeneration, too complex
- #9 (Layer looping): Already failed in Exp 9, middle layers are now useful
- #10 (Head classification): Research-grade, not competition-ready
- #13 (Hard token sampling): Already proved dead in Exp 11

**Total estimated cost: ~$8.10**
**Total estimated time: ~75 min**

## Execution Order

1. Resume 1x pod (or create new with shared volume in US-MO-1)
2. Upload Exp 15 checkpoint
3. Run Batch 1 (all eval-only, sequential)
4. Run Batch 2 (4 short training runs, sequential)
5. Analyze results, pick winners
6. Create 8xH100 pod
7. Run Batch 3 (full submission run with winning config)
8. Download checkpoint + run int5/int6 quantization
9. Stop all pods
