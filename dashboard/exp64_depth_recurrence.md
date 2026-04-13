# Exp 64 — Depth Recurrence on SP8192 Stack

**Goal**: Close the 0.0015 gap to SOTA 1.0810 by adding depth recurrence. Push raw BPB down first; TTT should follow proportionally.

## Current Baseline (exp62b)

| Metric | Value |
|--------|-------|
| Architecture | 11L × 512d × MLP 4x LeakyReLU², SP8192 |
| Raw val_bpb (EMA) | **1.1202** (3-seed mean) |
| Post-TTT val_bpb | 1.0825 |
| Step time | 81.5 ms/step |
| Steps @ 600s | ~7360 |

## What SOTA Does (PR #1493)

- 11 physical layers, **3-layer depth recurrence (L3-5)** activated at 35% of training → 17 effective layers
- Raw BPB (sliding): 1.0827 → TTT: **1.0810**

Depth recurrence = loop some physical layers multiple times per forward pass. Adds compute/step but zero params.

## Key Trade-off

More effective layers → better BPB **per step**, but more ms/step → fewer steps @ 600s. Sweet spot depends on which dominates.

Our exp58 benchmark showed recurrence *hurt* us at that config. But we had 7L dim=624 with heavy recurrence (13 eff). SOTA's 3-layer loop on 11L is lighter and might pay off now.

## Run Configs

| Config | Physical | Loop | Effective | Activation | Expected ms/step |
|--------|----------|------|-----------|-----------|------------------|
| A (baseline) | 11 | — | 11 | — | 81.5 |
| B: loop L4-5 x2 | 11 | [4,5] twice | 13 | @50% | ~95 |
| C: loop L3-5 x2 | 11 | [3,4,5] twice | 14 | @35% | ~100 |
| D: loop L4-5 x2 + L3-5 x2 progressive | 11 | mini + big | 14 | @50%, @65% | ~100 |
| E: 10L + loop L3-5 x2 | 10 | [3,4,5] twice | 13 | @35% | ~90 |

Start with **C** (matches SOTA exactly) then D/E if C underperforms.

## Measurement

Each run: full 600s training, report:
- Raw val_bpb (EMA, pre-quant)
- Post-quant val_bpb (GPTQ+SDClip)
- Step time, total steps
- Artifact size

No TTT for these exploration runs (save compute). Only run TTT on the best raw config.

## Execution Plan

1. Implement recurrence toggle in train_gpt.py (already present in SOTA refs — port cleanly)
2. Smoke test 60s to verify step time
3. Full 600s single-seed for each config (~$1.50 each on 8xH100)
4. Pick best, run 3 seeds + TTT for final validation

Budget: ~$15 for full exploration.
