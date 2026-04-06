# Exp 47: What Do Recurrence Passes Actually Do?

**Date**: 2026-04-06
**Model**: exp40-D float checkpoint (7 unique blocks, dim=624, 26.5M params)
**GPU**: 1xH100 SXM (RunPod)

---

## 1. The Question

Our model reuses blocks 3-4 three times and blocks 5-6 twice, creating 13 effective layers from 7 unique blocks. The trained schedule is `[0,1,2,3,4,3,4,3,4,5,6,5,6]`.

We know this works well (val_bpb=1.1172 with GPTQ+TTT). But we don't understand what's happening inside the recurrence. When block 3 runs for the second and third time, is it:
- **Converging** — refining toward a fixed-point representation, each pass making smaller corrections?
- **Transforming** — building progressively different representations, each pass doing distinct work?

The answer has direct implications for exp44 (Relaxed Recurrence LoRA). If passes converge, per-pass adapters would fight the convergence. If passes do distinct work, per-pass adapters would let each pass specialize.

## 2. Method

We hooked block 3's output at each of its 3 appearances in the forward pass (positions 3, 5, and 7 in the schedule). For 64 validation sequences (131K tokens), we recorded the hidden state output and computed:

- **Cosine similarity** between pass outputs — how similar are the directions?
- **L2 distance** — how far apart in absolute terms?
- **Mean norm** — are representations growing, shrinking, or stable?
- **Relative change** — how much does each pass move the representation relative to its input magnitude?

## 3. Results

### Block 3 produces increasingly different representations across passes

**Cosine similarity between passes:**

| Comparison | Cosine Similarity |
|------------|------------------|
| Pass 1 → Pass 2 | 0.787 |
| Pass 2 → Pass 3 | 0.859 |
| Pass 1 → Pass 3 | **0.595** |

Pass 1 and pass 3 are the **least similar** pair. If block 3 were converging to a fixed point, pass 1→3 would be the *most* similar (both approaching the attractor). Instead, each pass moves the representation further from where it started.

**Representation norms grow monotonically:**

| Pass | Position in Schedule | Mean Norm | Growth vs Prior |
|------|---------------------|-----------|-----------------|
| 1 | 3 | 188,481 | — |
| 2 | 5 | 221,943 | +18% |
| 3 | 7 | 245,646 | +11% |

A fixed-point attractor would stabilize the norm. Instead, the block pumps energy into the representation with each pass — amplifying features that downstream blocks need.

**Each pass transforms less, but still substantially:**

| Transition | Relative Change (delta norm / input norm) |
|------------|------------------------------------------|
| Pass 1 → Pass 2 | 0.741 (74% of input magnitude) |
| Pass 2 → Pass 3 | 0.579 (58% of input magnitude) |

The changes diminish (0.74 → 0.58) but remain large. At a fixed point, relative change would be near zero. Pass 3 is still doing significant work.

## 4. Interpretation

### It's iterative refinement, not convergence

The data paints a clear picture: block 3 applies the same weights to progressively more abstract inputs, producing a **directional trajectory** rather than convergence to a fixed point.

- **Pass 1** (after blocks 0-2): Processes shallow features — token identity, positional information. The input is still close to the raw embedding.
- **Pass 2** (after blocks 0-2-3-4): Processes medium-depth features. Attention now operates on richer representations where tokens "know" about their neighbors' semantics.
- **Pass 3** (after blocks 0-2-3-4-3-4): Processes deep features — syntactic roles, longer-range dependencies. This is the most abstract input the block sees.

Each pass applies identical Q, K, V, and MLP projections, but the *input distribution* is different each time. The block has learned weights that work across all three distributions — a compromise that creates directional refinement rather than convergence.

### Adjacent passes are more similar than distant ones

Cosine similarity is 0.86 for pass 2→3 but only 0.60 for pass 1→3. This makes sense: each pass makes an incremental transformation, so consecutive outputs are closer together. But the cumulative effect is a large shift — pass 3's output is quite different from pass 1's.

### The `resid_mix` bottleneck is real

Each block blends its running state `x` with the original embedding `x0`:
```python
x = resid_mix[0] * x + resid_mix[1] * x0
```

This blending ratio is fixed across all passes. But pass 1's `x` is shallow (3 layers deep) while pass 3's `x` is deep (7 layers deep). The optimal blend is almost certainly different for each pass — more `x0` early (raw tokens still informative), less `x0` later (deep features dominate). The block can't make this distinction.

## 5. Implications for Exp44 (Relaxed Recurrence LoRA)

The activation data strongly supports adding per-pass adapters:

1. **Passes do distinct work** (cosine sim 0.60-0.86) — there's room for specialization
2. **Pass 3 still transforms significantly** (relative change 0.58) — it's not redundant
3. **The `resid_mix` bottleneck is real** — even just making `resid_mix` pass-aware (~3,744 params) could help
4. **Norms grow per pass** — per-pass scaling could be valuable

The minimal experiment: give each recurrence pass its own `resid_mix`. If that improves the loss, full per-pass LoRA adapters (on Q/V projections) should help more.

## 6. Implications for GPTQ

The growing norms across passes (188K → 222K → 246K) mean quantization errors in block 3 get amplified through recurrence. A rounding error on pass 1 feeds into a larger-norm input on pass 2, which feeds into an even larger-norm input on pass 3. This compounds the damage.

This explains why GPTQ was so effective (-0.009 BPB): reducing rounding errors in blocks 3-4 has outsized impact because those errors propagate through 3 passes with growing magnitude.

---

## Appendix: Schedule Ablation

We also ran the trained model with different layer schedules at inference time (same weights, different forward pass order). These results should be interpreted carefully: they measure **how dependent the trained model is on its exact schedule**, not what would happen if we trained with different schedules. A model run with the wrong architecture will obviously perform badly.

| Schedule | Eff Layers | val_bpb | Delta vs trained |
|----------|-----------|---------|-----------------|
| `[0,1,2,3,4,3,4,3,4,5,6,5,6]` (as trained) | 13 | 1.1493 | baseline |
| `[0,1,2,3,4,3,4,5,6,5,6]` (2x) | 11 | 1.6919 | +0.543 |
| `[0,1,2,3,4,5,6]` (no recurrence) | 7 | 2.7739 | +1.625 |
| `[0,1,2,3,4,3,4,3,4,5,6]` (3x middle only) | 11 | 2.6173 | +1.468 |
| `[0,1,2,3,4,5,6,5,6]` (2x end only) | 9 | 1.8744 | +0.725 |
| `[0,1,2,3,4,3,4,3,4,3,4,5,6,5,6,5,6]` (4x) | 17 | 3.8153 | +2.666 |

SGD TTT was also evaluated on the top 3 schedules (500 docs). TTT recovered less than 6% of the gap for the 2x schedule and 0.2% for no-recurrence, confirming that TTT adapts weights but can't fix a wrong computational graph.

One observation from this data: removing end recurrence (blocks 5-6) hurts more than removing middle recurrence (blocks 3-4) — BPB +1.47 vs +0.73. This is partly confounded by skip weight mismatches, but suggests the decoder passes are more critical, likely because they sit closest to the output and consume U-Net skip connections.

---

**Raw data**: [exp47_recurrence_ablation_results.json](exp47_recurrence_ablation_results.json)
**Full log**: [exp47_recurrence_ablation.log](exp47_recurrence_ablation.log)
**Script**: `scripts/experiments/exp47_recurrence_ablation.py`
