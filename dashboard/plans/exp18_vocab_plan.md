# Experiment 18 — Vocabulary Increase (2048 and 4096)

## Hypothesis

66% of hard learnable bits come from word-initial single-letter predictions. A larger vocabulary merges common words into single tokens, eliminating this ambiguity. Local analysis shows:

| Vocab | Token Reduction | Word-Initial 1-char | Embedding Cost | Est. BPB |
|-------|----------------|---------------------|---------------|----------|
| 1024 (current) | — | 12.9% | 1.0 MB | 1.235 |
| 2048 | 16.2% | 10.2% | 2.0 MB | 1.138 |
| 4096 | 28.2% | 7.6% | 4.0 MB | 1.065 |

## Why the competition failed with larger vocab

PR #123 (vocab 4096, 8L) got 1.1642 and PR #200 (SP4096, 9L) got 1.2012 — both worse than the 1024-vocab winner at 1.1326. The reason: 4x larger embedding table forced fewer layers to fit in 16MB.

**Why we might succeed:** We have int5+int6 quantization which compresses much better than their int8. Our 11-layer model with int5 MLP fits in 15 MB. The extra 1-3 MB for a larger embedding is absorbable.

## Param Budget Check

| Config | Embed Params | Embed Size (int8+zstd est.) | Model Total | Est. Artifact |
|--------|-------------|---------------------------|-------------|---------------|
| 1024 vocab, 11L (current) | 524K | ~0.4 MB | 26.5M | 15.0 MB |
| 2048 vocab, 11L | 1.05M | ~0.8 MB | 27.0M | **15.4 MB** ✅ |
| 4096 vocab, 11L | 2.10M | ~1.6 MB | 28.1M | **16.2 MB** ❌ barely over |
| 4096 vocab, 10L | 2.10M | ~1.6 MB | 25.6M | **15.1 MB** ✅ (drop 1 layer) |

**2048 fits easily. 4096 needs either dropping a layer or more aggressive quantization.**

## Experiment Design

### Phase 1: Build tokenizers (local, free, ~5 min)

Already done for local analysis. For the real experiment, we need tokenizers trained on the TRAINING data (not val). The competition's `download_hf_docs_and_tokenize.py` handles this — just change the vocab_size parameter.

### Phase 2: Short training runs (300s each on 1xH100 with grad accum)

Run both vocab sizes for 300s with identical settings to compare val_bpb at the same step count.

| Run | Vocab | Layers | Batch | Seq | LR | Wallclock | Est. Cost |
|-----|-------|--------|-------|-----|-----|-----------|-----------|
| 18a | 2048 | 11 | 524K | 2048 | 0.04 | 300s | ~$0.50 |
| 18b | 4096 | 11 | 524K | 2048 | 0.04 | 300s | ~$0.50 |
| 18c | 4096 | 10 | 524K | 2048 | 0.04 | 300s | ~$0.50 |
| control | 1024 | 11 | 524K | 2048 | 0.04 | 300s | already done (1.6870) |

**Total Phase 2 cost: ~$1.50 + pod setup**

### Phase 3: Full run on winner (4850s on 1xH100)

Whichever vocab shows the best val_bpb at 300s gets a full run.

**Est. cost: ~$3.60**

## Implementation Steps

1. **Train 2048 and 4096 sentencepiece tokenizers** on FineWeb training data
   - Modify `download_hf_docs_and_tokenize.py` to accept vocab_size parameter
   - Or train directly from the existing text data on the pod

2. **Re-tokenize training + val data** with new tokenizers
   - Creates new binary shards in `data/datasets/fineweb10B_sp2048/` and `sp4096/`

3. **Modify train_gpt_submission.py** for each vocab
   - Change `VOCAB_SIZE` default
   - Point to correct tokenizer and data paths
   - Keep everything else identical

4. **Run 300s A/B/C test**
   - Compare val_bpb at step ~490

5. **Full run on winner**

## What to watch for

- **Per-token loss**: Should INCREASE (harder softmax) but total BPB should DECREASE (fewer tokens)
- **Step time**: Larger vocab = slightly larger embedding lookup but negligible effect
- **Artifact size**: 2048 should fit. 4096 may need adjustment.
- **LR sensitivity**: Larger vocab may need different LR — run sweep if 300s result is underwhelming

## Risks

1. **Tokenizer quality**: Our locally-trained tokenizer may not match the competition's carefully built one. The competition uses specific settings in `download_hf_docs_and_tokenize.py`.
2. **Data regeneration time**: Re-tokenizing 10B tokens takes ~30 min on pod.
3. **The competition already tried and failed**: Their failure was due to param budget, which we've solved. But there may be other issues (embedding training dynamics, rare token coverage).
4. **LR may not transfer**: Different vocab = different loss landscape. May need LR sweep.

## Decision Rules

**If 2048 val_bpb < control by >0.01 at 300s**: Full run. Expected final ~1.12-1.14.
**If 4096 val_bpb < 2048 by >0.01**: Consider 4096 with 10L or int5 embedding.
**If both worse than control**: Vocab increase doesn't help at this model size. Stop.
