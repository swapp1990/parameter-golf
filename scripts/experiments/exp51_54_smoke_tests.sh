#!/bin/bash
# Phase 2 Smoke Tests: 500 steps each, 1xH100
# Run from /runpod-volume/parameter-golf

set -e
cd /runpod-volume/parameter-golf

COMMON_ENV="DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 NUM_LAYERS=7 MODEL_DIM=624 MLP_MULT=3 \
LAYER_SCHEDULE=0,1,2,3,4,3,4,3,4,5,6,5,6 \
XSA_LAST_N=7 TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=524288 \
MAX_WALLCLOCK_SECONDS=700 \
WARMDOWN_ITERS=150 \
VAL_LOSS_EVERY=100 \
EMA_ENABLED=0 \
TTT_ENABLED=0"

TRAIN_SCRIPT=records/track_non_record_16mb/2026-03-24_11L_XSA_SwiGLU_LoRATTT_1xH100/train_gpt.py

echo "============================================"
echo "Exp 53: QK-Gain=4.0 (baseline uses 1.5)"
echo "============================================"
eval $COMMON_ENV RUN_ID=exp53_qkgain4 QK_GAIN_INIT=4.0 \
  torchrun --standalone --nproc_per_node=1 $TRAIN_SCRIPT \
  2>&1 | tee /runpod-volume/exp53.log
echo "Exp 53 done"

echo ""
echo "============================================"
echo "Exp 54: Batch=786K (baseline uses 524K)"
echo "============================================"
eval $COMMON_ENV RUN_ID=exp54_batch786k TRAIN_BATCH_TOKENS=786432 \
  torchrun --standalone --nproc_per_node=1 $TRAIN_SCRIPT \
  2>&1 | tee /runpod-volume/exp54.log
echo "Exp 54 done"

echo ""
echo "============================================"
echo "Baseline: Standard config (for comparison)"
echo "============================================"
eval $COMMON_ENV RUN_ID=exp_baseline \
  torchrun --standalone --nproc_per_node=1 $TRAIN_SCRIPT \
  2>&1 | tee /runpod-volume/exp_baseline.log
echo "Baseline done"

echo ""
echo "============================================"
echo "All smoke tests complete. Compare val_loss at step 500:"
echo "============================================"
echo "Baseline:"
grep "val_loss" /runpod-volume/exp_baseline.log | tail -3
echo ""
echo "Exp 53 (QK-Gain=4.0):"
grep "val_loss" /runpod-volume/exp53.log | tail -3
echo ""
echo "Exp 54 (Batch=786K):"
grep "val_loss" /runpod-volume/exp54.log | tail -3
