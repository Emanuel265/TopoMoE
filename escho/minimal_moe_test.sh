#!/bin/bash

# =========================
# Basic Config
# =========================
SEQ_LEN=2048
NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16

GLOBAL_BATCH_SIZE=512
MICRO_BATCH_SIZE=8
NUM_GPUS=64

TRAIN_TOKENS=300000000000
WARMUP_TOKENS=375000000
LR_DECAY_TOKENS=300000000000

LR=1.2e-4
MIN_LR=1.0e-6

# =========================
# MoE Config
# =========================
NUM_EXPERTS=128
MOE_LOSS_COEFF=0.01

# =========================
# Paths
# =========================
DATA_PATH=/data/the_pile_public_merged_nopreprocessing/pile_text_document
VOCAB_PATH=/data/the_pile_public_merged_nopreprocessing/gpt2-vocab.json
MERGE_PATH=/data/the_pile_public_merged_nopreprocessing/gpt2-merges.txt

OUTPUT_DIR=./output
CHECKPOINT_PATH=${OUTPUT_DIR}/checkpoints
TENSORBOARD_DIR=${OUTPUT_DIR}/tensorboard
LOG_FILE=${OUTPUT_DIR}/train.log

mkdir -p ${CHECKPOINT_PATH}
mkdir -p ${TENSORBOARD_DIR}

# =========================
# DeepSpeed Config
# =========================
cat <<EOT > ds_config.json
{
  "train_batch_size": ${GLOBAL_BATCH_SIZE},
  "train_micro_batch_size_per_gpu": ${MICRO_BATCH_SIZE},
  "gradient_accumulation_steps": 1,
  "fp16": { "enabled": true },
  "zero_optimization": { "stage": 0 }
}
EOT

# =========================
# Run Training
# =========================
deepspeed escho/minimal_moe_test.py \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --num-attention-heads ${NUM_ATTN_HEADS} \
  --seq-length ${SEQ_LEN} \
  --max-position-embeddings ${SEQ_LEN} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --train-tokens ${TRAIN_TOKENS} \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-warmup-tokens ${WARMUP_TOKENS} \
  --lr-decay-tokens ${LR_DECAY_TOKENS} \
  --lr-decay-style cosine \
  --num-experts ${NUM_EXPERTS} \
  --moe-loss-coeff ${MOE_LOSS_COEFF} \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --no-pipeline-parallel \
  --vocab-file ${VOCAB_PATH} \
  --merge-file ${MERGE_PATH} \
  --data-path ${DATA_PATH} \
  --data-impl mmap \
  --save ${CHECKPOINT_PATH} \
  --load ${CHECKPOINT_PATH} \
  --tensorboard-dir ${TENSORBOARD_DIR} \
  --fp16 \
  --deepspeed \
  --deepspeed_config ds_minimal_moe_test.json \
  &> ${LOG_FILE}
