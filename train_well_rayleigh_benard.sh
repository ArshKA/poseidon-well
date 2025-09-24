#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NCCL_P2P_DISABLE=1

export HF_HOME="/data0/arshkon/checkpoints/poseidon"
export HF_HUB_CACHE="/data0/arshkon/checkpoints/poseidon"

export WANDB_PROJECT="poseidon-well-rayleigh-benard"
export WANDB_RUN_NAME="well-rayleigh-benard-$(date +%Y%m%d-%H%M%S)"
export WANDB_CACHE_DIR="/data0/arshkon/wandb_cache"
export WANDB_DIR="/data0/arshkon/wandb"

CONFIG_FILE="/home/arshkon/Projects/poseidon/configs/well_rayleigh_benard_training.yaml"
DATA_PATH="/data0/arshkon/data/the_well/datasets/rayleigh_benard"
CHECKPOINT_PATH="/data0/arshkon/checkpoints/poseidon"
PRETRAINED_MODEL="camlab-ethz/Poseidon-B"

mkdir -p "$CHECKPOINT_PATH"

echo "Starting Poseidon training on Well Rayleigh Benard dataset..."
echo "Data Path: $DATA_PATH"
echo "Checkpoint Path: $CHECKPOINT_PATH"
echo "Pretrained Model: $PRETRAINED_MODEL"
echo ""

accelerate launch --num_processes=8 scOT/train.py \
    --config "$CONFIG_FILE" \
    --data_path "$DATA_PATH" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --wandb_project_name "$WANDB_PROJECT" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --finetune_from "$PRETRAINED_MODEL" \
    --replace_embedding_recovery

echo "Training completed!"
