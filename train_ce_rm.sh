#!/bin/bash

export NCCL_P2P_DISABLE=1

export HF_TOKEN="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"
export HF_HOME="/data0/arshkon/checkpoints/poseidon"
export HF_HUB_CACHE="/data0/arshkon/checkpoints/poseidon"

export WANDB_PROJECT="poseidon-ce-rm"
export WANDB_RUN_NAME="ce-rm-base-$(date +%Y%m%d-%H%M%S)"
export WANDB_CACHE_DIR="/data0/arshkon/wandb_cache"
export WANDB_DIR="/data0/arshkon/wandb"

# Configuration
CONFIG_FILE="/home/arshkon/Projects/poseidon/configs/ce_rm_training.yaml"
DATA_PATH="/data0/arshkon/data/poseidon/CE-RM"
CHECKPOINT_PATH="/data0/arshkon/checkpoints/poseidon"
PRETRAINED_MODEL="camlab-ethz/Poseidon-B"  # Use Base model for finetuning

# Create checkpoint directory if it doesn't exist
mkdir -p "$CHECKPOINT_PATH"


echo "Starting Poseidon training on CE-RM dataset..."
echo "Config: $CONFIG_FILE"
echo "Data Path: $DATA_PATH"
echo "Checkpoint Path: $CHECKPOINT_PATH"
echo "Pretrained Model: $PRETRAINED_MODEL"
echo ""

# Run the training command
accelerate launch scOT/train.py \
    --config "$CONFIG_FILE" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --wandb_project_name "$WANDB_PROJECT" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --data_path "$DATA_PATH" \
    --finetune_from "$PRETRAINED_MODEL" \
    --replace_embedding_recovery

echo "Training completed!"
