#!/bin/bash

# Poseidon Training Script for CE-RM (Richtmyer-Meshkov) Dataset - Training from Scratch
# Generated based on the assembled data and project structure

# Set environment variables
export HF_HOME="/tmp/hf_cache"  # Adjust if needed
export WANDB_PROJECT="poseidon-ce-rm-scratch"
export WANDB_RUN_NAME="ce-rm-scratch-$(date +%Y%m%d-%H%M%S)"

# Configuration
CONFIG_FILE="/home/arshkon/Projects/poseidon/configs/ce_rm_training.yaml"
DATA_PATH="/data0/arshkon/data/poseidon/CE-RM"
CHECKPOINT_PATH="/home/arshkon/Projects/poseidon/checkpoints"

# Create checkpoint directory if it doesn't exist
mkdir -p "$CHECKPOINT_PATH"

# Ensure we're in the right directory
cd /home/arshkon/Projects/poseidon

# Check if data file exists
if [ ! -f "$DATA_PATH/CE-RM.nc" ]; then
    echo "Error: Data file $DATA_PATH/CE-RM.nc not found!"
    echo "Make sure you have assembled the data using assemble_data.py"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    exit 1
fi

echo "Starting Poseidon training from scratch on CE-RM dataset..."
echo "Config: $CONFIG_FILE"
echo "Data Path: $DATA_PATH"
echo "Checkpoint Path: $CHECKPOINT_PATH"
echo ""

# Run the training command (without pretrained model)
accelerate launch scOT/train.py \
    --config "$CONFIG_FILE" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --wandb_project_name "$WANDB_PROJECT" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --data_path "$DATA_PATH"

echo "Training completed!"
