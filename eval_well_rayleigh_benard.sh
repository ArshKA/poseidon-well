#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

export HF_HOME="/data0/arshkon/checkpoints/poseidon"
export HF_HUB_CACHE="/data0/arshkon/checkpoints/poseidon"

# Set the checkpoint path - modify this to point to your trained model
CHECKPOINT_PATH="/data0/arshkon/checkpoints/poseidon/well-rayleigh-benard-latest"
DATA_PATH="/data0/arshkon/data/the_well/datasets/rayleigh_benard"
OUTPUT_FILE="rayleigh_benard_eval_$(date +%Y%m%d_%H%M%S).json"

echo "Starting Poseidon evaluation on Well Rayleigh Benard dataset..."
echo "Checkpoint Path: $CHECKPOINT_PATH"
echo "Data Path: $DATA_PATH"
echo "Output File: $OUTPUT_FILE"
echo ""

python scOT/inference.py \
    --dataset "well.WellRayleighBenard" \
    --data_path "$DATA_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --output_file "$OUTPUT_FILE" \
    --split "test" \
    --num_samples 100 \
    --max_num_time_steps 10 \
    --time_step_size 1

echo "Evaluation completed! Results saved to $OUTPUT_FILE"
