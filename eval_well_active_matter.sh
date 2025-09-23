#!/bin/bash

# Simple wrapper using the original inference.py script
# Usage: ./eval_well_active_matter_simple.sh <model_checkpoint_path>

# Configuration
DATA_PATH="/data0/arshkon/data/the_well/datasets/active_matter"
BATCH_SIZE=16
AR_STEPS=5
OUTPUT_FILE="active_matter_eval_$(date +%Y%m%d_%H%M%S).csv"

# Check if model path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_checkpoint_path>"
    echo ""
    echo "Example:"
    echo "  $0 /data0/arshkon/checkpoints/poseidon/well-active-matter-20250923-123456"
    echo "  $0 camlab-ethz/Poseidon-B  # for pretrained model"
    echo ""
    exit 1
fi

MODEL_PATH="$1"

echo "=========================================="
echo "Well Active Matter Model Evaluation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH" 
echo "Output: $OUTPUT_FILE"
echo "Batch Size: $BATCH_SIZE"
echo "AR Steps: $AR_STEPS"
echo ""

# Use the original inference script directly with single GPU
CUDA_VISIBLE_DEVICES=0 python -m scOT.inference \
    --mode eval \
    --model_path "$MODEL_PATH" \
    --dataset well.WellActiveMatter \
    --data_path "$DATA_PATH" \
    --file "$OUTPUT_FILE" \
    --ckpt_dir "/data0/arshkon/checkpoints/poseidon" \
    --batch_size $BATCH_SIZE \
    --ar_steps $AR_STEPS

echo ""
echo "Evaluation completed! Results saved to: $OUTPUT_FILE"
