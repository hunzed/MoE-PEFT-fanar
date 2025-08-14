#!/bin/bash

# Arabic Fine-tuning Script for Fanar Model with MixLoRA
set -e

BASE_MODEL="QCRI/Fanar-1-9B-Instruct"
CONFIG_FILE="templates/arabic_fanar_mixlora.json"
OUTPUT_DIR="./outputs/arabic_fanar_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="./logs/arabic_fanar_training_$(date +%Y%m%d_%H%M%S).log"

mkdir -p outputs logs

echo "Starting Arabic Fine-tuning with MixLoRA on Fanar"
echo "Model: $BASE_MODEL"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"

python moe_peft.py \
    --base_model "$BASE_MODEL" \
    --config "$CONFIG_FILE" \
    --dir "$OUTPUT_DIR" \
    --log_file "$LOG_FILE" \
    --bf16 \
    --tf32 \
    --verbose \
    "$@"

echo "Training completed. Results saved to: $OUTPUT_DIR"
