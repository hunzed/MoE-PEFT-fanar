#!/bin/bash

# Arabic Fine-tuning Script for Fanar Model with MixLoRA and Wandb Integration
#
# Usage:
#   Basic training: ./train_arabic_fanar.sh
#   
#   With wandb logging:
#   WANDB_PROJECT="my-project" WANDB_NAME="my-experiment" ./train_arabic_fanar.sh
#   
#   With custom parameters:
#   ./train_arabic_fanar.sh --epochs 5 --batch_size 16
#
set -e

# Default values
BASE_MODEL="fanar"
ADAPTER_NAME="arabic_fanar_mixlora_$(date +%Y%m%d_%H%M%S)"
ROUTING_STRATEGY="mixlora"
EPOCHS=1
BATCH_SIZE=8
MICRO_BATCH_SIZE=4
LEARNING_RATE=2e-4
NUM_EXPERTS=8
TOP_K=2
SAVE_STEP=500
EVALUATE_STEPS=2000

# Wandb settings (set these to enable wandb logging)
WANDB_PROJECT="${WANDB_PROJECT:-arabic-fanar-experiments}"
WANDB_NAME="${WANDB_NAME:-$ADAPTER_NAME}"
WANDB_TAGS="${WANDB_TAGS:-arabic fanar mixlora culture-classification}"

echo "=== Arabic Fine-tuning with MixLoRA on Fanar ==="
echo "Adapter Name: $ADAPTER_NAME"
echo "Routing Strategy: $ROUTING_STRATEGY"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Experts: $NUM_EXPERTS"
echo "Top-K: $TOP_K"
echo ""

# Build the command
CMD="python arabic_finetune.py \
    --name $ADAPTER_NAME \
    --model $BASE_MODEL \
    --routing $ROUTING_STRATEGY \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --save_step $SAVE_STEP \
    --evaluate_steps $EVALUATE_STEPS"

# Add wandb parameters if they are set
if [ ! -z "$WANDB_PROJECT" ]; then
    CMD="$CMD --wandb_project $WANDB_PROJECT"
    echo "Wandb Project: $WANDB_PROJECT"
fi

if [ ! -z "$WANDB_NAME" ]; then
    CMD="$CMD --wandb_name $WANDB_NAME"
    echo "Wandb Run Name: $WANDB_NAME"
fi

if [ ! -z "$WANDB_TAGS" ]; then
    CMD="$CMD --wandb_tags $WANDB_TAGS"
    echo "Wandb Tags: $WANDB_TAGS"
fi

# Add any additional arguments passed to this script
CMD="$CMD $@"

echo "Running command:"
echo "$CMD"

# Execute the training
eval $CMD

echo ""
echo "Training completed"
if [ ! -z "$WANDB_PROJECT" ]; then
    echo "Check your wandb dashboard: https://wandb.ai/$WANDB_PROJECT"
fi
