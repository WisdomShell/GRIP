#!/bin/bash

set -e
# Configuration
BACKEND="fsdp"
LOCAL_DIR="/path/to/your/RL_model/global_step_xxx/actor"
HF_MODEL_PATH="/path/to/your/global_step_xxx/" #SFT_Model
TARGET_DIR="/path/to/your/RL_model/step_xxx"
HF_UPLOAD_PATH=""

TIE_WORD_EMBEDDING=false
IS_VALUE_MODEL=false

# Run conversion
echo "Starting conversion from DAPO checkpoint to HF format..."

python model_merger.py \
    --backend "$BACKEND" \
    --local_dir "$LOCAL_DIR" \
    --hf_model_path "$HF_MODEL_PATH" \
    --target_dir "$TARGET_DIR" \
    $( [ "$TIE_WORD_EMBEDDING" = true ] && echo "--tie-word-embedding" ) \
    $( [ "$IS_VALUE_MODEL" = true ] && echo "--is-value-model" )

echo "Conversion completed! HF model saved to: $TARGET_DIR"