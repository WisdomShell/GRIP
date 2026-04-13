#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_DIR="/path/to/your/Meta-Llama-3-8B-Instruct/"
INPUT_FILE="/path/to/your/merged_train_set.jsonl"
BASE_OUTPUT_DIR="/path/to/your/GRIP/GRIP/data" #temp dataset
ES_HOST="localhost"
ES_INDEX="wiki"
BATCH_SIZE=64


torchrun --nproc_per_node=8 --master_port=29500 data_generation/make_first_steps.py \
    --model_dir "${MODEL_DIR}" \
    --input_file "${INPUT_FILE}" \
    --output_dir "${BASE_OUTPUT_DIR}" \
    --es_host "${ES_HOST}" \
    --es_index "${ES_INDEX}" \
    --batch_size ${BATCH_SIZE} \
    --target 11250 \


echo "finish！"