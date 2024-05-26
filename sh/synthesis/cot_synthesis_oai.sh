#!/bin/sh

set -ex


DATA_NAME=""
DATA_PATH=""/$DATA_NAME
API_KEY_PATH=""

MODEL=gpt-4o
SAVE_PATH=""/$DATA_NAME-synthetic_data

python -m synthesis_cot.generate_oai \
    --model $MODEL \
    --prompts_dataset $DATA_PATH \
    --prompt_column prompt \
    --checkpoint_path $SAVE_PATH \
    --checkpoint_interval 5 \
    --max_samples "-1" \
    --start_sample 0 \
    --end_sample 2000 \
    --api_key_path $API_KEY_PATH
