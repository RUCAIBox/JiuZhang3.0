#!/bin/sh

set -ex

PROMPT_COLUMN="prompt_mix"
DATA_PATH=$1
SAVE_PATH=$2
MODEL_PATH="ToheartZhang/JiuZhang3.0-Synthesis-7B"

mkdir -p $SAVE_PATH

NUM_GPUS=1

python -m synthesis_cot.generate \
    --model $MODEL_PATH \
    --prompts_dataset $DATA_PATH \
    --prompt_column $PROMPT_COLUMN \
    --checkpoint_path $SAVE_PATH \
    --checkpoint_interval 8000 \
    --num_gpus $NUM_GPUS \
    --max_samples "-1" \
    --temperature 1.0 \
