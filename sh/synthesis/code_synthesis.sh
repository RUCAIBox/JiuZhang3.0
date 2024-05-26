#!/bin/sh

set -ex

DATA_PATH=""
MODEL_PATH=""
SAVE_PATH=""


NUM_GPUS=1

python -m synthesis_code.generate \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --save_path $SAVE_PATH
