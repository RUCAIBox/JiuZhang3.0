set -ex

export CXX=g++
export OMP_NUM_THREADS=20
export TRANSFORMERS_OFFLINE=1

NNODES=1

min_port=1024
max_port=65535
range=$((max_port - min_port + 1))
random_port=$((RANDOM % range + min_port))
NUM_GPUS=4
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=32
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU/$NNODES))

PT_PATH="deepseek-ai/deepseek-math-7b-rl"

nlr=1e-5
epoch=5

size=2000
INST_TYPE="pure_no_inst"
PROMPT_FORMAT="nothing"
DATA_NAME=""
DATA_PATH=""/$DATA_NAME
MODEL_NAME=deepseek-math-rl-sft-$DATA_NAME
CKPT_PATH=""/$MODEL_NAME-$nlr-cosine


torchrun --nproc_per_node=8 \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$random_port \
    train_synthesis.py \
    --model_name_or_path $PT_PATH \
    --data_path $DATA_PATH \
    --output_dir $CKPT_PATH \
    --bf16 True \
    --num_train_epochs $epoch \
    --model_max_length 2048 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate $nlr \
    --weight_decay 0.1 \
    --warmup_ratio 0.0 \
    --logging_steps 2 \
    --deepspeed ./configs/stage_2.json \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to none \
    --lr_scheduler_type "cosine" \
    --flash_attention \
    --inst_type $INST_TYPE \
    --prompt_format $PROMPT_FORMAT \
    --overwrite_output_dir

