set +e
set -x

export CXX=g++
export OMP_NUM_THREADS=20
export TRANSFORMERS_OFFLINE=1

min_port=1024
max_port=65535
range=$((max_port - min_port + 1))
random_port=$((RANDOM % range + min_port))

NNODES=1
NUM_GPUS=1
BATCH_SIZE_PER_GPU=8
TOTAL_BATCH_SIZE=512
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU/$NNODES))


epoch=1
lr=1e-5
wsd_ratio=0.85

PT_PATH=$1
DATA_PATH=$2
OUTPUT_PATH=$3
DS_STAGE=$4
if [ $DS_STAGE="2" ]; then
    DS_PATH=./configs/stage_2.json
else
    DS_PATH=./configs/stage_2.json
fi

torchrun --nproc_per_node=8 \
    train_pack.py \
    --model_name_or_path $PT_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --bf16 True \
    --num_train_epochs $epoch \
    --model_max_length 2048 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 2 \
    --deepspeed $DS_PATH \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to none \
    --lr_scheduler_type "linear" \
    --flash_attention \
    --preprocessing_num_workers 96 \
    --save_on_each_node False \
    --use_wsd \
    --no_shuffle \
    --stable_ratio $wsd_ratio
