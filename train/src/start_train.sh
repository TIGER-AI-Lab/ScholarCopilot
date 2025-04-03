#!/bin/bash

export GPUS_PER_NODE=8
export NNODES=4
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
export MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-60000}}
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

# conda activate scholar_copilot

output_dir="../model_output/"

torchrun --nproc_per_node $GPUS_PER_NODE \
 --master_addr $MASTER_ADDR \
 --node_rank $NODE_RANK \
 --master_port $MASTER_PORT \
 --nnodes $NNODES \
 train.py \
 --deepspeed ds_zero3_config.json \
 --output_dir ${output_dir} \
 --model_name_or_path  \
 --save_steps 200 \
 --dataset_name json \
 --dataset_path ../../train_data/scholar_copilot_train_data_500k.json \
 --bf16 \
 --normalize \
 --temperature 0.01 \
 --per_device_train_batch_size 1 \
 --gradient_checkpointing \
 --learning_rate 1e-5 \
 --query_max_len 16384 \
 --passage_max_len 16384 \
 --num_train_epochs 1 \
 --logging_steps 1 \
 --overwrite_output_dir \
 --gradient_accumulation_steps 1

