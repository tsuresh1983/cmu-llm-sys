#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ZERO_STAGE=$1
OUTPUT=./output_llama2_7b_lora
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT


deepspeed --num_gpus=2 main.py \
   --data_split 2,4,4 \
   --model_name_or_path meta-llama/Llama-2-7b-hf \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --dtype bf16 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --offload \
   --lora_dim 8 \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
