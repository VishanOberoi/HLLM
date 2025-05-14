#!/bin/bash
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate

# 1B: 32 A100s for ≈ 4.1days
cd code && python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
--MAX_ITEM_LIST_LENGTH 10 --epochs 5 --optim_args.learning_rate 1e-4 \
--checkpoint_dir saved_dir \
--loss nce --MAX_TEXT_LENGTH 256 --dataset Pixel8M \
--text_path text_path \
--item_pretrain_dir item_pretrain_dir \
--user_pretrain_dir user_pretrain_dir \
--train_batch_size 16


user_pretrain_dir="/data/user_data/jingyuah/HLLM_weights/checkpoints/TinyLlama-1.1B-intermediate-step-1431k-3T"
item_pretrain_dir="/data/user_data/jingyuah/HLLM_weights/checkpoints/TinyLlama-1.1B-intermediate-step-1431k-3T"

checkpoint_dir="/data/group_data/cx_group/REC/checkpoints/HLLM-ml-1m"

inter_path="/home/jingyuah/REC_source/HLLM/dataset"
info_path="/home/jingyuah/REC_source/HLLM/information"

epoch=5


# Item and User LLM are initialized by specific pretrain_dir.
cd /home/jingyuah/REC_source/HLLM/code
python3 main.py \
    --config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
    --loss nce \
    --epochs $epoch \
    --dataset ml1m \
    --train_batch_size 16 \
    --MAX_TEXT_LENGTH 256 \
    --MAX_ITEM_LIST_LENGTH 10 \
    --checkpoint_dir $checkpoint_dir \
    --optim_args.learning_rate 1e-4 \
    --item_pretrain_dir $item_pretrain_dir \
    --user_pretrain_dir $user_pretrain_dir \
    --text_path $info_path \
    --data_path $inter_path \
    --text_keys '[\"movie_title\",\"genre\"]'



python3 main.py \
  --config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
  --loss nce \
  --epochs 5 \
  --dataset ml1m \
  --train_batch_size 16 \
  --MAX_TEXT_LENGTH 256 \
  --MAX_ITEM_LIST_LENGTH 10 \
  --checkpoint_dir $checkpoint_dir \
  --optim_args.learning_rate 1e-4 \
  --item_pretrain_dir $item_pretrain_dir \
  --user_pretrain_dir $item_pretrain_dir \
  --text_path /home/jingyuah/REC_source/HLLM/information \
  --gradient_checkpointing True \
  --stage 3
