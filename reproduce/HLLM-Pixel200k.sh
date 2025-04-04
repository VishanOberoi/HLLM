#!/bin/bash
# Vishan's HLLM run on Pixel200K (1 GPU on Babel)

cd code

python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
--MAX_ITEM_LIST_LENGTH 10 \
--epochs 5 \
--optim_args.learning_rate 1e-4 \
--checkpoint_dir checkpoints/pixel200k-run1 \
--loss nce \
--MAX_TEXT_LENGTH 256 \
--dataset Pixel200K \
--text_path /full/path/to/information/Pixel200K.csv \
--item_pretrain_dir /full/path/to/tinyllama \
--user_pretrain_dir /full/path/to/tinyllama \
--train_batch_size 8 \
--gradient_checkpointing True
