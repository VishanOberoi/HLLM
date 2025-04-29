#!/bin/bash
#SBATCH --job-name=hllm_ml1m_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem-per-gpu=64G
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --output=hllm_ml1m_train_%j.out
#SBATCH --error=hllm_ml1m_train_%j.err

# Activate your virtual environment
source ~/hllm_pixel_venv/bin/activate

# Navigate to the HLLM code directory
cd /home/voberoi/HLLM/code
export WANDB_API_KEY=575ba3bae4ccbe9a3a558560f69a9295bb1626c4

# Run the main training script
python3 main.py \
  --config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
  --loss nce \
  --epochs 5 \
  --dataset ml1m \
  --train_batch_size 16 \
  --MAX_TEXT_LENGTH 256 \
  --MAX_ITEM_LIST_LENGTH 10 \
  --checkpoint_dir ../checkpoints/ml1m_run \
  --optim_args.learning_rate 1e-4 \
  --item_pretrain_dir ~/scratch/llms/tinyllama \
  --user_pretrain_dir ~/scratch/llms/tinyllama \
  --text_path /home/voberoi/HLLM/information \
  --gradient_checkpointing True \
  --stage 3
