# HLLM Training Pipeline (PixelRec Dataset)

This repo contains training code for the HLLM (Hybrid Large Language Model) on the PixelRec dataset. The training has been successfully reproduced with 8 A6000 GPUs using DeepSpeed for distributed training.

## ✅ Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd HLLM
```

### 2. Create and activate a Python virtual environment
```bash
python3 -m venv ~/hllm_pixel_venv
source ~/hllm_pixel_venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
If needed, manually install compatible versions:
```bash
pip install torch==2.1.0 torchvision torchaudio
pip install deepspeed==0.12.6
pip install flash-attn xformers wandb
pip install flash-attn --no-build-isolation
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn==2.5.9.post1
```

### 4. Dataset & Pretrained Checkpoints
- Place the `Pixel200K.csv` or text files in the following paths:
  - `dataset/`
  - `information/`
- Place pretrained TinyLlama model in:
  ```
  ~/scratch/llms/tinyllama
  ```

## 📊 WandB Logging
To enable Weights & Biases tracking, set your API key in the SLURM script:
```bash
export WANDB_API_KEY=your_key
```

## 🚀 Submitting a Training Job (SLURM)
Use the following SLURM script to launch training on 8 A6000 GPUs:

```bash
#!/bin/bash
#SBATCH --job-name=hllm_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:A6000:8
#SBATCH --mem-per-gpu=64G
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --output=hllm_train_%j.out
#SBATCH --error=hllm_train_%j.err

source ~/hllm_pixel_venv/bin/activate
cd /home/voberoi/HLLM/code
export WANDB_API_KEY=your_key

python3 main.py \
  --config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
  --loss nce \
  --epochs 5 \
  --dataset Pixel200K \
  --train_batch_size 16 \
  --MAX_TEXT_LENGTH 256 \
  --MAX_ITEM_LIST_LENGTH 10 \
  --checkpoint_dir ../checkpoints/pixel200k_run \
  --optim_args.learning_rate 1e-4 \
  --item_pretrain_dir ~/scratch/llms/tinyllama \
  --user_pretrain_dir ~/scratch/llms/tinyllama \
  --text_path /home/voberoi/HLLM/information \
  --gradient_checkpointing True \
  --stage 3
```

## 🛠 Notes
- `--stage 3` enables long-context memory-efficient mode
- Logging and checkpoints are enabled by default
- To resume from a checkpoint, specify `--auto_resume True` and update `checkpoint_dir`

## 🧠 Next Steps
- Complete full 5-epoch training
- Begin evaluation and leaderboard analysis
- Integrate with benchmarking pipeline

