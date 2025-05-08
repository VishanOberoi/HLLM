import torch
import os
import json
import sys

def print_section(title):
    print(f"\n{title}")
    print("-" * 50)

# 1. Check the DeepSpeed checkpoint
ckpt_dir = "checkpoints/ml1m_run/HLLM-0.pth"
print(f"Checking DeepSpeed checkpoint in: {ckpt_dir}")

# Add checkpoint directory to path to import zero_to_fp32
sys.path.append(ckpt_dir)
from zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

print_section("Checkpoint Structure")
print("Model state files:")
for f in os.listdir(os.path.join(ckpt_dir, "checkpoint")):
    if "model_states" in f:
        print(f"- {f}")

# Convert checkpoint to fp32 and load it
print("\nConverting checkpoint to fp32...")
try:
    state_dict = get_fp32_state_dict_from_zero_checkpoint(os.path.join(ckpt_dir, "checkpoint"))
    print("\nCheckpoint loaded successfully!")
    
    # Print model architecture
    print("\nModel architecture:")
    for key in state_dict.keys():
        if isinstance(state_dict[key], torch.Tensor):
            print(f"{key}: {state_dict[key].shape}")
except Exception as e:
    print(f"Error loading checkpoint: {str(e)}")

# # 2. Check training logs
# log_path = "checkpoints/ml1m_run/HLLM/Apr-24-2025_00-18-51.log"
# print_section("Training Log Analysis")
# print(f"Reading log from: {log_path}")

# with open(log_path, 'r') as f:
#     lines = f.readlines()

# # Look for validation scores and data split info
# validation_lines = []
# data_split_lines = []
# for line in lines:
#     if "valid_score" in line or "best_valid_score" in line:
#         validation_lines.append(line.strip())
#     if "train size" in line.lower() or "valid size" in line.lower() or "test size" in line.lower():
#         data_split_lines.append(line.strip())

# print("\nValidation Scores from Log:")
# for line in validation_lines:
#     print(line)

# print("\nData Split Information from Log:")
# for line in data_split_lines:
#     print(line)

# # 3. Check evaluation configuration
# print_section("Evaluation Configuration")
# eval_config_path = "code/config/eval_ml1m.yaml"
# if os.path.exists(eval_config_path):
#     with open(eval_config_path, 'r') as f:
#         eval_config = f.read()
#     print("Evaluation config:")
#     print(eval_config)
# else:
#     print(f"Warning: Evaluation config not found at {eval_config_path}")

# # 4. Check wandb summary for comparison
# wandb_summary_path = "code/wandb/run-20250424_001915-kfgr5a2b/files/wandb-summary.json"
# if os.path.exists(wandb_summary_path):
#     print_section("Wandb Summary")
#     with open(wandb_summary_path, 'r') as f:
#         wandb_summary = json.load(f)
#     print("Best metrics from wandb:")
#     for key in wandb_summary:
#         if 'best' in key.lower() or 'valid' in key.lower():
#             print(f"{key}: {wandb_summary[key]}")
# else:
#     print(f"Warning: Wandb summary not found at {wandb_summary_path}") 