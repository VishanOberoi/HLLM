#!/bin/bash
#SBATCH --job-name=HLLM-encode
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12

#SBATCH --partition=general       
#SBATCH --mem=256G 
#SBATCH --gres=gpu:A6000:1

#SBATCH --time=48:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user="jingyuah@cs.cmu.edu"

echo "Job Starts"

eval "$(conda shell.bash hook)"
conda activate hllm

echo "activated"

user_pretrain_dir="/data/user_data/jingyuah/HLLM_weights/checkpoints/TinyLlama-1.1B-Chat-v0.4"
item_pretrain_dir="/data/user_data/jingyuah/HLLM_weights/checkpoints/TinyLlama-1.1B-Chat-v0.4"

# checkpoint_dir="/data/user_data/jingyuah/HLLM_weights/checkpoints/repro_tiny_llama_1.1b_pixelrec_200K"
checkpoint_dir="/data/group_data/cx_group/REC/checkpoints/HLLM-ml-1m"

inter_path="/data/user_data/jingyuah/HLLM_weights/data/dataset"
info_path="/data/user_data/jingyuah/HLLM_weights/data/information"

epoch=5

num_shards=64
shard=$1
item_batch_size=160
num_cpus=12
save_step=200

echo $shard 

# Clueweb id map path
id_map_path="/data/group_data/cx_group/REC/ClueWeb-Reco/ClueWeb-Reco_public/cwid_to_id.tsv"
embedding_output_dir="/data/group_data/cx_group/REC/ClueWeb-Reco/HLLM_exps/HLLM_ml-1m/raw"
# embedding_output_dir="/data/user_data/jingyuah/HLLM/" # item_encode_b1.pkl"
# embedding_output_dir="/data/user_data/jingyuah/HLLM/tiny_confirm"


export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))


# Item and User LLM are initialized by specific pretrain_dir.
python3 /home/jingyuah/HLLM/code/seq_encode.py \
    --config_file /home/jingyuah/HLLM/code/overall/LLM_deepspeed.yaml /home/jingyuah/HLLM/code/HLLM/HLLM.yaml \
    --loss nce \
    --epochs $epoch \
    --train_batch_size 16 \
    --MAX_TEXT_LENGTH 256 \
    --MAX_ITEM_LIST_LENGTH 10 \
    --item_size 160 \
    --seq_size 128 \
    --checkpoint_dir $checkpoint_dir \
    --optim_args.learning_rate 1e-4 \
    --item_pretrain_dir $item_pretrain_dir \
    --user_pretrain_dir $user_pretrain_dir \
    --text_path $info_path \
    --data_path $inter_path \
    --text_keys '[\"title\",\"description\"]'  \
    --best_model_path $item_pretrain_dir \
    --id_map_path $id_map_path \
    --output_path $embedding_output_dir/clueweb-b-en.${shard}-of-${num_shards}.pkl \
    --batch_size $item_batch_size \
    --dataset_number_of_shards $num_shards \
    --dataset_shard_index $shard \
    --save_step $save_step \
    --num_workers $num_cpus \
    --item_encoding 


# (Pdb) items[:5, :5]
# tensor([[-0.0713,  0.0400,  0.1416, -0.0610,  0.1167],
#         [-0.2148,  0.5000, -1.1328,  0.9727,  0.6250],
#         [-1.1953,  0.5625,  0.9336, -1.3125, -0.4980],
#         [-0.7461,  0.1387, -0.3359,  0.6914, -0.2910],
#         [ 0.5000, -0.5820, -0.7734,  2.4219, -0.3613]], device='cuda:0')

# tensor([[-0.0713,  0.0400,  0.1416, -0.0610,  0.1167],
#         [-0.2148,  0.5000, -1.1328,  0.9727,  0.6250],
#         [-1.1953,  0.5625,  0.9336, -1.3125, -0.4980],
#         [-0.7461,  0.1387, -0.3359,  0.6914, -0.2910],
#         [ 0.5000, -0.5820, -0.7734,  2.4219, -0.3613]], device='cuda:0')

# after all embeddings are computed, merge into binary 
python /home/jingyuah/HLLM/code/clever_merge.py 


echo "Job Ends"
