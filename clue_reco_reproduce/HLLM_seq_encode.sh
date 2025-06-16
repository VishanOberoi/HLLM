#!/bin/bash
#SBATCH --job-name=HLLM-seq_features
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --partition=general       
#SBATCH --mem=512G 
#SBATCH --gres=gpu:0

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

num_cpus=4


# Clueweb id map path
# embedding_output_dir="/data/user_data/jingyuah/HLLM" 
embedding_output_dir="/data/user_data/jingyuah/HLLM/tiny_confirm"
item_embed_path="${embedding_output_dir}/item_embed_full_streamed.bin"
seq_data_path="/data/group_data/cx_group/REC/ClueWeb-Reco/ClueWeb-Reco_public/ordered_id_splits/test_input.tsv"
feature_output_path="${embedding_output_dir}/seq_features.pkl"
seq_embed_output_path="${embedding_output_dir}/seq_embed.bin"

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
    --batch_size 160 \
    --checkpoint_dir $checkpoint_dir \
    --optim_args.learning_rate 1e-4 \
    --item_pretrain_dir $item_pretrain_dir \
    --user_pretrain_dir $user_pretrain_dir \
    --text_path $info_path \
    --data_path $inter_path \
    --text_keys '[\"title\",\"tag\",\"description\"]'  \
    --best_model_path $item_pretrain_dir \
    --output_path $embedding_output_dir/clueweb-b-en.${shard}-of-${num_shards}.pkl \
    --item_embed_path $item_embed_path \
    --feature_output_path $feature_output_path \
    --seq_data_path $seq_data_path \
    --num_workers $num_cpus \
    --compute_seq_item_feature 

# first feature -> matched 
# tensor([[-1.4297,  1.1797,  1.0938,  0.6328, -0.1309],
#         [ 0.2031,  0.2676, -0.8945,  1.9141,  0.0265],
#         [-0.4941,  1.0859,  0.5391,  0.5625,  0.2139],
#         [-0.5781, -0.7188,  1.7812, -1.0703, -0.7891],
#         [ 0.1484,  0.7148,  1.8125,  0.8633, -0.3555]], device='cuda:0',
#        dtype=torch.bfloat16)


python3 /home/jingyuah/HLLM/code/seq_encode.py \
    --config_file /home/jingyuah/HLLM/code/overall/LLM_deepspeed.yaml /home/jingyuah/HLLM/code/HLLM/HLLM.yaml \
    --loss nce \
    --epochs $epoch \
    --train_batch_size 16 \
    --MAX_TEXT_LENGTH 256 \
    --MAX_ITEM_LIST_LENGTH 10 \
    --batch_size 128 \
    --checkpoint_dir $checkpoint_dir \
    --optim_args.learning_rate 1e-4 \
    --item_pretrain_dir $item_pretrain_dir \
    --user_pretrain_dir $user_pretrain_dir \
    --text_path $info_path \
    --data_path $inter_path \
    --text_keys '[\"title\",\"tag\",\"description\"]'  \
    --best_model_path $item_pretrain_dir \
    --output_path $embedding_output_dir/clueweb-b-en.${shard}-of-${num_shards}.pkl \
    --feature_output_path $feature_output_path \
    --num_workers $num_cpus \
    --seq_embed_output_path $seq_embed_output_path \
    --seq_encoding 

# computed ml-1m first batch embeddings: matched; only bf16 and fp32 recast issue 
# >>> sample_seq_embed[:5, :5]
# tensor([[-0.1016,  1.0469,  0.6836, -0.8164, -0.4121],
#         [-0.3242, -0.0815, -0.8516,  0.7852, -1.0703],
#         [-0.7773, -0.1602, -0.4746,  0.2676,  0.7148],
#         [ 2.3750, -0.7500, -0.8164, -0.2314,  0.9492],
#         [ 1.3672,  0.6094, -1.6797,  0.6914,  0.0977]], device='cuda:0',
#        dtype=torch.bfloat16)

# array([[-0.1015625 ,  1.046875  ,  0.68359375, -0.81640625, -0.41210938],
#        [-0.32421875, -0.08154297, -0.8515625 ,  0.78515625, -1.0703125 ],
#        [-0.77734375, -0.16015625, -0.47460938,  0.26757812,  0.71484375],
#        [ 2.375     , -0.75      , -0.81640625, -0.23144531,  0.94921875],
#        [ 1.3671875 ,  0.609375  , -1.6796875 ,  0.69140625,  0.09765625]],
#       dtype=float32)