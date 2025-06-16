#!/bin/bash
#SBATCH --job-name=cos_prediction
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4

#SBATCH --partition=general       
#SBATCH --mem=1024G 
#SBATCH --time=48:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user="jingyuah@cs.cmu.edu"

echo "Job Starts"

eval "$(conda shell.bash hook)"
conda activate hllm


# similarity_pred: use DiskANN (https://github.com/microsoft/DiskANN) to compute embedding KNNS 
# base_dir="/data/group_data/cx_group/REC/ClueWeb-Reco/HLLM_exps/HLLM_ml-1m"
# base_dir="/data/user_data/jingyuah/HLLM"
base_dir="/data/user_data/jingyuah/HLLM/tiny_confirm"
embed_file="${base_dir}/item_embed_full_streamed.bin"
query_file="${base_dir}/seq_embed.bin"
predict_file="${base_dir}/prediction.bin"
K=1000

python /home/jingyuah/HLLM/code/custom_eval.py \
        --seq_emb_path $query_file \
        --item_emb_path $embed_file \
        --k $K \
        --output_binary_path $predict_file

# (Pdb) seq_output[:5, :5]
# tensor([[-0.0023,  0.0234,  0.0153, -0.0182, -0.0092],
#         [-0.0069, -0.0017, -0.0181,  0.0167, -0.0228],
#         [-0.0170, -0.0035, -0.0104,  0.0059,  0.0156],
#         [ 0.0530, -0.0167, -0.0182, -0.0052,  0.0212],
#         [ 0.0304,  0.0135, -0.0374,  0.0154,  0.0022]], device='cuda:0',
#        dtype=torch.bfloat16)

# tensor([[-0.0023,  0.0234,  0.0153, -0.0182, -0.0092],
#         [-0.0069, -0.0017, -0.0181,  0.0167, -0.0228],
#         [-0.0170, -0.0035, -0.0104,  0.0059,  0.0156],
#         [ 0.0530, -0.0167, -0.0182, -0.0052,  0.0212],
#         [ 0.0304,  0.0135, -0.0374,  0.0154,  0.0022]], dtype=torch.bfloat16)


# item_feature[:5, :5]
# tensor([[-0.0022,  0.0012,  0.0043, -0.0018,  0.0035],
#         [-0.0046,  0.0108, -0.0245,  0.0210,  0.0135],
#         [-0.0255,  0.0120,  0.0200, -0.0281, -0.0107],
#         [-0.0160,  0.0030, -0.0072,  0.0148, -0.0062],
#         [ 0.0107, -0.0125, -0.0166,  0.0520, -0.0078]], device='cuda:0',
#        dtype=torch.bfloat16)
# tensor([[-0.0022,  0.0012,  0.0043, -0.0018,  0.0035],
#         [-0.0046,  0.0108, -0.0245,  0.0210,  0.0135],
#         [-0.0255,  0.0120,  0.0200, -0.0281, -0.0107],
#         [-0.0160,  0.0030, -0.0072,  0.0148, -0.0062],
#         [ 0.0107, -0.0125, -0.0166,  0.0520, -0.0078]], dtype=torch.bfloat16)



# scores[:5, :5]
# tensor([[   -inf,  0.1060,  0.2539,  0.3652,  0.0359],
#         [   -inf,  0.4922,  0.0933, -0.0435,  0.4961],
#         [   -inf,  0.2871,  0.2402,  0.1338,  0.2012],
#         [   -inf,  0.1406,  0.0047, -0.0442,  0.0688],
#         [   -inf,  0.2832,  0.0205,  0.0206,  0.2295]], device='cuda:0')

# tensor([[   -inf,  0.1064,  0.2559,  0.3652,  0.0359],
#         [   -inf,  0.4922,  0.0933, -0.0435,  0.4961],
#         [   -inf,  0.2871,  0.2402,  0.1328,  0.2012],
#         [   -inf,  0.1406,  0.0047, -0.0442,  0.0688],
#         [   -inf,  0.2832,  0.0205,  0.0206,  0.2295]], dtype=torch.bfloat16)



topk_idx[:5, :5]
tensor([[ 435,  670,  676,  542,  762],
        [ 420,  211, 1245,  196,  427],
        [ 256,   34,   48,  202,  298],
        [1017,  854,  142,  149,  706],
        [1096, 1086, 1267, 2304, 2020]],

tensor([[ 309,  269,  330,  435,  116],
        [  15,   11,  414,  188,  195],
        [ 275,  256,   34,   48,  298],
        [1017,  137,  854,  501,  142],
        [1089, 1087, 1083, 1078, 1080]])

echo "Job Ends" 