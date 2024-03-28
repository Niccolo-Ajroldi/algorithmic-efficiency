#!/bin/bash

source ~/.bashrc
conda activate alpe

export CUDA_VISIBLE_DEVICES=3
# export CUBLAS_WORKSPACE_CONFIG=:4096:8
# export OMP_NUM_THREADS=8

# Job specific vars
workload=ogbg
dataset=ogbg
submission=submissions/lawa_trick/lawa.py
search_space=exp/seal/lawa_trick/space_1.json
trials=1
name="lawa_trick_check_1"
eval_freq=1000

# Execute python script
# torchrun \
#     --redirects 1:0 \
#     --standalone \
#     --nnodes=1 \
#     --nproc_per_node=2 \
#     $CODE_DIR/submission_runner.py \
python3 $CODE_DIR/submission_runner.py \
    --workload=$workload \
    --framework=pytorch \
    --tuning_ruleset=external \
    --data_dir=$DATA_DIR/$dataset \
    --submission_path=$submission \
    --tuning_search_space=$search_space \
    --num_tuning_trials=$trials \
    --experiment_dir=$EXP_DIR  \
    --experiment_name=$name \
    --overwrite \
    --save_checkpoints=False \
    --rng_seed=1

# --use_wandb \