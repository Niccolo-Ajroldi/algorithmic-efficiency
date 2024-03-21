#!/bin/bash

source ~/.bashrc
conda activate alpe

export CUDA_VISIBLE_DEVICES=''
# export CUDA_VISIBLE_DEVICES=0,1
# export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Job specific vars
workload=mnist
dataset=MNIST
submission=submissions/lawa_cpu/lawa.py
search_space=exp/seal/lawa_cpu/space_1.json
trials=1
name="lawa_cpu_real_01_on_CPU"
eval_freq=20

# Execute python script
# torchrun \
#     --redirects 1:0 \
#     --standalone \
#     --nnodes=1 \
#     --nproc_per_node=2 \
python3 \
    $CODE_DIR/submission_runner_fixed_eval.py \
    --eval_freq=$eval_freq \
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
    --use_wandb \
    --save_checkpoints=False \
    --max_global_steps 5000 \
    --rng_seed=1996 \
    --torch_deterministic
