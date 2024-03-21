#!/bin/bash

source ~/.bashrc
conda activate alpe

export CUDA_VISIBLE_DEVICES=3,4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8

# Job specific vars
workload=ogbg
dataset=ogbg
submission=submissions/lawa_cpu/lawa.py
search_space=exp/seal/lawa_cpu/space_1.json
trials=1
name="lawa_cpu_k1_01_gpux2_det"
eval_freq=1000

# Execute python script
# python3 $CODE_DIR/submission_runner_fixed_eval.py \
torchrun \
    --redirects 1:0 \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
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
    --save_checkpoints=False \
    --rng_seed=1 \
    --pytorch_eval_num_workers=8 \
    --torch_deterministic

    # --use_wandb \