#!/bin/bash

source ~/.bashrc
conda activate alpe

export CUDA_VISIBLE_DEVICES=''

# Job specific vars
workload=mnist
dataset=MNIST
submission=exp/seal/lawa/nadamw/lawa.py
search_space=exp/seal/lawa/nadamw/space.json
trials=1
name="lawa_k1_cpu_01"

# Execute python script
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
    --use_wandb \
    --save_checkpoints=False \
    --max_global_steps 10000 \
    --rng_seed=1996
