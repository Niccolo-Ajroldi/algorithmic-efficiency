#!/bin/bash

source ~/.bashrc
conda activate alpe

# export CUDA_VISIBLE_DEVICES=0

# Job specific vars
workload=mnist
dataset=MNIST
submission=submissions/lawa_cpu_bf16/lawa.py
search_space=exp/seal/lawa_ema/nadamw/space_1.json
trials=1
name="check_01"

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
    --save_checkpoints=False \
    --rng_seed=1996
