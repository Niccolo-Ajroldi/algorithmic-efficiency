#!/bin/bash

source ~/.bashrc
conda activate alpe

export CUDA_VISIBLE_DEVICES=0

# Job specific vars
workload=mnist
dataset=MNIST
submission=submissions/lawa_ema/lawa_ema.py
search_space=exp/seal/lawa_ema/nadamw/space_1.json
trials=1
name="lawa_ema_beta_0"

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
    --max_global_steps 1000 \
    --rng_seed=1996
