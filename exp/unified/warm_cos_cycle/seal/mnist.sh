#!/bin/bash

source ~/.bashrc
conda activate alpe

export CUDA_VISIBLE_DEVICES=5

# Job specific vars
workload=ogbg
dataset=ogbg
submission=submissions/warm_cos_cycle/nadamw_warm_cos_cycle.py
search_space=exp/unified/warm_cos_cycle/json/trial_5_2cycle.json
trials=1
name="warm_cos_ogbg"

# Execute python script
python3 \
    $CODE_DIR/submission_runner.py \
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
    --use_wandb \
    --extra_wandb_logging \
    --rng_seed=1996

    # --torch_deterministic
