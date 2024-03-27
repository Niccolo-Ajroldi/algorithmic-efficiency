#!/bin/bash

source ~/.bashrc
conda activate alpe

export CODE_DIR=~/algorithmic-efficiency
export EXP_DIR=/ptmp/deok/exp/algoperf
export DATA_DIR=/ptmp/najroldi/data

# Job specific vars
workload=ogbg
dataset=ogbg
submission=submissions/warm_cos_cycle/nadamw_warm_cos_cycle.py
search_space=exp/unified/warm_cos_cycle/json/trial_5.json
trials=1
name="destiny_check"

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
