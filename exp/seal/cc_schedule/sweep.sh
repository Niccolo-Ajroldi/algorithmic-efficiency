#!/bin/bash

source ~/.bashrc
conda activate alpe

# Job specific vars
workload=ogbg
dataset=ogbg
submission=submissions/cc_schedule/nadamw.py
search_space=exp/seal/cc_schedule/trial_5.json
trials=1
name="cc_schedule_debug_01"

trial_index=0
export CUDA_VISIBLE_DEVICES=$trial_index
python3 \
    $CODE_DIR/submission_runner.py \
    --workload=$workload \
    --framework=pytorch \
    --tuning_ruleset=external \
    --data_dir=$DATA_DIR/$dataset \
    --submission_path=$submission \
    --tuning_search_space=$search_space \
    --num_tuning_trials=$trials \
    --fixed_space \
    --trial_index=$trial_index \
    --experiment_dir=$EXP_DIR  \
    --experiment_name=$name \
    --overwrite \
    --save_checkpoints=False \
    --use_wandb \
    --extra_wandb_logging \
    --rng_seed=1996

