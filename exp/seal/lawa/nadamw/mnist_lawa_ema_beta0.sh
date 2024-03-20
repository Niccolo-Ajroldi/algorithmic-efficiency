#!/bin/bash

source ~/.bashrc
conda activate alpe

export CUDA_VISIBLE_DEVICES=''

# Job specific vars
workload=mnist
dataset=MNIST
submission=submissions/lawa_ema/lawa_ema.py
search_space=exp/seal/lawa/nadamw/space_3.json
trials=1
name="lawa_ema_real_cpu_02"
eval_freq=20

# Execute python script
python3 $CODE_DIR/submission_runner_fixed_eval.py \
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
    --rng_seed=1996
