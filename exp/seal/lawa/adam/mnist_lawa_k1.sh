#!/bin/bash

source ~/.bashrc
conda activate alpe

export CUDA_VISIBLE_DEVICES=''

# Job specific vars
workload=mnist
dataset=MNIST
submission=submissions/lawa/mnist/lawa_adam.py
search_space=exp/seal/lawa/space_1.json
trials=1
name="rng_lawa_adam_01_cpu"

# Print GPU infos
# nvidia-smi

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
