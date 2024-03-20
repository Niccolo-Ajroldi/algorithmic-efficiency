#!/bin/bash

source ~/.bashrc
conda activate alpe

export CUDA_VISIBLE_DEVICES=''

# Job specific vars
workload=mnist
dataset=MNIST
submission=reference_algorithms/development_algorithms/mnist/mnist_pytorch/submission.py
search_space=exp/seal/rng/discrete_space.json
trials=1
name="rng_mnist_adam_CPU_1"

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
    --rng_seed=1996
