#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate alpe

export CUDA_VISIBLE_DEVICES=6

# Workload
dataset=MNIST
workload=mnist

# Job specific vars
submission='submissions/lawa/dev/mnist_lawa.py'
name="mnist_lawa_2_reload"

search_space='submissions/lawa/dev/space_1.json'
trials=1

# Execute python script
python \
  $CODE_DIR/submission_runner.py \
  --workload=$workload \
  --framework=pytorch \
  --tuning_ruleset=external \
  --data_dir=$DATA_DIR/$dataset \
  --imagenet_v2_data_dir=$DATA_DIR/$dataset \
  --submission_path=$submission \
  --tuning_search_space=$search_space \
  --num_tuning_trials=$trials \
  --experiment_dir=$EXP_DIR  \
  --experiment_name=$name \
  --save_intermediate_checkpoints=True \
  --resume_last_run=True \
  --fixed_space \
  --rng_seed=1996 \
  --use_wandb

# --overwrite \