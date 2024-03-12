#!/bin/bash

source ~/.bashrc
conda activate alpe

export CUDA_VISIBLE_DEVICES=0

export CODE_DIR=~/algorithmic-efficiency
export EXP_DIR=/is/sg2/najroldi/exp/algoperf
export DATA_DIR=/is/sg2/najroldi/data

# Workload
dataset=ogbg
workload=ogbg

# Job specific vars
submission='submissions/lawa/nadamw_cos.py'
search_space='submissions/lawa/space_1.json'
name="lawa_lawa_2"
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
  --save_intermediate_checkpoints=False \
  --overwrite \
  --rng_seed=1996 \
  --use_wandb
