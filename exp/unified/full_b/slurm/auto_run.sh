#!/bin/bash

export CODE_DIR=~/algorithmic-efficiency
export EXP_DIR=/ptmp/najroldi/exp/algoperf
export DATA_DIR=/ptmp/najroldi/data

# Workload
dataset=imagenet
workload=imagenet_resnet

# Same seed across trials
study=1
rng_seed=1

# Submission
submission='prize_qualification_baselines/external_tuning/nadamw_full_b_lighter.py'
search_space='prize_qualification_baselines/external_tuning/tuning_search_space.json'

# Experiment name
base_name="resnet_FAST_02_srun"

# Set config
experiment_name="${base_name}/study_${study}"
num_tuning_trials=1

# Execute python script
torchrun \
  --redirects 1:0,2:0,3:0 \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  $CODE_DIR/submission_runner.py \
  --workload=$workload \
  --framework=pytorch \
  --tuning_ruleset=external \
  --data_dir=$DATA_DIR/$dataset \
  --imagenet_v2_data_dir=$DATA_DIR/$dataset \
  --submission_path=$submission \
  --tuning_search_space=$search_space \
  --num_tuning_trials=$num_tuning_trials \
  --experiment_dir=$EXP_DIR  \
  --experiment_name=$experiment_name \
  --save_intermediate_checkpoints=False \
  --save_checkpoints=False \
  --resume_last_run \
  --rng_seed=$rng_seed \
  --use_wandb
