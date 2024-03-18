#!/bin/bash

source ~/.bashrc
conda activate alpe

export CODE_DIR=~/algorithmic-efficiency
export EXP_DIR=/ptmp/najroldi/exp/algoperf
export DATA_DIR=/ptmp/najroldi/data

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

# Job specific vars
workload=criteo1tb
dataset=criteo1tb
submission='prize_qualification_baselines/external_tuning/pytorch_nadamw_full_budget.py'
search_space=exp/seal/lawa/nadamw/space_1.json
trials=1
name="A100_x1_OOM"

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
    --save_checkpoints=False \
    --max_global_steps 5000 \
    --rng_seed=1996

