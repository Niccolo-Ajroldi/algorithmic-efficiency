#!/bin/bash

#SBATCH --job-name=trapez
#SBATCH --array=1-20
#SBATCH --error=/ptmp/najroldi/logs/algoperf/err/%x_%A_%a.err
#SBATCH --output=/ptmp/najroldi/logs/algoperf/out/%x_%A_%a.out
#SBATCH --time=04:00:00
#SBATCH --ntasks 1
#SBATCH --requeue
# --- 4 GPUs on a full node ---
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=500000

source ~/.bashrc
conda activate alpe

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CODE_DIR=~/algorithmic-efficiency
export EXP_DIR=/ptmp/najroldi/exp/algoperf
export DATA_DIR=/ptmp/najroldi/data

# Workload
dataset=fastmri
workload=fastmri

# Submission
# submission='submissions/nadamw_triang/nadamw_triang.py'
# search_space='submissions/nadamw_triang/space_1.json'
submission='submissions/nadamw_trapez/nadamw_trapez.py'
search_space='submissions/nadamw_trapez/space_1.json'

# Experiment name, study
# base_name="nadamw_triang_01"
base_name="nadamw_trapez_01"
study=1

# Set config
experiment_name="${base_name}/study_${study}"
num_tuning_trials=$SLURM_ARRAY_TASK_MAX
trial_index=$SLURM_ARRAY_TASK_ID
rng_seed=1166838470 #$study # same seed across trials

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
  --trial_index=$trial_index \
  --rng_seed=$rng_seed \
  --experiment_dir=$EXP_DIR  \
  --experiment_name=$experiment_name \
  --save_intermediate_checkpoints=False \
  --resume_last_run \
  --use_wandb
