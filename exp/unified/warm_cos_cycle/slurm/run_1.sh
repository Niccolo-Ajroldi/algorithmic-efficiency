#!/bin/bash

#SBATCH --job-name=wc_schedule
#SBATCH --array=1
#SBATCH --error=/ptmp/deok/logs/algoperf/err/%x_%A_%a.err
#SBATCH --output=/ptmp/deok/logs/algoperf/out/%x_%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --ntasks 1
#SBATCH --requeue
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=500000

source ~/miniconda3/etc/profile.d/conda.sh
conda activate alpe

# Env vars
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CODE_DIR=~/algorithmic-efficiency
export EXP_DIR=/ptmp/deok/exp/algoperf
export DATA_DIR=/ptmp/najroldi/data

dataset=$1
workload=$2
search_space=$3
study=$4
exp_name=$5

submission=submissions/warm_cos_cycle/nadamw_warm_cos_cycle.py
num_tuning_trials=${SLURM_ARRAY_TASK_MAX}
trial_index=${SLURM_ARRAY_TASK_ID}
rng_seed=1

srun exp/unified/warm_cos_cycle/slurm/auto_run.sh \
  ${dataset} \
  ${workload} \
  ${submission} \
  ${search_space} \
  ${exp_name} \
  ${study} \
  ${num_tuning_trials} \
  ${trial_index} \
  ${rng_seed}
