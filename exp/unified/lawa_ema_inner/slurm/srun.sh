#!/bin/bash

#SBATCH --job-name=ogbg_gamble_dc
#SBATCH --array=1-4
#SBATCH --error=/ptmp/najroldi/logs/algoperf/err/%x_%A_%a.err
#SBATCH --output=/ptmp/najroldi/logs/algoperf/out/%x_%A_%a.out
#SBATCH --time=02:00:00
#SBATCH --ntasks 1
#SBATCH --requeue
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=500000

source ~/miniconda3/etc/profile.d/conda.sh
conda activate alpe

# Env vars
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CODE_DIR=~/algorithmic-efficiency
export EXP_DIR=/ptmp/najroldi/exp/algoperf
export DATA_DIR=/ptmp/najroldi/data

dataset=ogbg
workload=ogbg
submission=submissions/lawa_ema_trick_gamble/lawa.py
search_space=exp/unified/lawa_ema_inner/json_dc/trial_1.json
exp_name=sub_lawaema_gamble_14
study=1
num_tuning_trials=${SLURM_ARRAY_TASK_MAX}
trial_index=${SLURM_ARRAY_TASK_ID}
rng_seed=1

chmod +x exp/unified/lawa_ema_inner/slurm/auto_run.sh

srun exp/unified/lawa_ema_inner/slurm/auto_run.sh \
  ${dataset} \
  ${workload} \
  ${submission} \
  ${search_space} \
  ${exp_name} \
  ${study} \
  ${num_tuning_trials} \
  ${trial_index} \
  ${rng_seed}
