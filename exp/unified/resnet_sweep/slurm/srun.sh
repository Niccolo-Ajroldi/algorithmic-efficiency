#!/bin/bash

#SBATCH --job-name=resnet_sweep
#SBATCH --array=1-36
#SBATCH --error=/ptmp/najroldi/logs/algoperf/err/%x_%A_%a.err
#SBATCH --output=/ptmp/najroldi/logs/algoperf/out/%x_%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --ntasks 1
#SBATCH --requeue
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=500000

source ~/.bashrc
conda activate alpe

# Env vars
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

dataset=imagenet
workload=imagenet_resnet
submission=prize_qualification_baselines/external_tuning/nadamw_full_b_lighter.py
search_space=exp/unified/resnet_sweep/json/sweep_1.json
exp_name=resnet_sweep_check_1
study=1
num_tuning_trials=${SLURM_ARRAY_TASK_MAX}
trial_index=${SLURM_ARRAY_TASK_ID}
rng_seed=1

chmod +x exp/unified/resnet_sweep/slurm/auto_run.sh

srun exp/unified/resnet_sweep/slurm/auto_run.sh \
  ${dataset} \
  ${workload} \
  ${submission} \
  ${search_space} \
  ${exp_name} \
  ${study} \
  ${num_tuning_trials} \
  ${trial_index} \
  ${rng_seed}
