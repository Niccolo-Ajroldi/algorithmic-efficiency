#!/bin/bash

#SBATCH --job-name=lawa_cpu
#SBATCH --array=1-4
#SBATCH --error=/ptmp/najroldi/logs/algoperf/err/%x_%A_%a.err
#SBATCH --output=/ptmp/najroldi/logs/algoperf/out/%x_%A_%a.out
#SBATCH --time=24:00:00
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
dataset=imagenet
workload=imagenet_vit

# Same seed across trials
study=1
rng_seed=1

# Submission
submission='submissions/lawa_cpu/lawa.py'
search_space='exp/unified/lawa_cpu/json/trial_4.json'

# Experiment name
base_name="lawa_cpu_01"

# Set config
experiment_name="${base_name}/study_${study}"
num_tuning_trials=${SLURM_ARRAY_TASK_MAX}
trial_index=${SLURM_ARRAY_TASK_ID}

# Librispeech tokenizer path
tokenizer_path=''
if [ "$dataset" = "librispeech" ]; then
    tokenizer_path="${DATA_DIR}/librispeech/spm_model.vocab"
fi

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
  --librispeech_tokenizer_vocab_path=$tokenizer_path \
  --submission_path=$submission \
  --tuning_search_space=$search_space \
  --num_tuning_trials=$num_tuning_trials \
  --trial_index=$trial_index \
  --experiment_dir=$EXP_DIR  \
  --experiment_name=$experiment_name \
  --save_intermediate_checkpoints=False \
  --save_checkpoints=False \
  --resume_last_run \
  --rng_seed=$rng_seed \
  --fixed_space \
  --use_wandb

# resume_last_run: is important when using parallel trials
# multiple jobs will acess the same folder together

# --use_wandb
