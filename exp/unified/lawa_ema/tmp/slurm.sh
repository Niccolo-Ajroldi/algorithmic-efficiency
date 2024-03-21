#!/bin/bash

#SBATCH --job-name=lawa_ema_ogbg
#SBATCH --array=1-180
#SBATCH --error=/ptmp/najroldi/logs/algoperf/err/%x_%A_%a.err
#SBATCH --output=/ptmp/najroldi/logs/algoperf/out/%x_%A_%a.out
#SBATCH --time=03:00:00
#SBATCH --ntasks 1
#SBATCH --requeue
# --- 1 GPUs on a single node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=125000

source ~/.bashrc
conda activate alpe

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CODE_DIR=~/algorithmic-efficiency
export EXP_DIR=/ptmp/najroldi/exp/algoperf
export DATA_DIR=/ptmp/najroldi/data

# Workload
dataset=ogbg
workload=ogbg

# Same seed across trials
study=1
rng_seed=1

# Submission
submission=submissions/lawa_ema/lawa_ema.py
search_space=exp/unified/lawa_ema/tmp/ogbg.json

# Experiment name
base_name="lawa_ema"

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
python3 \
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
  --rng_seed=$rng_seed \
  --experiment_dir=$EXP_DIR  \
  --experiment_name=$experiment_name \
  --save_intermediate_checkpoints=False \
  --save_checkpoints=False \
  --use_wandb \
  --resume_last_run \
  --fixed_space # OCIO!

# resume_last_run: is important when using parallel trials
# multiple jobs will acess the same folder together