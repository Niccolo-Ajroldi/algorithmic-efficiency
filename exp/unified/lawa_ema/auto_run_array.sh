#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate alpe

export OMP_NUM_THREADS=12

# Job specific vars
dataset=$1
workload=$2
submission=$3
search_space=$4
name=$5
study=$6
num_tuning_trials=$7

# $8 should be $(Process)
# CONDOR job arrays range from 0 to n-1, so we add +1 here
# $((...)) is for arithmetic substitution in .sh
trial_index=$(($8 + 1))

# Same seed across trials
rng_seed=$9 # OCIO!! modified for fastmri and criteo

# Experiment name
experiment_name="${name}/study_${study}"

# Librispeech tokenizer path
tokenizer_path=''
if [ "$dataset" = "librispeech" ]; then
    tokenizer_path="${DATA_DIR}/librispeech/spm_model.vocab"
fi

# Execute python scripts
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
  --resume_last_run \
  --use_wandb \
  --fixed_space # OCIO! modified
