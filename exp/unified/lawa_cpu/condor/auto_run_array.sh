#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate alpe

# Env vars
export OMP_NUM_THREADS=32
export HOME=/home/najroldi
export CODE_DIR=/home/najroldi/algorithmic-efficiency
export EXP_DIR=/fast/najroldi/exp/algoperf
export DATA_DIR=/fast/najroldi/data

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
rng_seed=$9

# Experiment name
experiment_name="${name}"

# Librispeech tokenizer path
tokenizer_path=''
if [ "$dataset" = "librispeech" ]; then
    tokenizer_path="${DATA_DIR}/librispeech/spm_model.vocab"
fi

# Imagenet is in a different folder on mpi cluster
data_dir_2=$DATA_DIR/$dataset
if [ "$dataset" = "imagenet" ]; then
    data_dir="/is/cluster/fast/jpiles/imagenet"
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
  --data_dir=$data_dir_2 \
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
  --use_wandb \
  --rng_seed=$rng_seed \
  --fixed_space
