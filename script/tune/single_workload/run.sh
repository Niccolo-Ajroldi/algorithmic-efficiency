#!/bin/bash

set -euo pipefail

echo "which conda:"
which conda || true

echo "which python:"
which python

echo "Activating virtual environment..."
source /home/najroldi/algorithmic-efficiency/.venv/bin/activate

module load cuda/12.9

# Env vars
# export OMP_NUM_THREADS=2 # TODO: check
export HOME=/home/najroldi
export CODE_DIR=/home/najroldi/algorithmic-efficiency
export EXP_DIR=/fast/najroldi/exp/algoperf
export DATA_DIR=/fast/najroldi/data

# Framework and tuning ruleset
framework=pytorch
tuning_ruleset=external

# Job specific vars
process=$1
workload=$2
submission=$3
search_space=$4
num_tuning_trials=$5
study=$6
name=$7
rng_seed=$8
n_gpus=$9

# Map process to hparam indices: process i gets trial i.
# Slicing is 0-indexed, end-exclusive: [hparam_start_index, hparam_end_index)
hparam_start_index=$process
hparam_end_index=$((process + 1))

# Map workload to dataset via an associative array
declare -A workload_to_dataset=(
  [criteo1tb]="criteo1tb"
  [fastmri]="fastmri"
  [imagenet_resnet]="imagenet"
  [imagenet_vit]="imagenet"
  [librispeech_conformer]="librispeech"
  [librispeech_deepspeech]="librispeech"
  [ogbg]="ogbg"
  [wmt]="wmt"
  [finewebedu_lm]="fineweb_edu_10B"
)
dataset=${workload_to_dataset[$workload]}

# Experiment name
experiment_name=${name}_study_${study}

# Librispeech tokenizer path
tokenizer_path=''
if [ "$dataset" = "librispeech" ]; then
    tokenizer_path="${DATA_DIR}/librispeech/spm_model.vocab"
fi

# Increase num_workers on imagenet
framework="pytorch"
pytorch_eval_num_workers=0
if [ "$dataset" == "imagenet" ] && [ "$framework" == "pytorch" ]; then
  pytorch_eval_num_workers=4
fi

# Execute python script
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$n_gpus \
    $CODE_DIR/submission_runner.py \
    --workload=$workload \
    --framework=$framework \
    --tuning_ruleset=$tuning_ruleset \
    --data_dir=$DATA_DIR/$dataset \
    --imagenet_v2_data_dir=$DATA_DIR/$dataset \
    --librispeech_tokenizer_vocab_path=$tokenizer_path \
    --submission_path=$submission \
    --tuning_search_space=$search_space \
    --num_tuning_trials=$num_tuning_trials \
    --hparam_start_index=$hparam_start_index \
    --hparam_end_index=$hparam_end_index \
    --experiment_dir=$EXP_DIR  \
    --experiment_name=$experiment_name \
    --rng_seed=$rng_seed \
    --save_intermediate_checkpoints=False \
    --save_checkpoints=False \
    --resume_last_run=True \
    --pytorch_eval_num_workers=$pytorch_eval_num_workers \
    --use_wandb \
    --torch_compile=True
