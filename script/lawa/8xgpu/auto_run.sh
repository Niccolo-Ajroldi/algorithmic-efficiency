#!/bin/bash

source ~/miniforge3/etc/profile.d/conda.sh
conda activate alpe

# Env vars
export HOME=/home/najroldi
export CODE_DIR=/home/najroldi/algorithmic-efficiency
export EXP_DIR=/fast/najroldi/exp/algoperf
export DATA_DIR=/fast/najroldi/data
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
export TMPDIR=/fast/najroldi/tmp

module load cuda/11.8

# # Will this allow to set pytorch_eval_num_workers=0? -> yes
# # will it allow to have correct evals on workers>1 ??
# export OMP_NUM_THREADS=1 
# export MKL_NUM_THREADS=1

# Job specific vars
workload=$1
framework=$2
submission=$3
search_space=$4
num_tuning_trials=$5
study=$6
name=$7
rng_seed=$8
allow_tf_32=$9
run_until_the_end=${10}

cluster_id=${11}
process_id=${12}

# CONDOR job arrays range from 0 to n-1, so we add +1 here
# $((...)) is for arithmetic substitution in .sh
trial_index=$((${process_id} + 1))

workload_list=(
  criteo1tb
  fastmri
  imagenet_resnet
  imagenet_vit
  librispeech_conformer
  librispeech_deepspeech
  ogbg
  wmt
)

# in bash need to explicitly declare an associative array
declare -A workload_to_dataset=(
  [criteo1tb]="criteo1tb"
  [fastmri]="fastmri"
  [imagenet_resnet]="imagenet"
  [imagenet_vit]="imagenet"
  [librispeech_conformer]="librispeech"
  [librispeech_deepspeech]="librispeech"
  [ogbg]="ogbg"
  [wmt]="wmt"
)

# Determine dataset based on workload
dataset=${workload_to_dataset[$workload]}

# Experiment name
experiment_name="${name}_study_${study}"

# Librispeech tokenizer path
tokenizer_path=''
if [ "$dataset" = "librispeech" ]; then
    tokenizer_path="${DATA_DIR}/librispeech/spm_model.vocab"
fi

# Increase num_workers on imagenet
pytorch_eval_num_workers=0
if [ "$dataset" == "imagenet" ] && [ "$framework" == "pytorch" ]; then
  pytorch_eval_num_workers=2
fi

# allow_tf_32
allow_tf_32_flag=False
if [ "$allow_tf_32" == "1" ]; then
  allow_tf_32_flag=True
fi

# run_until_the_end
run_until_the_end_flag=False
if [ "$run_until_the_end" == "1" ]; then
  run_until_the_end_flag=True
fi

# Execute python script
torchrun \
  --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  $CODE_DIR/submission_runner.py \
  --workload=$workload \
  --framework=$framework \
  --tuning_ruleset=external \
  --data_dir=$DATA_DIR/$dataset \
  --imagenet_v2_data_dir=$DATA_DIR/$dataset \
  --librispeech_tokenizer_vocab_path=$tokenizer_path \
  --submission_path=$submission \
  --tuning_search_space=$search_space \
  --num_tuning_trials=$num_tuning_trials \
  --experiment_dir=$EXP_DIR  \
  --experiment_name=$experiment_name \
  --save_intermediate_checkpoints=False \
  --save_checkpoints=False \
  --use_wandb \
  --rng_seed=$rng_seed \
  --torch_compile=True \
  --allow_tf32=$allow_tf_32_flag \
  --run_until_the_end=$run_until_the_end_flag \
  --halve_CUDA_mem=False \
  --pytorch_eval_num_workers=$pytorch_eval_num_workers \
  --cluster_id $cluster_id \
  --process_id $process_id

# --fixed_space \
# --trial_index=$trial_index \
# --resume_last_run \

# --max_pct_of_global_steps=0.1
# resuming is needed with multiple paralell processes accessing the same dir
