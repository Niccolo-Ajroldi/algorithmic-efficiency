#!/bin/bash

source ~/miniforge3/etc/profile.d/conda.sh
conda activate alpe

# Env vars
export HOME=/home/najroldi
export CODE_DIR=/home/najroldi/algorithmic-efficiency
export EXP_DIR=/fast/najroldi/exp/algoperf
export DATA_DIR=/fast/najroldi/data
export TMPDIR=/fast/najroldi/tmp

export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy

# Job specific vars
workload=${1}
framework=${2}
submission=${3}
search_space=${4}
num_tuning_trials=${5}
study=${6}

name=${7}
baseline_ckpt_dir=${8}
eval_every_n_steps=${9}

rng_seed=${10}
allow_tf_32=${11}
run_until_the_end=${12}
target_setting=${13}

cluster_id=${14}
process_id=${15}

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
if [ "$allow_tf_32" == "True" ]; then
  allow_tf_32_flag=True
fi

# run_until_the_end
run_until_the_end_flag=False
if [ "$run_until_the_end" == "True" ]; then
  run_until_the_end_flag=True
fi

# max_pct_of_global_steps
max_pct_of_global_steps=1.0
if [ "$target_setting" == "True" ]; then
  max_pct_of_global_steps=0.75
fi

# Execute python script
OMP_NUM_THREADS=1 torchrun \
  --redirects 1:0 \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  $CODE_DIR/eval_ckpt.py \
  --workload=$workload \
  --framework=$framework \
  --tuning_ruleset=external \
  --data_dir=$DATA_DIR/$dataset \
  --imagenet_v2_data_dir=$DATA_DIR/$dataset \
  --librispeech_tokenizer_vocab_path=$tokenizer_path \
  --submission_path=$submission \
  --tuning_search_space=$search_space \
  --num_tuning_trials=$num_tuning_trials \
  --fixed_space \
  --trial_index=$trial_index \
  --experiment_dir=$EXP_DIR  \
  --experiment_name=$experiment_name \
  --resume_last_run=True \
  --eval_every_n_steps=$eval_every_n_steps \
  --save_checkpoints=False \
  --save_intermediate_checkpoints=False \
  --baseline_ckpt_dir=$baseline_ckpt_dir \
  --overwrite \
  --use_wandb \
  --rng_seed=$rng_seed \
  --torch_compile=True \
  --allow_tf32=$allow_tf_32_flag \
  --run_until_the_end=$run_until_the_end_flag \
  --halve_CUDA_mem=False \
  --pytorch_eval_num_workers=$pytorch_eval_num_workers \
  --log_lr=False \
  --max_pct_of_global_steps=$max_pct_of_global_steps \
  --cluster_id $cluster_id \
  --process_id $process_id

# resuming is needed with multiple paralell processes accessing the same dir
