#!/bin/bash

source ~/miniforge3/etc/profile.d/conda.sh
conda activate alpe

# Env vars
export OMP_NUM_THREADS=2 # TODO: check
export HOME=/home/najroldi
export CODE_DIR=/home/najroldi/algorithmic-efficiency
export EXP_DIR=/fast/najroldi/exp/algoperf
export DATA_DIR=/fast/najroldi/data

# Job specific vars
workload_or_id=$1
framework=$2
submission=$3
search_space=$4
name=$5
study=$6
num_tuning_trials=$7
rng_seed=$8
allow_tf_32=$9
halve_cuda_mem=${10}

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

# Determine workload based on workload_or_id
if [[ "$workload_or_id" =~ ^[0-7]+$ ]]; then
  # workload_or_id is a valid index (0-7), treat it as job ID and get workload from list
  job_id=$workload_or_id
  workload=${workload_list[$job_id]}
else
  # treat workload_or_id as an explicit workload name
  workload=$workload_or_id
fi

dataset=${workload_to_dataset[$workload]}

# Experiment name
# experiment_name="${name}_${workload}_${framework}"
experiment_name=${name}

# Librispeech tokenizer path
tokenizer_path=''
if [ "$dataset" = "librispeech" ]; then
    tokenizer_path="${DATA_DIR}/librispeech/spm_model.vocab"
fi

# Increase num_workers on imagenet
eval_num_workers=0
if [ "$dataset" == "imagenet" ] && [ "$framework" == "pytorch" ]; then
  eval_num_workers=4
fi

# allow_tf_32
allow_tf_32_flag=False
if [ "$allow_tf_32" == "1" ]; then
  allow_tf_32_flag=True
fi

# allow_tf_32
halve_cuda_mem_flag=False
if [ "$halve_cuda_mem" == "1" ]; then
  halve_cuda_mem_flag=True
fi

# Execute python script
torchrun \
  --redirects 1:0 \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
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
  --hparam_start_index=0 \
  --hparam_end_index=1 \
  --max_pct_of_global_steps=0.1 \
  --experiment_dir=$EXP_DIR  \
  --experiment_name=$experiment_name \
  --save_intermediate_checkpoints=False \
  --save_checkpoints=False \
  --use_wandb \
  --rng_seed=$rng_seed \
  --torch_compile=True \
  --allow_tf32=$allow_tf_32_flag \
  --halve_CUDA_mem=$halve_cuda_mem_flag \
  --pytorch_eval_num_workers=$eval_num_workers \
  --overwrite
