#!/bin/bash

source ~/miniforge3/etc/profile.d/conda.sh
conda activate alpe

# Env vars
export HOME=/home/najroldi
export CODE_DIR=/home/najroldi/algorithmic-efficiency
export EXP_DIR=/fast/najroldi/exp/algoperf
export DATA_DIR=/fast/najroldi/data

# Job specific vars
framework=$1
submission=$2
search_space=$3
study=$4
name=$5
rng_seed=$6
allow_tf_32=$7
cluster_id=$8
process_id=$9

# Define workload from process_id
# CONDOR job arrays range from 0 to n-1
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
if ((process_id < 0 || process_id >= ${#workload_list[@]})); then
    echo "Error: Invalid process_id. Must be between 0 and ${#workload_list[@]}-1."
    exit 1
fi
workload=${workload_list[$process_id]}

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

# Execute python script
OMP_NUM_THREADS=1 torchrun \
  --redirects 1:0,2:0,3:0 \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  $CODE_DIR/submission_runner.py \
  --workload=$workload \
  --framework=$framework \
  --tuning_ruleset=external \
  --data_dir=$DATA_DIR/$dataset \
  --imagenet_v2_data_dir=$DATA_DIR/$dataset \
  --librispeech_tokenizer_vocab_path=$tokenizer_path \
  --submission_path=$submission \
  --tuning_search_space=$search_space \
  --num_tuning_trials=1 \
  --experiment_dir=$EXP_DIR  \
  --experiment_name=$experiment_name \
  --save_intermediate_checkpoints=False \
  --save_checkpoints=False \
  --use_wandb \
  --rng_seed=$rng_seed \
  --torch_compile=True \
  --allow_tf32=$allow_tf_32_flag \
  --halve_CUDA_mem=False \
  --pytorch_eval_num_workers=$pytorch_eval_num_workers \
  --resume_last_run \
  --cluster_id $cluster_id \
  --process_id $process_id \
  --max_pct_of_global_steps=0.1

# resuming is needed with multiple paralell processes accessing the same dir
