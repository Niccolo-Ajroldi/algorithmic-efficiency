#!/bin/bash

source ~/miniforge3/etc/profile.d/conda.sh
conda activate alpe

# Env vars
export HOME=/home/najroldi
export CODE_DIR=/home/najroldi/algorithmic-efficiency
export EXP_DIR=/fast/najroldi/exp/algoperf
export DATA_DIR=/fast/najroldi/data
export TMPDIR=/fast/najroldi/tmp

# Job specific vars
workload_or_id=ogbg
framework=pytorch
submission=prize_qualification_baselines/external_tuning/pytorch_nadamw_full_budget.py
search_space=script/debug/debug.json
name=resume_debug_01
study=1
num_tuning_trials=1
rng_seed=96
allow_tf_32=1
halve_cuda_mem=0
max_pct_of_global_steps=0.5

workload_list=(
  cifar
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
  [cifar]="cifar10"
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
if [[ "$workload_or_id" =~ ^[0-8]+$ ]]; then
  # workload_or_id is a valid index (0-8), treat it as job ID and get workload from list
  job_id=$workload_or_id
  workload=${workload_list[$job_id]}
else
  # treat workload_or_id as an explicit workload name
  workload=$workload_or_id
fi

dataset=${workload_to_dataset[$workload]}

# Experiment name
# experiment_name="${name}_${workload}_${framework}"
experiment_name=resume_debug_02
resume_experiment_name="resume_debug_01/${workload}_pytorch/trial_1"
resume_last_run=True

# Librispeech tokenizer path
tokenizer_path=''
if [ "$dataset" = "librispeech" ]; then
    tokenizer_path="${DATA_DIR}/librispeech/spm_model.vocab"
fi

# Increase num_workers on imagenet
eval_num_workers=0
if [ "$dataset" == "imagenet" ] && [ "$framework" == "pytorch" ]; then
  eval_num_workers=2
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
  --overwrite \
  --save_checkpoints=True \
  --save_intermediate_checkpoints=True \
  --resume_last_run=$resume_last_run \
  --resume_experiment_name=$resume_experiment_name \
  --rng_seed=$rng_seed \
  --torch_compile=True \
  --allow_tf32=$allow_tf_32_flag \
  --halve_CUDA_mem=$halve_cuda_mem_flag \
  --pytorch_eval_num_workers=$eval_num_workers

  # --use_wandb


  # --trial_index=2 \
  # --fixed_space \
  # --hparam_start_index=0 \
  # --hparam_end_index=1 \
  # --max_pct_of_global_steps=0.2 \
  # --profile \

