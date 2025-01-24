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
workload=librispeech_conformer
framework=pytorch
submission=submissions/lawa_ema/lawa_ema_offline.py
search_space="reference_algorithms/target_setting_algorithms/${workload}/ema_01.json"
num_tuning_trials=1
study=2

name=eval_ckpt_newp_debug_ema_01
baseline_ckpt_dir=/fast/najroldi/exp/algoperf/nadamw_newp_01_study_1/librispeech_conformer_pytorch/trial_1
eval_every_n_steps=2048


rng_seed=2
allow_tf_32=True
run_until_the_end=True
target_setting=True

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
  --rng_seed=$rng_seed \
  --torch_compile=True \
  --allow_tf32=$allow_tf_32_flag \
  --run_until_the_end=$run_until_the_end_flag \
  --pytorch_eval_num_workers=$pytorch_eval_num_workers \
  --use_wandb \
  --max_pct_of_global_steps=$max_pct_of_global_steps

  # --deterministic=True \


# resuming is needed with multiple paralell processes accessing the same dir
