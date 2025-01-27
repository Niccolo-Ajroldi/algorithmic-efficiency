#!/bin/bash

#SBATCH --job-name=pg_ema_no_decay_01
#SBATCH --array=1-9
#SBATCH --error=/ptmp/najroldi/logs/algoperf/err/%x_%A_%a.err
#SBATCH --output=/ptmp/najroldi/logs/algoperf/out/%x_%A_%a.out
#SBATCH --time=10:00:00
#SBATCH --ntasks 1
#SBATCH --requeue
# --- 4 GPUs on a full node ---
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=1
#SBATCH --mem=300000

source ~/.bashrc
conda activate alpe

export CODE_DIR=~/algorithmic-efficiency
export EXP_DIR=/ptmp/najroldi/exp/algoperf
export DATA_DIR=/ptmp/najroldi/data

# export HTTP_PROXY=$http_proxy
# export HTTPS_PROXY=$https_proxy

# Job specific vars
workload=librispeech_conformer
framework=pytorch

# ## NADAW
# submission=prize_qualification_baselines/external_tuning/pytorch_nadamw_full_budget.py
# search_space=script/lawa/nadam/nadamw_prova.json
# num_tuning_trials=1
# study=1
# name=nadamw_prova_01

## EMA
submission=submissions/lawa_ema/lawa_ema_no_decay.py
search_space=script/lawa/ema/lawa_trial_5_tune_13.json
num_tuning_trials=6
study=1
name=pg_ema_no_decay_02

rng_seed=${study}
allow_tf_32=1
run_until_the_end=1
target_setting=0

cluster_id=${SLURM_ARRAY_JOB_ID}
process_id=${SLURM_ARRAY_TASK_ID}

# SLURM job arrays range from 1 to n
trial_index=${SLURM_ARRAY_TASK_ID}

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

# max_pct_of_global_steps
max_pct_of_global_steps=1.0
if [ "$target_setting" == "1" ]; then
  max_pct_of_global_steps=0.75
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
  --num_tuning_trials=$num_tuning_trials \
  --fixed_space \
  --trial_index=$trial_index \
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
  --resume_last_run \
  --overwrite \
  --max_pct_of_global_steps=$max_pct_of_global_steps \
  --cluster_id $cluster_id \
  --process_id $process_id

# resuming is needed with multiple paralell processes accessing the same dir
