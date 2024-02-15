#!/bin/bash

# activate conda env, export DATA_DIR
source exp/data_setup/set_env.sh

export CUDA_VISIBLE_DEVICES=0

python3 datasets/dataset_setup.py \
    --data_dir $DATA_DIR \
    --wmt
