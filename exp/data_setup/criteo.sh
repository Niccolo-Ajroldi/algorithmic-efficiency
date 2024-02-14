#!/bin/bash

# activate conda env, export DATA_DIR
source data_setup/set_env.sh

python3 datasets/dataset_setup.py \
    --data_dir $DATA_DIR \
    --temp_dir $DATA_DIR/tmp \
    --criteo1tb 