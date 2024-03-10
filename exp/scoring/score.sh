#!/bin/bash

source ~/.bashrc
conda activate alpe

submission_dir=/ptmp/najroldi/exp/algoperf/scored_submissions
output_dir=/ptmp/najroldi/results/algoperf/scored_submission

python3 scoring/score_submissions.py \
  --submission_directory=$submission_dir \
  --output_dir=$output_dir \
  --compute_performance_profiles=True
