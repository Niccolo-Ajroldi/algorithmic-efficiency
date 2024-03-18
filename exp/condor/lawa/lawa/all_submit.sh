#!/bin/bash

chmod +x exp/condor/lawa/lawa/auto_run_array.sh
chmod +x exp/condor/lawa/lawa/auto_run.sh

condor_submit job.sub -append "environment=\"ENV_SEED=1\""

# submit baseline
# 1 study, 1 trial
condor_submit_bid 25 exp/condor/lawa/lawa/baseline.sub

# submit a grid
# 1 study, 12 trials
condor_submit_bid 25 exp/condor/lawa/lawa/lawa_array.sub

# submit k=1
# 1 study, 1 trial
condor_submit_bid 25 exp/condor/lawa/lawa/lawa_k1.sub

# run this script 5 times, modifying the study, but not the seed
# total: 5x14=70 runs
