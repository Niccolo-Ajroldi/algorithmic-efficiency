#!/bin/bash

chmod +x exp/condor/lawa/lawa/auto_run_array.sh

echo "fastmri"
condor_submit_bid 25 exp/condor/lawa/lawa/fastmri_array.sub
