#!/bin/bash

chmod +x exp/unified/baseline/condor/auto_run_array.sh

# condor_submit_bid 25 exp/unified/baseline/condor/criteo.sub

condor_submit_bid 25 exp/unified/baseline/condor/wmt.sub

condor_submit_bid 25 exp/unified/baseline/condor/resnet.sub

condor_submit_bid 25 exp/unified/baseline/condor/vit.sub

condor_submit_bid 25 exp/unified/baseline/condor/deepspeech.sub

condor_submit_bid 25 exp/unified/baseline/condor/conformer.sub
