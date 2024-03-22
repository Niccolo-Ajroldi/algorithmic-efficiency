#!/bin/bash

chmod +x exp/unified/lawa_cpu/condor/auto_run_1GPU.sh
chmod +x exp/unified/lawa_cpu/condor/auto_run_8GPU.sh

condor_submit_bid 25 exp/unified/lawa_cpu/condor/criteo.sub
condor_submit_bid 25 exp/unified/lawa_cpu/condor/wmt.sub
