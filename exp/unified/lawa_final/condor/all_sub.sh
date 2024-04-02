#!/bin/bash

# condor_submit_bid 500 exp/unified/lawa_final/condor/c_nadamw.sub

condor_submit_bid 2000 exp/unified/lawa_final/condor/c_lawa.sub
condor_submit_bid 2000 exp/unified/lawa_final/condor/c_lawa_bf16.sub

condor_submit_bid 2000 exp/unified/lawa_final/condor/c_lawa_ema.sub
condor_submit_bid 2000 exp/unified/lawa_final/condor/c_lawa_ema_bf16.sub

# condor_submit_bid 500 exp/unified/lawa_final/condor/c_nadamw_16GB.sub



