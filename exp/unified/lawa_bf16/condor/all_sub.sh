#!/bin/bash

condor_submit_bid 500 exp/unified/lawa_bf16/condor/ogbg_queue.sub
condor_submit_bid 500 exp/unified/lawa_bf16/condor/ogbg_ema.sub
condor_submit_bid 500 exp/unified/lawa_bf16/condor/ogbg_nadamw.sub

condor_submit_bid 500 exp/unified/lawa_bf16/condor/criteo_queue.sub
condor_submit_bid 500 exp/unified/lawa_bf16/condor/criteo_ema.sub
condor_submit_bid 500 exp/unified/lawa_bf16/condor/criteo_nadamw.sub
