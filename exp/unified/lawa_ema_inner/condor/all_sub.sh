#!/bin/bash

chmod +x exp/unified/lawa_ema_inner/condor/auto_run_array.sh

# condor_submit_bid 2000 exp/unified/lawa_ema_inner/condor/criteo.sub
condor_submit_bid 2000 exp/unified/lawa_ema_inner/condor/fastmri.sub
condor_submit_bid 2000 exp/unified/lawa_ema_inner/condor/wmt.sub
condor_submit_bid 2000 exp/unified/lawa_ema_inner/condor/ogbg.sub
condor_submit_bid 2000 exp/unified/lawa_ema_inner/condor/librispeech_deepspeech.sub
condor_submit_bid 2000 exp/unified/lawa_ema_inner/condor/librispeech_conformer.sub
condor_submit_bid 2000 exp/unified/lawa_ema_inner/condor/imagenet_vit.sub
