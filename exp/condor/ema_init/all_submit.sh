#!/bin/bash

chmod +x exp/condor/ema_init/auto_run.sh
chmod +x exp/condor/ema_init/auto_run_array.sh

echo "criteo"
condor_submit_bid 25 exp/condor/ema_init/criteo.sub

echo "fastmri"
condor_submit_bid 25 exp/condor/ema_init/fastmri.sub

# echo "imagenet_resnet"
# condor_submit_bid 25 exp/condor/ema_init/imagenet_resnet.sub

# echo "imagenet_vit"
# condor_submit_bid 25 exp/condor/ema_init/imagenet_vit.sub

# echo "librispeech_conformer"
# condor_submit_bid 25 exp/condor/ema_init/librispeech_conformer.sub

# echo "librispeech_deepspeech"
# condor_submit_bid 25 exp/condor/ema_init/librispeech_deepspeech.sub

echo "ogbg"
condor_submit_bid 25 exp/condor/ema_init/ogbg_arry.sub

# echo "wmt"
# condor_submit_bid 25 exp/condor/ema_init/wmt.sub