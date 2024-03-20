#!/bin/bash

echo "criteo"
condor_submit_bid 200 exp/condor/lawa/oom/criteo.sub

# echo "fastmri"
# condor_submit_bid 200 exp/condor/lawa/oom/fastmri.sub

echo "imagenet_resnet"
condor_submit_bid 200 exp/condor/lawa/oom/imagenet_resnet.sub

echo "imagenet_vit"
condor_submit_bid 200 exp/condor/lawa/oom/imagenet_vit.sub

echo "librispeech_conformer"
condor_submit_bid 200 exp/condor/lawa/oom/librispeech_conformer.sub

echo "librispeech_deepspeech"
condor_submit_bid 200 exp/condor/lawa/oom/librispeech_deepspeech.sub

# echo "ogbg"
# condor_submit_bid 200 exp/condor/lawa/oom/ogbg.sub

echo "wmt"
condor_submit_bid 200 exp/condor/lawa/oom/wmt.sub