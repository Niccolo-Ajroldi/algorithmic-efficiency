#!/bin/bash

chmod +x exp/unified/full_b/slurm/imagenet_resnet.sh
chmod +x exp/unified/full_b/slurm/imagenet_vit.sh
chmod +x exp/unified/full_b/slurm/wmt.sh

sbatch exp/unified/full_b/slurm/imagenet_resnet.sh
sbatch exp/unified/full_b/slurm/imagenet_vit.sh
sbatch exp/unified/full_b/slurm/wmt.sh
