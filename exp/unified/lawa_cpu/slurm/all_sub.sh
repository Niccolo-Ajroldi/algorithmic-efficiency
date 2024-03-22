#!/bin/bash

chmod +x exp/unified/lawa_cpu/slurm/imagenet_resnet.sh
chmod +x exp/unified/lawa_cpu/slurm/imagenet_vit.sh
chmod +x exp/unified/lawa_cpu/slurm/librispeech_conformer.sh
chmod +x exp/unified/lawa_cpu/slurm/librispeech_deepspeech.sh

sbatch exp/unified/lawa_cpu/slurm/imagenet_resnet.sh
sbatch exp/unified/lawa_cpu/slurm/imagenet_vit.sh
sbatch exp/unified/lawa_cpu/slurm/librispeech_conformer.sh
sbatch exp/unified/lawa_cpu/slurm/librispeech_deepspeech.sh
