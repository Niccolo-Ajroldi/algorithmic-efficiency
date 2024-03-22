#!/bin/bash

chmod +x exp/unified/lawa_overhead/slurm/nadamw.sh
chmod +x exp/unified/lawa_overhead/slurm/lawa.sh
chmod +x exp/unified/lawa_overhead/slurm/lawa_cpu.sh
chmod +x exp/unified/lawa_overhead/slurm/lawa_ema.sh

sbatch exp/unified/lawa_overhead/slurm/nadamw.sh
sbatch exp/unified/lawa_overhead/slurm/lawa.sh
sbatch exp/unified/lawa_overhead/slurm/lawa_cpu.sh
sbatch exp/unified/lawa_overhead/slurm/lawa_ema.sh
