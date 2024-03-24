#!/bin/bash

sbatch exp/unified/lawa_overhead/slurm/nadamw.sh
sbatch exp/unified/lawa_overhead/slurm/lawa.sh
sbatch exp/unified/lawa_overhead/slurm/lawa_ema.sh
sbatch exp/unified/lawa_overhead/slurm/lawa_cpu.sh
sbatch exp/unified/lawa_overhead/slurm/lawa_cpu_smart.sh
sbatch exp/unified/lawa_overhead/slurm/lawa_cpu_smart_bfp16.sh
