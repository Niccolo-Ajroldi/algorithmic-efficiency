#!/bin/bash

chmod +x exp/slurm/lawa/lawa/ogbg/lawa_array.sh
chmod +x exp/slurm/lawa/lawa/ogbg/baseline_single.sh

# # submit baseline (5x12=60)
# sbatch exp/slurm/lawa/lawa/ogbg/baseline_single.sh ogbg ogbg 1
# sbatch exp/slurm/lawa/lawa/ogbg/baseline_single.sh ogbg ogbg 2
# sbatch exp/slurm/lawa/lawa/ogbg/baseline_single.sh ogbg ogbg 3
# sbatch exp/slurm/lawa/lawa/ogbg/baseline_single.sh ogbg ogbg 4
# sbatch exp/slurm/lawa/lawa/ogbg/baseline_single.sh ogbg ogbg 5

# # submit lawa (5x1)
# sbatch exp/slurm/lawa/lawa/ogbg/lawa_array.sh ogbg ogbg 1
# sbatch exp/slurm/lawa/lawa/ogbg/lawa_array.sh ogbg ogbg 2
# sbatch exp/slurm/lawa/lawa/ogbg/lawa_array.sh ogbg ogbg 3
# sbatch exp/slurm/lawa/lawa/ogbg/lawa_array.sh ogbg ogbg 4
# sbatch exp/slurm/lawa/lawa/ogbg/lawa_array.sh ogbg ogbg 5

# submit baseline (5x12=60)
sbatch exp/slurm/lawa/lawa/ogbg/baseline_single.sh criteo1tb criteo1tb 1
sbatch exp/slurm/lawa/lawa/ogbg/baseline_single.sh criteo1tb criteo1tb 2
sbatch exp/slurm/lawa/lawa/ogbg/baseline_single.sh criteo1tb criteo1tb 3
sbatch exp/slurm/lawa/lawa/ogbg/baseline_single.sh criteo1tb criteo1tb 4
sbatch exp/slurm/lawa/lawa/ogbg/baseline_single.sh criteo1tb criteo1tb 5

# # submit baseline (5x1)
# sbatch exp/slurm/lawa/lawa/ogbg/lawa_array.sh criteo1tb criteo1tb 1
# sbatch exp/slurm/lawa/lawa/ogbg/lawa_array.sh criteo1tb criteo1tb 2
# sbatch exp/slurm/lawa/lawa/ogbg/lawa_array.sh criteo1tb criteo1tb 3
# sbatch exp/slurm/lawa/lawa/ogbg/lawa_array.sh criteo1tb criteo1tb 4
# sbatch exp/slurm/lawa/lawa/ogbg/lawa_array.sh criteo1tb criteo1tb 5