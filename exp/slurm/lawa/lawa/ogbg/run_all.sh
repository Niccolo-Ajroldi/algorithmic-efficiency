#!/bin/bash

chmod +x exp/slurm/lawa/lawa/ogbg/lawa_array.sh
chmod +x exp/slurm/lawa/lawa/ogbg/baseline_single.sh
chmod +x exp/slurm/lawa/lawa/ogbg/baseline_single_24h.sh

# # submit baseline (5x1)
# sbatch exp/slurm/lawa/lawa/ogbg/baseline_single.sh ogbg ogbg 1
# sbatch exp/slurm/lawa/lawa/ogbg/baseline_single.sh ogbg ogbg 2
# sbatch exp/slurm/lawa/lawa/ogbg/baseline_single.sh ogbg ogbg 3
# sbatch exp/slurm/lawa/lawa/ogbg/baseline_single.sh ogbg ogbg 4
# sbatch exp/slurm/lawa/lawa/ogbg/baseline_single.sh ogbg ogbg 5

# # submit lawa (5x12=60)
# sbatch exp/slurm/lawa/lawa/ogbg/lawa_array.sh ogbg ogbg 1
# sbatch exp/slurm/lawa/lawa/ogbg/lawa_array.sh ogbg ogbg 2
# sbatch exp/slurm/lawa/lawa/ogbg/lawa_array.sh ogbg ogbg 3
# sbatch exp/slurm/lawa/lawa/ogbg/lawa_array.sh ogbg ogbg 4
# sbatch exp/slurm/lawa/lawa/ogbg/lawa_array.sh ogbg ogbg 5

# # submit baseline (5x1)
# sbatch exp/slurm/lawa/lawa/ogbg/baseline_single.sh criteo1tb criteo1tb 1
# sbatch exp/slurm/lawa/lawa/ogbg/baseline_single.sh criteo1tb criteo1tb 2
# sbatch exp/slurm/lawa/lawa/ogbg/baseline_single.sh criteo1tb criteo1tb 3
# sbatch exp/slurm/lawa/lawa/ogbg/baseline_single.sh criteo1tb criteo1tb 4
# sbatch exp/slurm/lawa/lawa/ogbg/baseline_single.sh criteo1tb criteo1tb 5

# # submit lawa (5x12=60)
# sbatch exp/slurm/lawa/lawa/ogbg/lawa_array.sh criteo1tb criteo1tb 1
# sbatch exp/slurm/lawa/lawa/ogbg/lawa_array.sh criteo1tb criteo1tb 2
# sbatch exp/slurm/lawa/lawa/ogbg/lawa_array.sh criteo1tb criteo1tb 3
# sbatch exp/slurm/lawa/lawa/ogbg/lawa_array.sh criteo1tb criteo1tb 4
# sbatch exp/slurm/lawa/lawa/ogbg/lawa_array.sh criteo1tb criteo1tb 5

# submit baseline (5x1)
sbatch exp/slurm/lawa/lawa/ogbg/baseline_single_24h.sh librispeech librispeech_deepspeech 1
sbatch exp/slurm/lawa/lawa/ogbg/baseline_single_24h.sh librispeech librispeech_deepspeech 2
sbatch exp/slurm/lawa/lawa/ogbg/baseline_single_24h.sh librispeech librispeech_deepspeech 3
sbatch exp/slurm/lawa/lawa/ogbg/baseline_single_24h.sh librispeech librispeech_deepspeech 4
sbatch exp/slurm/lawa/lawa/ogbg/baseline_single_24h.sh librispeech librispeech_deepspeech 5

# submit baseline (5x1)
sbatch exp/slurm/lawa/lawa/ogbg/baseline_single_24h.sh librispeech librispeech_conformer 1
sbatch exp/slurm/lawa/lawa/ogbg/baseline_single_24h.sh librispeech librispeech_conformer 2
sbatch exp/slurm/lawa/lawa/ogbg/baseline_single_24h.sh librispeech librispeech_conformer 3
sbatch exp/slurm/lawa/lawa/ogbg/baseline_single_24h.sh librispeech librispeech_conformer 4
sbatch exp/slurm/lawa/lawa/ogbg/baseline_single_24h.sh librispeech librispeech_conformer 5
