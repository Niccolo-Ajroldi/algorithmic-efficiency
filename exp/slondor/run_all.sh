#!/bin/bash

chmod +x exp/slondor/sl.sh
chmod +x exp/slondor/sl_4h.sh

# sbatch exp/slondor/sl_4h.sh fastmri fastmri 1
# sbatch exp/slondor/sl_4h.sh fastmri fastmri 2
# sbatch exp/slondor/sl_4h.sh fastmri fastmri 3
# sbatch exp/slondor/sl_4h.sh fastmri fastmri 4
# sbatch exp/slondor/sl_4h.sh fastmri fastmri 5

# sbatch exp/slondor/sl_4h.sh criteo1tb criteo1tb 1
# sbatch exp/slondor/sl_4h.sh criteo1tb criteo1tb 2
# sbatch exp/slondor/sl_4h.sh criteo1tb criteo1tb 3
# sbatch exp/slondor/sl_4h.sh criteo1tb criteo1tb 4
# sbatch exp/slondor/sl_4h.sh criteo1tb criteo1tb 5

# sbatch exp/slondor/sl.sh ogbg ogbg 1
# sbatch exp/slondor/sl.sh ogbg ogbg 2
# sbatch exp/slondor/sl.sh ogbg ogbg 3
# sbatch exp/slondor/sl.sh ogbg ogbg 4
# sbatch exp/slondor/sl.sh ogbg ogbg 5

sbatch exp/slondor/sl.sh librispeech librispeech_conformer 1
sbatch exp/slondor/sl.sh librispeech librispeech_conformer 2
sbatch exp/slondor/sl.sh librispeech librispeech_conformer 3
sbatch exp/slondor/sl.sh librispeech librispeech_conformer 4
sbatch exp/slondor/sl.sh librispeech librispeech_conformer 5

sbatch exp/slondor/sl.sh librispeech librispeech_deepspeech 1
sbatch exp/slondor/sl.sh librispeech librispeech_deepspeech 2
sbatch exp/slondor/sl.sh librispeech librispeech_deepspeech 3
sbatch exp/slondor/sl.sh librispeech librispeech_deepspeech 4
sbatch exp/slondor/sl.sh librispeech librispeech_deepspeech 5

sbatch exp/slondor/sl.sh wmt wmt 1
sbatch exp/slondor/sl.sh wmt wmt 2
sbatch exp/slondor/sl.sh wmt wmt 3
sbatch exp/slondor/sl.sh wmt wmt 4
sbatch exp/slondor/sl.sh wmt wmt 5