#!/bin/bash

chmod +x exp/slondor/sl.sh

sbatch exp/slondor/sl.sh fastmri fastmri 1
sbatch exp/slondor/sl.sh fastmri fastmri 2
sbatch exp/slondor/sl.sh fastmri fastmri 3
sbatch exp/slondor/sl.sh fastmri fastmri 4
sbatch exp/slondor/sl.sh fastmri fastmri 5

sbatch exp/slondor/sl.sh criteo1tb criteo1tb 1
sbatch exp/slondor/sl.sh criteo1tb criteo1tb 2
sbatch exp/slondor/sl.sh criteo1tb criteo1tb 3
sbatch exp/slondor/sl.sh criteo1tb criteo1tb 4
sbatch exp/slondor/sl.sh criteo1tb criteo1tb 5

sbatch exp/slondor/sl.sh ogbg ogbg 1
sbatch exp/slondor/sl.sh ogbg ogbg 2
sbatch exp/slondor/sl.sh ogbg ogbg 3
sbatch exp/slondor/sl.sh ogbg ogbg 4
sbatch exp/slondor/sl.sh ogbg ogbg 5
