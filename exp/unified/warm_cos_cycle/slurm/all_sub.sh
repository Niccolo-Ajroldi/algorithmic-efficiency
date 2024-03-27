#!/bin/bash

chmod +x exp/unified/warm_cos_cycle/slurm/auto_run.sh
chmod +x ${RUN_1}
chmod +x ${RUN_2}

RUN_1=exp/unified/warm_cos_cycle/slurm/run_1.sh
RUN_2=exp/unified/warm_cos_cycle/slurm/run_2.sh
exp_name="wc_schedule"

trial_1=exp/unified/warm_cos_cycle/json/trial_1.json
trial_2=exp/unified/warm_cos_cycle/json/trial_2.json
trial_5=exp/unified/warm_cos_cycle/json/trial_5.json
trial_5_and_1=exp/unified/warm_cos_cycle/json/trial_5_and_1.json

for study in {1..5}
do
  sbatch ${RUN_1} ogbg ogbg ${trial_5} ${study} ${exp_name}
  sbatch ${RUN_1} fastmri fastmri ${trial_1} ${study} ${exp_name}
  sbatch ${RUN_1} imagenet imagenet_vit ${trial_2} ${study} ${exp_name}

  sbatch ${RUN_2} wmt wmt ${trial_5_and_1} ${study} ${exp_name}
  sbatch ${RUN_2} librsipeech librispeech_conformer ${trial_5_and_1} ${study} ${exp_name}
  sbatch ${RUN_2} librispeech librispeech_deepspeech ${trial_5_and_1} ${study} ${exp_name}
done
