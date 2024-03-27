#!/bin/bash

RUN_1=exp/unified/warm_cos_cycle/slurm/run_1.sh
RUN_2=exp/unified/warm_cos_cycle/slurm/run_2.sh

chmod +x exp/unified/warm_cos_cycle/slurm/auto_run.sh
chmod +x ${RUN_1}
chmod +x ${RUN_2}

exp_name="wc_schedule"

trial_1=exp/unified/warm_cos_cycle/json/trial_1.json
trial_2=exp/unified/warm_cos_cycle/json/trial_2.json
trial_5=exp/unified/warm_cos_cycle/json/trial_5.json
trial_5_and_1=exp/unified/warm_cos_cycle/json/trial_5_and_1.json

study=1
sbatch ${RUN_1} ogbg ogbg ${trial_5} ${study} ${exp_name}

