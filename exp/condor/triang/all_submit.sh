#!/bin/bash

chmod +x exp/condor/triang/auto_run_array.sh

condor_submit_bid 25 exp/condor/triang/triang_array.sub
