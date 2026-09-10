#!/usr/bin/env bash
set -euo pipefail
cd /mnt/zhoukaixuan_workspace/code/DVP-MVS
export CUDA_VISIBLE_DEVICES=0
./build/APD moge3_dvp_sample_00001/scene 0 > moge3_dvp_sample_00001/run.log 2>&1
