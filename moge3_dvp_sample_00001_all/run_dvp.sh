#!/usr/bin/env bash
set -euo pipefail
cd /mnt/zhoukaixuan_workspace/code/DVP-MVS
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
./build/APD moge3_dvp_sample_00001_all/scene 0 2>&1 | tee moge3_dvp_sample_00001_all/dvp.log
