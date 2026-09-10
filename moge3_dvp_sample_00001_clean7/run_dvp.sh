#!/usr/bin/env bash
set -euo pipefail
cd /mnt/zhoukaixuan_workspace/code/DVP-MVS
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export DVP_FUSION_MIN_CONSISTENT=2
export DVP_DEPTH_SPECKLE_SIZE=7
./build/APD moge3_dvp_sample_00001_clean7/scene 0 2>&1 | tee moge3_dvp_sample_00001_clean7/dvp.log
