#!/usr/bin/env bash
set -euo pipefail
OUT=/mnt/zhoukaixuan_workspace/code/DVP-MVS/moge3_dvp_sample_00001_clean7/openmvs_fusion
/mnt/nuplan/open-source-projects/openMVS/install/bin/OpenMVS/DensifyPointCloud \
  -w "$OUT/dmaps" \
  -i /mnt/nuplan/l3data-reconstruction-bingxing/tem-test/colmap+openmvs/batch8_first8_roadmesh/results/sample_00001_clip_M18-2_07_20251202093910_DF_f76_176_left/02_openmvs/global/scene.mvs \
  -o "$OUT/dvp_openmvs_fused.mvs" \
  --dense-config-file "$OUT/dense_config.cfg" \
  --mask-path /mnt/nuplan/l3data-reconstruction-bingxing/tem-test/colmap+openmvs/batch8_first8_roadmesh/results/sample_00001_clip_M18-2_07_20251202093910_DF_f76_176_left/01_masks/openmvs_masks \
  --ignore-mask-label 255 \
  --view-neighbors-file /mnt/nuplan/l3data-reconstruction-bingxing/tem-test/colmap+openmvs/batch8_first8_roadmesh/results/sample_00001_clip_M18-2_07_20251202093910_DF_f76_176_left/00_audit/neighbors_diverse_pm20_top20.txt \
  --cuda-device -2 --max-threads 8 \
  --resolution-level 1 --max-resolution 960 --min-resolution 640 --sub-resolution-levels 2 \
  --number-views 20 --number-views-fuse 2 --iters 4 --geometric-iters 0 \
  --fusion-mode 0 --fusion-filter 2 \
  --fusion-depth-diff-threshold 0.010 --fusion-reprojection-threshold 2.5 \
  --postprocess-dmaps 1 --estimate-colors 2 --estimate-normals 2 \
  --tower-mode 0 --estimate-roi 0 --crop-to-roi 0 --filter-point-cloud 0 \
  --remove-dmaps 0 --archive-type 2 -v 3
