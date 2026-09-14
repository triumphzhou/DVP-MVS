#!/usr/bin/env bash
# Batch pipeline: prepared multi-view data -> MoGeV3 -> DVP-MVS -> OpenMVS fused PLY.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
DVP_PREPARE="$SCRIPT_DIR/prepare_dvp_scene.py"
DVP_FINALIZE="$SCRIPT_DIR/finalize_dvp_scene.py"
DMAP_CONVERTER="$SCRIPT_DIR/convert_dvp_to_openmvs_dmap.py"
DENSE_CONFIG="$SCRIPT_DIR/openmvs_dvp_depth_filter.cfg"

DMAP_PYTHON=${PYTHON_BIN:-python3}
MOGE_ROOT=${DVP_MOGE_ROOT:-/mnt/zhoukaixuan_workspace/code/MoGe}
MOGE_PYTHON=${DVP_MOGE_PYTHON:-$MOGE_ROOT/.venv/bin/python}
MOGE_WEIGHTS=${DVP_MOGE_WEIGHTS:-/mnt/zhoukaixuan_workspace/code/weights/moge-3-vitl/model.pt}
DVP_BIN=${DVP_BIN:-$REPO/build/APD}
OPENMVS_BIN_DIR=${OPENMVS_BIN_DIR:-/mnt/nuplan/open-source-projects/openMVS/install/bin/OpenMVS}
OPENMVS_DENSIFY=${OPENMVS_DENSIFY:-$OPENMVS_BIN_DIR/DensifyPointCloud}
OPENMVS_RUNTIME=${OPENMVS_RUNTIME:-/mnt/nuplan/l3data-reconstruction-bingxing/colmap+openmvs/openmvs_runtime}
COLMAP_RUNTIME=${COLMAP_RUNTIME:-/mnt/nuplan/gsapro-main-test/.deps/colmap-runtime}
EXTRA_LD_LIBRARY_PATH=${EXTRA_LD_LIBRARY_PATH:-"$OPENMVS_RUNTIME:$COLMAP_RUNTIME/usr/lib:$COLMAP_RUNTIME/usr/lib/x86_64-linux-gnu:$COLMAP_RUNTIME/usr/lib/x86_64-linux-gnu/atlas:$COLMAP_RUNTIME/lib/x86_64-linux-gnu"}

DEPTH_MIN=0.1
DEPTH_MAX=100
GPU_IDS=${DVP_GPU_IDS:-0,1,2,3,4,5,6}
OPENMVS_THREADS=${OPENMVS_THREADS:-8}
UNBOUNDED_LIDAR=1
RESUME=0
CHECK_ONLY=0
FROM_STAGE=1
TO_STAGE=4
BATCH_FILE=
SAMPLE_ROOT=
RESULT_ROOT=

usage() {
  cat <<'EOF'
Usage (one prepared sample):
  bash mvs_process/run_batch_mvsnet_to_openmvs_ply.sh \
    --sample-root /path/to/preprocessed/sample [--result-root DIR] [options]

Usage (batch TSV):
  bash mvs_process/run_batch_mvsnet_to_openmvs_ply.sh \
    --batch-file mvs_process/batch.example.tsv [options]

sample-root must be an output of run_preprocess_1_to_8.sh containing:
  03_converted/
  06_masks/openmvs_masks/
  07_sparse/triangulated_text/images.txt
  08_openmvs_input/global/scene.mvs
  08_openmvs_input/global/neighbors_diverse_*.txt
  audit/neighbors_diverse_*.tsv

The batch TSV columns are:
  sample_root<TAB>result_root
result_root is optional; outputs are written under sample_root when it is empty.

Options:
  --depth-min M          Minimum depth (default: 0.1 m)
  --depth-max M          Maximum depth (default: 100 m)
  --gpu-ids LIST         GPUs for MoGe/DVP (default: 0,1,2,3,4,5,6)
  --from-stage N         Start at stage N, 1-4 (default: 1)
  --to-stage N           Stop after stage N, 1-4 (default: 4)
  --resume               Reuse stages with completion markers
  --bounded-lidar        Use the 80 m bound for MoGe/LiDAR scale fitting
  --check                Validate dependencies only
  -h, --help             Show this help

Stages:
  1  converted data -> MoGeV3 priors and DVP MVSNet-style scene
  2  DVP-MVS PatchMatch and DVP APD.ply fusion
  3  DVP depth maps -> OpenMVS DMAP
  4  OpenMVS depth filter and dense-fuse -> PLY

This script does not scan or convert PKL files and does not run COLMAP.
EOF
}

die() { echo "ERROR: $*" >&2; exit 2; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }
require_dir() { [[ -d "$1" ]] || die "missing directory: $1"; }
require_exe() { [[ -x "$1" ]] || die "missing executable: $1"; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }
mark_done() { printf 'PASS %s\n' "$(date -u '+%FT%TZ')" >"$1"; }
stage_done() { [[ "$RESUME" == 1 && -f "$1" ]]; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-file) BATCH_FILE=$2; shift 2 ;;
    --sample-root) SAMPLE_ROOT=$2; shift 2 ;;
    --result-root) RESULT_ROOT=$2; shift 2 ;;
    --depth-min) DEPTH_MIN=$2; shift 2 ;;
    --depth-max) DEPTH_MAX=$2; shift 2 ;;
    --gpu-ids) GPU_IDS=$2; shift 2 ;;
    --from-stage) FROM_STAGE=$2; shift 2 ;;
    --to-stage) TO_STAGE=$2; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --bounded-lidar) UNBOUNDED_LIDAR=0; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$FROM_STAGE" =~ ^[1-4]$ && "$TO_STAGE" =~ ^[1-4]$ ]] || die "stages must be integers in 1-4"
(( FROM_STAGE <= TO_STAGE )) || die "--from-stage must not exceed --to-stage"
[[ "$DEPTH_MIN" =~ ^[0-9]+([.][0-9]+)?$ && "$DEPTH_MAX" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid depth bound"
awk -v low="$DEPTH_MIN" -v high="$DEPTH_MAX" 'BEGIN { exit !(low > 0 && low < high) }' || die "expected 0 < depth-min < depth-max"
IFS=',' read -r -a GPU_ARRAY <<<"$GPU_IDS"
[[ ${#GPU_ARRAY[@]} -gt 0 ]] || die "--gpu-ids is empty"
for gpu in "${GPU_ARRAY[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU ID: $gpu"; done

preflight() {
  if (( FROM_STAGE <= 1 && TO_STAGE >= 1 )); then
    require_file "$DVP_PREPARE"; require_file "$DVP_FINALIZE"
    require_exe "$MOGE_PYTHON"; require_file "$MOGE_WEIGHTS"
    env PYTHONPATH="$MOGE_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$MOGE_PYTHON" -c \
      'import cv2, numpy, torch, moge' || die "MoGe Python cannot import the required runtime"
  fi
  if (( FROM_STAGE <= 2 && TO_STAGE >= 2 )); then require_exe "$DVP_BIN"; fi
  if (( FROM_STAGE <= 4 && TO_STAGE >= 3 )); then
    require_file "$DMAP_CONVERTER"
    if [[ "$DMAP_PYTHON" == */* ]]; then require_exe "$DMAP_PYTHON"; else require_command "$DMAP_PYTHON"; fi
    "$DMAP_PYTHON" -c 'import cv2, numpy' || die "DMAP Python lacks cv2/numpy"
  fi
  if (( FROM_STAGE <= 4 && TO_STAGE >= 4 )); then
    require_file "$DENSE_CONFIG"; require_exe "$OPENMVS_DENSIFY"
    local missing
    missing=$(LD_LIBRARY_PATH="$EXTRA_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$OPENMVS_DENSIFY" | awk '/not found/{print $1}' | paste -sd, -)
    [[ -z "$missing" ]] || die "unresolved OpenMVS libraries: $missing"
  fi
}

preflight
if [[ "$CHECK_ONLY" == 1 ]]; then
  cat <<EOF
PASS: MVSNet-to-PLY dependencies found
MoGe Python:    $MOGE_PYTHON
MoGe weights:   $MOGE_WEIGHTS
DVP executable: $DVP_BIN
OpenMVS:        $OPENMVS_DENSIFY
GPUs:           $GPU_IDS
depth range:    $DEPTH_MIN-$DEPTH_MAX m
EOF
  exit 0
fi

if [[ -n "$BATCH_FILE" ]]; then
  require_file "$BATCH_FILE"
  [[ -z "$SAMPLE_ROOT$RESULT_ROOT" ]] || die "use either --batch-file or --sample-root"
else
  [[ -n "$SAMPLE_ROOT" ]] || { usage >&2; die "--sample-root or --batch-file is required"; }
fi
export LD_LIBRARY_PATH="$EXTRA_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

run_moge_prepare() {
  local run_root=$1 source=$2 masks=$3 neighbors_tsv=$4 state=$5 logs=$6
  local marker="$state/step_09_dvp_prepare.complete"
  if stage_done "$marker"; then echo "reuse stage 1 (MoGe/DVP scene)"; return; fi
  mkdir -p "$run_root" "$logs"
  local prepare_args=(--num-shards "${#GPU_ARRAY[@]}" --depth-min "$DEPTH_MIN" --depth-max "$DEPTH_MAX")
  local finalize_args=(--depth-min "$DEPTH_MIN" --depth-max "$DEPTH_MAX")
  if [[ "$UNBOUNDED_LIDAR" == 1 ]]; then prepare_args+=(--unbounded-lidar-calibration); finalize_args+=(--save-unbounded-prior); fi
  local shard status=0
  local -a pids=()
  for shard in "${!GPU_ARRAY[@]}"; do
    env CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$shard]}" PYTHONPATH="$MOGE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      DVP_RUN_ROOT="$run_root" DVP_SOURCE="$source" DVP_OPENMVS_MASKS="$masks" DVP_MOGE_WEIGHTS="$MOGE_WEIGHTS" \
      "$MOGE_PYTHON" -u "$DVP_PREPARE" --shard "$shard" "${prepare_args[@]}" >"$logs/moge-shard-$shard.log" 2>&1 &
    pids+=("$!")
  done
  for shard in "${!pids[@]}"; do
    if ! wait "${pids[$shard]}"; then echo "MoGe shard $shard failed: $logs/moge-shard-$shard.log" >&2; status=1; fi
  done
  [[ "$status" == 0 ]] || return "$status"
  env PYTHONPATH="$MOGE_ROOT${PYTHONPATH:+:$PYTHONPATH}" DVP_RUN_ROOT="$run_root" DVP_SOURCE="$source" DVP_NEIGHBORS_TSV="$neighbors_tsv" \
    "$MOGE_PYTHON" -u "$DVP_FINALIZE" "${finalize_args[@]}" 2>&1 | tee "$logs/dvp-finalize.log"
  require_file "$run_root/manifest.json"; require_file "$run_root/scene/pair.txt"; mark_done "$marker"
}

run_dvp() {
  local run_root=$1 state=$2 logs=$3 marker="$2/step_10_dvp.complete"
  if stage_done "$marker"; then echo "reuse stage 2 (DVP-MVS)"; return; fi
  rm -rf "$run_root/scene/APD" "$run_root/scene/APD_parallel_barrier"
  mkdir -p "$run_root/scene/APD" "$logs"
  local worker status=0 count=${#GPU_ARRAY[@]}
  local -a pids=()
  for worker in "${!GPU_ARRAY[@]}"; do
    (cd "$REPO"; env CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$worker]}" DVP_FUSION_MIN_CONSISTENT="${DVP_FUSION_MIN_CONSISTENT:-2}" \
      DVP_DEPTH_SPECKLE_SIZE="${DVP_DEPTH_SPECKLE_SIZE:-7}" "$DVP_BIN" "$run_root/scene" 0 "$worker" "$count" \
      >"$logs/dvp-worker-$worker.log" 2>&1) &
    pids+=("$!")
  done
  for worker in "${!pids[@]}"; do
    if ! wait "${pids[$worker]}"; then echo "DVP worker $worker failed: $logs/dvp-worker-$worker.log" >&2; status=1; fi
  done
  [[ "$status" == 0 ]] || return "$status"
  cat "$logs"/dvp-worker-*.log >"$logs/dvp.log"
  [[ -s "$run_root/scene/APD/APD.ply" ]] || die "DVP did not produce APD.ply"
  mark_done "$marker"
}

convert_dmaps() {
  local run_root=$1 source=$2 scene=$3 images_txt=$4 neighbors=$5 fusion_root=$6 state=$7 logs=$8 reset=$9
  local marker="$state/step_11_dmap_conversion.complete" dmaps="$fusion_root/dmaps"
  if [[ "$reset" == 0 ]] && stage_done "$marker"; then echo "reuse stage 3 (DMAP)"; return; fi
  [[ "$reset" == 1 ]] || rm -rf "$dmaps"
  mkdir -p "$dmaps" "$logs"
  "$DMAP_PYTHON" "$DMAP_CONVERTER" --run "$run_root" --source "$source" --scene "$scene" --images-txt "$images_txt" \
    --neighbors "$neighbors" --output "$dmaps" --depth-min "$DEPTH_MIN" --depth-max "$DEPTH_MAX" 2>&1 | tee "$logs/dmap-convert.log"
  local count; count=$(find "$dmaps" -maxdepth 1 -name 'depth*.dmap' | wc -l)
  [[ "$count" -eq 707 ]] || die "expected 707 DMAP files, found $count"
  [[ "$reset" == 1 ]] || mark_done "$marker"
}

run_openmvs() {
  local run_root=$1 source=$2 scene=$3 images_txt=$4 neighbors=$5 masks=$6 fusion_root=$7 state=$8 logs=$9
  local marker="$state/step_12_openmvs_fusion.complete"
  if stage_done "$marker"; then echo "reuse stage 4 (OpenMVS fusion)"; return; fi
  mkdir -p "$fusion_root" "$logs"
  convert_dmaps "$run_root" "$source" "$scene" "$images_txt" "$neighbors" "$fusion_root" "$state" "$logs" 1
  local output="$fusion_root/openmvs_filtered_fused.mvs"
  "$OPENMVS_DENSIFY" -w "$fusion_root/dmaps" -i "$scene" -o "$output" --dense-config-file "$DENSE_CONFIG" \
    --mask-path "$masks" --ignore-mask-label 255 --view-neighbors-file "$neighbors" --cuda-device -2 --max-threads "$OPENMVS_THREADS" \
    --resolution-level 1 --max-resolution 960 --min-resolution 640 --sub-resolution-levels 2 --number-views 20 \
    --number-views-fuse 2 --iters 4 --geometric-iters 0 --fusion-mode 0 --fusion-filter 2 \
    --fusion-depth-diff-threshold 0.010 --fusion-reprojection-threshold 2.5 --postprocess-dmaps 1 \
    --estimate-colors 2 --estimate-normals 2 --tower-mode 0 --estimate-roi 0 --crop-to-roi 0 \
    --filter-point-cloud 0 --remove-dmaps 0 --archive-type 2 -v 3 2>&1 | tee "$logs/openmvs-depth-filter-fusion.log"
  [[ -s "$fusion_root/openmvs_filtered_fused.ply" ]] || die "OpenMVS did not produce the fused PLY"
  mark_done "$marker"
}

process_sample() {
  local sample_root=$1 result_root=$2
  require_dir "$sample_root"; sample_root=$(cd -- "$sample_root" && pwd -P)
  [[ -n "$result_root" ]] || result_root="$sample_root"
  mkdir -p "$result_root"; result_root=$(cd -- "$result_root" && pwd -P)
  local sample_name=${sample_root##*/} source="$sample_root/03_converted" masks="$sample_root/06_masks/openmvs_masks"
  local scene="$sample_root/08_openmvs_input/global/scene.mvs" images_txt="$sample_root/07_sparse/triangulated_text/images.txt"
  local run_root="$result_root/09_dvp" fusion_root="$result_root/10_openmvs_fusion" state="$result_root/.state" logs="$result_root/logs"
  local -a neighbor_files=()
  mapfile -t neighbor_files < <(find "$sample_root/08_openmvs_input/global" -maxdepth 1 -name 'neighbors_diverse_pm*_top20.txt' -type f | sort)
  [[ ${#neighbor_files[@]} -eq 1 ]] || die "$sample_name: expected one top-20 neighbor file, found ${#neighbor_files[@]}"
  local neighbors=${neighbor_files[0]} neighbors_tsv="$sample_root/audit/$(basename "${neighbor_files[0]%.txt}").tsv"
  require_dir "$source"; require_dir "$masks"; require_file "$scene"; require_file "$images_txt"; require_file "$neighbors_tsv"
  mkdir -p "$state" "$logs"

  local request="$state/dvp_openmvs_request.txt" request_tmp="$state/dvp_openmvs_request.tmp"
  cat >"$request_tmp" <<EOF
pipeline_version=2
source=$sample_root
depth_min=$DEPTH_MIN
depth_max=$DEPTH_MAX
unbounded_lidar_calibration=$UNBOUNDED_LIDAR
EOF
  if [[ -f "$request" ]] && ! cmp -s "$request" "$request_tmp"; then
    rm -f "$request_tmp"; die "$sample_name: source/depth settings differ; choose another --result-root or remove stages 1-4"
  fi
  mv "$request_tmp" "$request"

  printf '\n[%s] %s, stages %s-%s\n' "$(date '+%F %T')" "$sample_name" "$FROM_STAGE" "$TO_STAGE"
  if (( FROM_STAGE <= 1 && TO_STAGE >= 1 )); then run_moge_prepare "$run_root" "$source" "$masks" "$neighbors_tsv" "$state" "$logs"; fi
  if (( FROM_STAGE <= 2 && TO_STAGE >= 2 )); then require_file "$run_root/scene/pair.txt"; run_dvp "$run_root" "$state" "$logs"; fi
  if (( FROM_STAGE <= 3 && TO_STAGE >= 3 )); then
    require_file "$run_root/scene/APD/APD.ply"; convert_dmaps "$run_root" "$source" "$scene" "$images_txt" "$neighbors" "$fusion_root" "$state" "$logs" 0
  fi
  if (( FROM_STAGE <= 4 && TO_STAGE >= 4 )); then
    require_file "$state/step_11_dmap_conversion.complete"; run_openmvs "$run_root" "$source" "$scene" "$images_txt" "$neighbors" "$masks" "$fusion_root" "$state" "$logs"
  fi
  echo "PASS: $sample_name completed through stage $TO_STAGE"
  [[ ! -s "$run_root/scene/APD/APD.ply" ]] || echo "  DVP point cloud:     $run_root/scene/APD/APD.ply"
  [[ ! -s "$fusion_root/openmvs_filtered_fused.ply" ]] || echo "  OpenMVS point cloud: $fusion_root/openmvs_filtered_fused.ply"
}

declare -A SEEN_RESULTS=()
PROCESSED=0
if [[ -n "$BATCH_FILE" ]]; then
  while IFS=$'\t' read -r sample_root result_root extra; do
    [[ -z "${sample_root//[[:space:]]/}" || "$sample_root" == \#* || "$sample_root" == sample_root ]] && continue
    [[ -z "$extra" ]] || die "unexpected third TSV column for $sample_root"
    [[ -n "$result_root" ]] || result_root="$sample_root"
    [[ -z "${SEEN_RESULTS[$result_root]:-}" ]] || die "duplicate result_root: $result_root"
    SEEN_RESULTS[$result_root]=1
    process_sample "$sample_root" "$result_root"; PROCESSED=$((PROCESSED + 1))
  done <"$BATCH_FILE"
else
  process_sample "$SAMPLE_ROOT" "$RESULT_ROOT"; PROCESSED=1
fi
(( PROCESSED > 0 )) || die "batch file contains no samples"
printf '\nPASS: all requested samples completed\n'
