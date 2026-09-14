#!/usr/bin/env bash
# Batch pipeline: L3 PKL -> MoGeV3 -> DVP-MVS depth -> OpenMVS filter/fusion -> PLY.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
STEPS="$SCRIPT_DIR/steps"
PREPROCESS="$SCRIPT_DIR/run_preprocess_1_to_8.sh"
DVP_PREPARE="$REPO/moge3_dvp_sample_00001_clean7/prepare_clean.py"
DVP_FINALIZE="$REPO/moge3_dvp_sample_00001_clean7/finalize_clean.py"
DMAP_CONVERTER="$STEPS/11_convert_dvp_to_openmvs_dmap.py"
DENSE_CONFIG="$SCRIPT_DIR/openmvs_dvp_depth_filter.cfg"

OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/nuplan/l3data-reconstruction-bingxing/preprocess_runs}
PREPROCESS_PYTHON=${PYTHON_BIN:-python3}
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
BUILD_SKY_MASKS=1
FROM_STAGE=1
TO_STAGE=12
BATCH_FILE=
SEGMENTS_TSV=
CLIP_NAME=
SIDE=
FRAME_START=
FRAME_END=
SAMPLE_NAME=

usage() {
  cat <<'EOF'
Usage (batch TSV):
  bash data_preprocess/run_batch_pkl_to_openmvs_ply.sh \
    --batch-file data_preprocess/batch.example.tsv \
    --output-root /path/to/output [options]

Usage (one segment):
  bash data_preprocess/run_batch_pkl_to_openmvs_ply.sh \
    --clip-name CLIP --side left|right --frame-start N --frame-end N \
    --output-root /path/to/output [options]

The TSV columns are:
  clip_name<TAB>side<TAB>frame_start<TAB>frame_end<TAB>sample_name

Options:
  --depth-min M          DVP/OpenMVS minimum depth (default: 0.1 m)
  --depth-max M          DVP/OpenMVS maximum depth (default: 100 m)
  --gpu-ids LIST         Comma-separated GPUs for MoGe/DVP (default: 0,1,2,3,4,5,6)
  --segments-tsv FILE    Reuse the step-1 scanner manifest for every batch row
  --from-stage N         Start at stage N, 1-12 (default: 1)
  --to-stage N           Stop after stage N, 1-12 (default: 12)
  --resume               Reuse stages with completion markers
  --bounded-lidar        Restore the 80 m upper bound for MoGe/LiDAR scale fitting
  --no-sky-mask          Use empty sky masks; camera-10 rig masking remains enabled
  --check                Validate executables, Python environments and scripts only
  -h, --help             Show this help

Stages 1-8 are PKL/COLMAP/OpenMVS scene preprocessing, stage 9 prepares MoGeV3
priors, stage 10 runs DVP-MVS, stage 11 converts DVP depths to OpenMVS DMAP,
and stage 12 applies OpenMVS depth filtering and fuses the final point cloud.
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
    --clip-name) CLIP_NAME=$2; shift 2 ;;
    --side) SIDE=$2; shift 2 ;;
    --frame-start) FRAME_START=$2; shift 2 ;;
    --frame-end) FRAME_END=$2; shift 2 ;;
    --sample-name) SAMPLE_NAME=$2; shift 2 ;;
    --output-root) OUTPUT_ROOT=$2; shift 2 ;;
    --segments-tsv) SEGMENTS_TSV=$2; shift 2 ;;
    --depth-min) DEPTH_MIN=$2; shift 2 ;;
    --depth-max) DEPTH_MAX=$2; shift 2 ;;
    --gpu-ids) GPU_IDS=$2; shift 2 ;;
    --from-stage) FROM_STAGE=$2; shift 2 ;;
    --to-stage) TO_STAGE=$2; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --bounded-lidar) UNBOUNDED_LIDAR=0; shift ;;
    --no-sky-mask) BUILD_SKY_MASKS=0; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$FROM_STAGE" =~ ^([1-9]|1[0-2])$ && "$TO_STAGE" =~ ^([1-9]|1[0-2])$ ]] || \
  die "--from-stage and --to-stage must be integers in 1-12"
(( FROM_STAGE <= TO_STAGE )) || die "--from-stage must not exceed --to-stage"
[[ "$DEPTH_MIN" =~ ^[0-9]+([.][0-9]+)?$ && "$DEPTH_MAX" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
  die "depth bounds must be positive decimal numbers"
awk -v low="$DEPTH_MIN" -v high="$DEPTH_MAX" 'BEGIN { exit !(low > 0 && low < high) }' || \
  die "expected 0 < --depth-min < --depth-max"
IFS=',' read -r -a GPU_ARRAY <<<"$GPU_IDS"
[[ ${#GPU_ARRAY[@]} -gt 0 ]] || die "--gpu-ids is empty"
for gpu in "${GPU_ARRAY[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU ID: $gpu"; done

preflight() {
  if (( FROM_STAGE <= 8 && TO_STAGE >= 1 )); then require_file "$PREPROCESS"; fi
  if (( FROM_STAGE <= 9 && TO_STAGE >= 9 )); then
    require_file "$DVP_PREPARE"; require_file "$DVP_FINALIZE"
  fi
  if (( FROM_STAGE <= 10 && TO_STAGE >= 10 )); then require_exe "$DVP_BIN"; fi
  if (( FROM_STAGE <= 12 && TO_STAGE >= 11 )); then require_file "$DMAP_CONVERTER"; fi
  if (( FROM_STAGE <= 12 && TO_STAGE >= 12 )); then
    require_file "$DENSE_CONFIG"; require_exe "$OPENMVS_DENSIFY"
  fi
  if [[ "$PREPROCESS_PYTHON" == */* ]]; then
    require_exe "$PREPROCESS_PYTHON"
  else
    require_command "$PREPROCESS_PYTHON"
  fi
  "$PREPROCESS_PYTHON" -c 'import cv2, numpy' || die "preprocessing Python lacks cv2/numpy"
  if (( FROM_STAGE <= 8 && TO_STAGE >= 1 )); then
    local preprocess_from=$FROM_STAGE preprocess_to=$TO_STAGE
    (( preprocess_to > 8 )) && preprocess_to=8
    local -a check_args=(--check --from-step "$preprocess_from" --to-step "$preprocess_to")
    [[ "$BUILD_SKY_MASKS" == 0 ]] && check_args+=(--no-sky-mask)
    PYTHON_BIN="$PREPROCESS_PYTHON" bash "$PREPROCESS" "${check_args[@]}"
  fi
  if (( FROM_STAGE <= 9 && TO_STAGE >= 9 )); then
    require_exe "$MOGE_PYTHON"
    require_file "$MOGE_WEIGHTS"
    env PYTHONPATH="$MOGE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      "$MOGE_PYTHON" -c 'import cv2, numpy, torch; from moge.model import import_model_class_by_version; import_model_class_by_version("v3")' || \
      die "MoGe Python cannot import the MoGeV3 runtime"
  fi
  if (( FROM_STAGE <= 12 && TO_STAGE >= 12 )); then
    local missing
    missing=$(LD_LIBRARY_PATH="$EXTRA_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      ldd "$OPENMVS_DENSIFY" | awk '/not found/{print $1}' | paste -sd, -)
    [[ -z "$missing" ]] || die "unresolved OpenMVS libraries: $missing"
  fi
}

preflight
if [[ "$CHECK_ONLY" == 1 ]]; then
  cat <<EOF
PASS: batch pipeline dependencies found
preprocess Python: $PREPROCESS_PYTHON
MoGe Python:      $MOGE_PYTHON
MoGe weights:     $MOGE_WEIGHTS
DVP executable:   $DVP_BIN
OpenMVS:          $OPENMVS_DENSIFY
GPUs:             $GPU_IDS
depth range:      $DEPTH_MIN-$DEPTH_MAX m
EOF
  exit 0
fi

if [[ -n "$BATCH_FILE" ]]; then
  require_file "$BATCH_FILE"
  [[ -z "$CLIP_NAME$SIDE$FRAME_START$FRAME_END$SAMPLE_NAME" ]] || \
    die "use either --batch-file or the single-segment arguments"
else
  [[ -n "$CLIP_NAME" && -n "$SIDE" && -n "$FRAME_START" && -n "$FRAME_END" ]] || {
    usage >&2
    die "a batch file or all single-segment selectors are required"
  }
fi

export LD_LIBRARY_PATH="$EXTRA_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

run_moge_prepare() {
  local run_root=$1 source=$2 masks=$3 neighbors_tsv=$4 state=$5 logs=$6
  local marker="$state/step_09_dvp_prepare.complete"
  if stage_done "$marker"; then echo "reuse completed stage 9"; return; fi
  mkdir -p "$run_root" "$logs"
  local prepare_args=(--num-shards "${#GPU_ARRAY[@]}" --depth-min "$DEPTH_MIN" --depth-max "$DEPTH_MAX")
  local finalize_args=(--depth-min "$DEPTH_MIN" --depth-max "$DEPTH_MAX")
  if [[ "$UNBOUNDED_LIDAR" == 1 ]]; then
    prepare_args+=(--unbounded-lidar-calibration)
    finalize_args+=(--save-unbounded-prior)
  fi
  local shard status=0
  local -a pids=()
  for shard in "${!GPU_ARRAY[@]}"; do
    env CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$shard]}" \
      PYTHONPATH="$MOGE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      DVP_RUN_ROOT="$run_root" DVP_SOURCE="$source" DVP_OPENMVS_MASKS="$masks" \
      DVP_MOGE_WEIGHTS="$MOGE_WEIGHTS" \
      "$MOGE_PYTHON" -u "$DVP_PREPARE" --shard "$shard" "${prepare_args[@]}" \
      >"$logs/moge-shard-$shard.log" 2>&1 &
    pids+=("$!")
  done
  for shard in "${!pids[@]}"; do
    if ! wait "${pids[$shard]}"; then
      echo "MoGe shard $shard failed; see $logs/moge-shard-$shard.log" >&2
      status=1
    fi
  done
  [[ "$status" == 0 ]] || return "$status"
  env PYTHONPATH="$MOGE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    DVP_RUN_ROOT="$run_root" DVP_SOURCE="$source" DVP_NEIGHBORS_TSV="$neighbors_tsv" \
    "$MOGE_PYTHON" -u "$DVP_FINALIZE" "${finalize_args[@]}" \
    2>&1 | tee "$logs/dvp-finalize.log"
  require_file "$run_root/manifest.json"
  require_file "$run_root/scene/pair.txt"
  mark_done "$marker"
}

run_dvp() {
  local run_root=$1 state=$2 logs=$3
  local marker="$state/step_10_dvp.complete"
  if stage_done "$marker"; then echo "reuse completed stage 10"; return; fi
  rm -rf "$run_root/scene/APD" "$run_root/scene/APD_parallel_barrier"
  mkdir -p "$run_root/scene/APD" "$logs"
  local worker status=0 count=${#GPU_ARRAY[@]}
  local -a pids=()
  for worker in "${!GPU_ARRAY[@]}"; do
    (
      cd "$REPO"
      env CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$worker]}" \
        DVP_FUSION_MIN_CONSISTENT="${DVP_FUSION_MIN_CONSISTENT:-2}" \
        DVP_DEPTH_SPECKLE_SIZE="${DVP_DEPTH_SPECKLE_SIZE:-7}" \
        "$DVP_BIN" "$run_root/scene" 0 "$worker" "$count" \
        >"$logs/dvp-worker-$worker.log" 2>&1
    ) &
    pids+=("$!")
  done
  for worker in "${!pids[@]}"; do
    if ! wait "${pids[$worker]}"; then
      echo "DVP worker $worker failed; see $logs/dvp-worker-$worker.log" >&2
      status=1
    fi
  done
  [[ "$status" == 0 ]] || return "$status"
  cat "$logs"/dvp-worker-*.log >"$logs/dvp.log"
  [[ -s "$run_root/scene/APD/APD.ply" ]] || die "DVP did not produce APD.ply"
  mark_done "$marker"
}

convert_dmaps() {
  local run_root=$1 source=$2 scene=$3 images_txt=$4 neighbors=$5 fusion_root=$6 state=$7 logs=$8 reset=$9
  local marker="$state/step_11_dmap_conversion.complete"
  if [[ "$reset" == 0 ]] && stage_done "$marker"; then echo "reuse completed stage 11"; return; fi
  local dmaps="$fusion_root/dmaps"
  if [[ "$reset" == 0 ]]; then
    rm -rf "$dmaps"
  fi
  mkdir -p "$dmaps" "$logs"
  "$PREPROCESS_PYTHON" "$DMAP_CONVERTER" \
    --run "$run_root" --source "$source" --scene "$scene" --images-txt "$images_txt" \
    --neighbors "$neighbors" --output "$dmaps" --depth-min "$DEPTH_MIN" --depth-max "$DEPTH_MAX" \
    2>&1 | tee "$logs/dmap-convert.log"
  local count
  count=$(find "$dmaps" -maxdepth 1 -name 'depth*.dmap' | wc -l)
  [[ "$count" -eq 707 ]] || die "expected 707 converted DMAP files, found $count"
  [[ "$reset" == 1 ]] || mark_done "$marker"
}

run_openmvs() {
  local run_root=$1 source=$2 scene=$3 images_txt=$4 neighbors=$5 masks=$6 fusion_root=$7 state=$8 logs=$9
  local marker="$state/step_12_openmvs_fusion.complete"
  if stage_done "$marker"; then echo "reuse completed stage 12"; return; fi
  mkdir -p "$fusion_root" "$logs"
  # A failed/partial OpenMVS attempt may have modified some DMAPs. Restore all
  # of them from DVP outputs before filtering and fusion.
  convert_dmaps "$run_root" "$source" "$scene" "$images_txt" "$neighbors" "$fusion_root" "$state" "$logs" 1
  local output="$fusion_root/openmvs_filtered_fused.mvs"
  "$OPENMVS_DENSIFY" \
    -w "$fusion_root/dmaps" -i "$scene" -o "$output" \
    --dense-config-file "$DENSE_CONFIG" --mask-path "$masks" --ignore-mask-label 255 \
    --view-neighbors-file "$neighbors" --cuda-device -2 --max-threads "$OPENMVS_THREADS" \
    --resolution-level 1 --max-resolution 960 --min-resolution 640 --sub-resolution-levels 2 \
    --number-views 20 --number-views-fuse 2 --iters 4 --geometric-iters 0 \
    --fusion-mode 0 --fusion-filter 2 --fusion-depth-diff-threshold 0.010 \
    --fusion-reprojection-threshold 2.5 --postprocess-dmaps 1 \
    --estimate-colors 2 --estimate-normals 2 --tower-mode 0 --estimate-roi 0 \
    --crop-to-roi 0 --filter-point-cloud 0 --remove-dmaps 0 --archive-type 2 -v 3 \
    2>&1 | tee "$logs/openmvs-depth-filter-fusion.log"
  [[ -s "$fusion_root/openmvs_filtered_fused.ply" ]] || die "OpenMVS did not produce the fused PLY"
  mark_done "$marker"
}

process_sample() {
  local clip=$1 side=$2 start=$3 end=$4 sample=$5
  [[ "$side" == left || "$side" == right ]] || die "$sample: side must be left or right"
  [[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ ]] || die "$sample: invalid frame range"
  (( end - start + 1 == 101 )) || die "$sample: frame range must contain exactly 101 frames"
  [[ "$sample" != */* ]] || die "sample_name must not contain '/': $sample"

  local root="$OUTPUT_ROOT/$sample"
  local state="$root/.state" logs="$root/logs"
  local source="$root/03_converted"
  local masks="$root/06_masks/openmvs_masks"
  local scene="$root/08_openmvs_input/global/scene.mvs"
  local neighbors="$root/08_openmvs_input/global/neighbors_diverse_pm20_top20.txt"
  local neighbors_tsv="$root/audit/neighbors_diverse_pm20_top20.tsv"
  local images_txt="$root/07_sparse/triangulated_text/images.txt"
  local run_root="$root/09_dvp"
  local fusion_root="$root/10_openmvs_fusion"
  mkdir -p "$OUTPUT_ROOT"

  printf '\n[%s] sample %s, stages %s-%s\n' "$(date '+%F %T')" "$sample" "$FROM_STAGE" "$TO_STAGE"
  if (( FROM_STAGE <= 8 && TO_STAGE >= 1 )); then
    local preprocess_from=$FROM_STAGE preprocess_to=$TO_STAGE
    (( preprocess_from < 1 )) && preprocess_from=1
    (( preprocess_to > 8 )) && preprocess_to=8
    local -a args=(--clip-name "$clip" --side "$side" --frame-start "$start" --frame-end "$end" \
      --sample-name "$sample" --output-root "$OUTPUT_ROOT" --from-step "$preprocess_from" --to-step "$preprocess_to")
    [[ -n "$SEGMENTS_TSV" ]] && args+=(--segments-tsv "$SEGMENTS_TSV")
    [[ "$RESUME" == 1 ]] && args+=(--resume)
    [[ "$BUILD_SKY_MASKS" == 0 ]] && args+=(--no-sky-mask)
    PYTHON_BIN="$PREPROCESS_PYTHON" bash "$PREPROCESS" "${args[@]}"
  fi

  mkdir -p "$state" "$logs"
  if (( TO_STAGE >= 9 )); then
    local request="$state/dvp_openmvs_request.txt" request_tmp="$state/dvp_openmvs_request.tmp"
    cat >"$request_tmp" <<EOF
pipeline_version=1
depth_min=$DEPTH_MIN
depth_max=$DEPTH_MAX
unbounded_lidar_calibration=$UNBOUNDED_LIDAR
EOF
    if [[ -f "$request" ]] && ! cmp -s "$request" "$request_tmp"; then
      rm -f "$request_tmp"
      die "$sample: depth/calibration settings differ from the existing run; use a new sample name or remove stages 9-12"
    fi
    mv "$request_tmp" "$request"
  fi
  if (( FROM_STAGE <= 9 && TO_STAGE >= 9 )); then
    require_dir "$source"; require_dir "$masks"; require_file "$neighbors_tsv"
    run_moge_prepare "$run_root" "$source" "$masks" "$neighbors_tsv" "$state" "$logs"
  fi
  if (( FROM_STAGE <= 10 && TO_STAGE >= 10 )); then
    require_file "$run_root/scene/pair.txt"
    run_dvp "$run_root" "$state" "$logs"
  fi
  if (( FROM_STAGE <= 11 && TO_STAGE >= 11 )); then
    require_file "$run_root/scene/APD/APD.ply"; require_file "$scene"; require_file "$images_txt"; require_file "$neighbors"
    convert_dmaps "$run_root" "$source" "$scene" "$images_txt" "$neighbors" "$fusion_root" "$state" "$logs" 0
  fi
  if (( FROM_STAGE <= 12 && TO_STAGE >= 12 )); then
    require_file "$state/step_11_dmap_conversion.complete"; require_dir "$masks"
    run_openmvs "$run_root" "$source" "$scene" "$images_txt" "$neighbors" "$masks" "$fusion_root" "$state" "$logs"
  fi

  echo "PASS: $sample completed through stage $TO_STAGE"
  [[ ! -s "$run_root/scene/APD/APD.ply" ]] || echo "  DVP point cloud:     $run_root/scene/APD/APD.ply"
  [[ ! -s "$fusion_root/openmvs_filtered_fused.ply" ]] || echo "  OpenMVS point cloud: $fusion_root/openmvs_filtered_fused.ply"
}

declare -A SEEN_SAMPLES=()
PROCESSED=0
if [[ -n "$BATCH_FILE" ]]; then
  while IFS=$'\t' read -r clip side start end sample extra; do
    [[ -z "${clip//[[:space:]]/}" || "$clip" == \#* || "$clip" == clip_name ]] && continue
    [[ -z "$extra" ]] || die "unexpected sixth TSV column for $clip"
    [[ -n "$sample" ]] || sample="${clip}_f${start}_${end}_${side}"
    [[ -z "${SEEN_SAMPLES[$sample]:-}" ]] || die "duplicate sample_name in batch: $sample"
    SEEN_SAMPLES[$sample]=1
    process_sample "$clip" "$side" "$start" "$end" "$sample"
    PROCESSED=$((PROCESSED + 1))
  done <"$BATCH_FILE"
else
  [[ -n "$SAMPLE_NAME" ]] || SAMPLE_NAME="${CLIP_NAME}_f${FRAME_START}_${FRAME_END}_${SIDE}"
  process_sample "$CLIP_NAME" "$SIDE" "$FRAME_START" "$FRAME_END" "$SAMPLE_NAME"
  PROCESSED=1
fi

(( PROCESSED > 0 )) || die "batch file contains no samples"
printf '\nPASS: all requested samples completed\n'
