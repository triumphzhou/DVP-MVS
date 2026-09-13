#!/usr/bin/env bash
set -euo pipefail

REPO=${DVP_REPO:-/mnt/zhoukaixuan_workspace/code/DVP-MVS}
ROOT=${DVP_RUN_ROOT:-$REPO/moge3_dvp_sample_00001_clean7}
PYTHON=${DVP_MOGE_PYTHON:-/mnt/zhoukaixuan_workspace/code/MoGe/.venv/bin/python}
NUM_SHARDS=${DVP_NUM_SHARDS:-7}
DEPTH_MIN=${DVP_DEPTH_MIN:-0.5}
DEPTH_MAX=${DVP_DEPTH_MAX:-80}
export DVP_RUN_ROOT="$ROOT"

prepare_args=(
  --num-shards "$NUM_SHARDS"
  --depth-min "$DEPTH_MIN"
  --depth-max "$DEPTH_MAX"
)
finalize_args=(--depth-min "$DEPTH_MIN" --depth-max "$DEPTH_MAX")
if [[ "${DVP_UNBOUNDED_LIDAR_CALIBRATION:-0}" == "1" ]]; then
  prepare_args+=(--unbounded-lidar-calibration)
  finalize_args+=(--save-unbounded-prior)
fi

mkdir -p "$ROOT"
pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" -u \
    "$REPO/moge3_dvp_sample_00001_clean7/prepare_clean.py" \
    --shard "$shard" "${prepare_args[@]}" >"$ROOT/moge-shard-$shard.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  tail -50 "$ROOT"/moge-shard-*.log
  exit "$status"
fi

"$PYTHON" -u "$REPO/moge3_dvp_sample_00001_clean7/finalize_clean.py" \
  "${finalize_args[@]}" | tee "$ROOT/finalize.log"
