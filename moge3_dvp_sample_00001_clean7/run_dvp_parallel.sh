#!/usr/bin/env bash
set -euo pipefail

REPO=${DVP_REPO:-/mnt/zhoukaixuan_workspace/code/DVP-MVS}
ROOT=${DVP_RUN_ROOT:-$REPO/moge3_dvp_sample_00001_clean7}
NUM_WORKERS=${DVP_NUM_WORKERS:-7}
cd "$REPO"
rm -rf "$ROOT/scene/APD" "$ROOT/scene/APD_parallel_barrier"
mkdir -p "$ROOT/scene/APD"

pids=()
for worker in $(seq 0 $((NUM_WORKERS - 1))); do
  (
    export CUDA_VISIBLE_DEVICES="$worker"
    export DVP_FUSION_MIN_CONSISTENT=2
    export DVP_DEPTH_SPECKLE_SIZE=7
    ./build/APD "$ROOT/scene" 0 "$worker" "$NUM_WORKERS" >"$ROOT/dvp-worker-$worker.log" 2>&1
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  tail -80 "$ROOT"/dvp-worker-*.log
  exit "$status"
fi

cat "$ROOT"/dvp-worker-*.log >"$ROOT/dvp.log"
test -s "$ROOT/scene/APD/APD.ply"
