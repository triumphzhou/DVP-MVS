#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/zhoukaixuan_workspace/code/DVP-MVS/moge3_dvp_sample_00001_clean7
PYTHON=/mnt/zhoukaixuan_workspace/code/MoGe/.venv/bin/python

pids=()
for shard in $(seq 0 6); do
  CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" -u "$ROOT/prepare_clean.py" \
    --shard "$shard" --num-shards 7 >"$ROOT/moge-shard-$shard.log" 2>&1 &
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

"$PYTHON" -u "$ROOT/finalize_clean.py" | tee "$ROOT/finalize.log"
