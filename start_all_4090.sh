#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
OUT=/data02/tanghui/C3_R2SN
mkdir -p "$OUT"
# Recommended: conda activate r2sn BEFORE this script.
# Run with 3 independent patients (2 threads each); tune after memory check.
nohup python -u run_parallel.py \
  --workers 3 --threads-per-worker 2 \
  --nroi 246 \
  --tag lesionaware_BN246_remainingROI_v4 \
  --margin-mm 0 \
  --min-voxels 200 \
  --registration-mask fixed \
  > "$OUT/BN246_remainingROI_v4.log" 2>&1 < /dev/null &
echo "Launcher PID: $!"
echo "Log: $OUT/BN246_remainingROI_v4.log"
