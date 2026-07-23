#!/usr/bin/env bash
# 120-episode prepare-tray selection. The checked-in manifest owns the exact
# IDs and provenance; the cache is a zero-copy logical view of the source.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "$(dirname "${BASH_SOURCE[0]}")/train_droid_flow.sh" \
  prepare_tray_baseline prepare_tray_baseline \
  51200000 25600000 \
  prepare-tray-droid prepare_tray_baseline_dit \
  runs/prepare_tray_baseline_dit \
  "$@"
