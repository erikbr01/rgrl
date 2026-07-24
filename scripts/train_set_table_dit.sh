#!/usr/bin/env bash
# Seed-42 60-right + 60-middle set-table baseline selection.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
exec "$(dirname "${BASH_SOURCE[0]}")/train_droid_flow.sh" \
  set_table_baseline set_table_baseline \
  800000 400000 \
  set-table-droid set_table_baseline_dit \
  runs/set_table_baseline_dit \
  "$@"
