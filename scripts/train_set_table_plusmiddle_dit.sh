#!/usr/bin/env bash
# Baseline plus 60 disjoint middle-task episodes. The 1.5x sample budget keeps
# epochs comparable to the 120-episode baseline.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
exec "$(dirname "${BASH_SOURCE[0]}")/train_droid_flow.sh" \
  set_table_plusmiddle set_table_baseline \
  droid_plus \
  set-table-droid set_table_plusmiddle_dit \
  runs/set_table_plusmiddle_dit \
  "$@"
