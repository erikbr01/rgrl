#!/usr/bin/env bash
# Baseline plus 60 disjoint right-task episodes. The 1.5x sample budget keeps
# epochs comparable to the 120-episode baseline.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
exec "$(dirname "${BASH_SOURCE[0]}")/train_droid_flow.sh" \
  set_table_plusright set_table_baseline \
  76800000 25600000 \
  set-table-droid set_table_plusright_dit \
  runs/set_table_plusright_dit \
  "$@"
