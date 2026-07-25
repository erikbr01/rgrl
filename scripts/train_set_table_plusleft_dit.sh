#!/usr/bin/env bash
# set_table baseline + 60 disjoint left episodes
# (configs/experiment/set_table_plusleft.yaml).
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
exec "$(dirname "${BASH_SOURCE[0]}")/train_droid_flow.sh" set_table_plusleft "$@"
