#!/usr/bin/env bash
# set_table baseline (configs/experiment/set_table_baseline.yaml).
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
exec "$(dirname "${BASH_SOURCE[0]}")/train_droid_flow.sh" set_table_baseline "$@"
