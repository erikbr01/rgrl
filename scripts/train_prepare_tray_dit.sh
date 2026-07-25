#!/usr/bin/env bash
# prepare_tray baseline (configs/experiment/prepare_tray_baseline.yaml).
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "$(dirname "${BASH_SOURCE[0]}")/train_droid_flow.sh" prepare_tray_baseline "$@"
