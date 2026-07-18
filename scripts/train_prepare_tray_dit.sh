#!/usr/bin/env bash
# Trains the dit_flow (ABC-DiT + rectified flow) policy on the
# prepare_tray_baseline DROID task. Two cameras only (wrist_image_right
# dropped) to fit training on a single GPU -- see
# configs/target/prepare_tray_baseline.yaml.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CUDA_VISIBLE_DEVICES=0 exec uv run python train.py \
  data=prepare_tray_baseline \
  target=prepare_tray_baseline \
  policy=dit_flow \
  'policy.image_hw=[162,288]' \
  training.wandb_project=prepare-tray-droid \
  out=runs/prepare_tray_baseline_dit \
  "$@"
