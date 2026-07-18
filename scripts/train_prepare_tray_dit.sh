#!/usr/bin/env bash
# Trains the dit_flow (ABC-DiT + rectified flow) policy on the
# prepare_tray_baseline DROID task. Two cameras only (wrist_image_right
# dropped) to fit training on a single GPU -- see
# configs/target/prepare_tray_baseline.yaml.
#
# image_hw is 160 (not diffusion_policy's 162): at 162 the ResNet
# encoder's GroupNorm-backward reduction kernel hits a torch.compile /
# Inductor codegen bug (illegal memory access, reproduces with just the
# encoder -- torch 2.13.0+cu126) -- 160 and 176 both compile cleanly, so
# it's specific to that odd spatial size, not a real shape constraint.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CUDA_VISIBLE_DEVICES=0 exec uv run python train.py \
  data=prepare_tray_baseline \
  target=prepare_tray_baseline \
  policy=dit_flow \
  'policy.image_hw=[160,288]' \
  training.wandb_project=prepare-tray-droid \
  training.wandb_name=prepare_tray_baseline_dit \
  training.save_every=100000 \
  training.steps=500000 \
  out=runs/prepare_tray_baseline_dit \
  "$@"
