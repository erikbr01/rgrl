#!/usr/bin/env bash
# Trains the dit_flow (ABC-DiT + rectified flow) policy on the
# prepare_tray_baseline DROID task. Two cameras only (wrist_image_right
# dropped) to fit training on a single GPU -- see
# configs/target/prepare_tray_baseline.yaml. Uses a fixed 120-episode
# subset (episode_ids in configs/data/prepare_tray_baseline.yaml) rather
# than the store's full 132 -- see rgrl.data._convert_episode_subset.
#
# image_hw is 160 (not diffusion_policy's 162): at 162 the ResNet
# encoder's GroupNorm-backward reduction kernel hits a torch.compile /
# Inductor codegen bug (illegal memory access, reproduces with just the
# encoder -- torch 2.13.0+cu126) -- 160 and 176 both compile cleanly, so
# it's specific to that odd spatial size, not a real shape constraint.
# (policy.image_hw was renamed policy.data_transforms -- see
# rgrl/transforms.py -- so the crop size is now set via the first
# data_transforms entry's `size`, not a top-level policy field.)
#
# pin_fallback_gb=8: force pinned-CPU-resident data rather than GPU-resident
# -- this recipe's DiT (depth=8, hidden=512) + batch_size=256 over two
# 160x288 cameras needs more VRAM headroom than a 24GB GPU has left over
# once data is also GPU-resident (see rgrl/dataset.py, rgrl/train_loop.py's
# TrainConfig.pin_fallback_gb). Explicit rather than relying on this
# store's total size happening to land above the 20GB default threshold.
#
# dataloader_workers=2: past pin_fallback_gb, data lives in a
# CpuSequenceDataset driven by a real torch DataLoader (spawn workers) --
# this overlaps the CPU gather + H2D copy with GPU compute instead of doing
# it synchronously in the training loop. Verified via a real sanity run:
# drops the "data" profiler fraction from ~25-28% to ~5-6% of step time.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CUDA_VISIBLE_DEVICES=0 exec uv run --extra video --extra wandb python train.py \
  data=prepare_tray_baseline \
  target=prepare_tray_baseline \
  policy=dit_flow \
  'policy.data_transforms.0.size=[160,288]' \
  training.wandb_project=prepare-tray-droid \
  training.wandb_name=prepare_tray_baseline_dit \
  training.save_every=100000 \
  training.steps=500000 \
  training.pin_fallback_gb=8 \
  training.dataloader_workers=2 \
  out=runs/prepare_tray_baseline_dit \
  "$@"
