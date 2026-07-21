#!/usr/bin/env bash
# Trains the dit_flow (ABC-DiT + rectified flow) policy on the
# set_table_baseline DROID task -- same recipe as
# scripts/train_prepare_tray_dit.sh, different data. Fixed 125-episode
# subset of the shared droid_converted store (episode_ids in
# configs/data/set_table_baseline.yaml -- "prepare plate - milk right"/
# "milk middle" episodes, the same task_filter diffusion_policy's
# set_table_baseline config uses, frozen to explicit indices here; see
# rgrl.data._convert_episode_subset).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CUDA_VISIBLE_DEVICES=1 exec uv run --extra video --extra wandb python train.py \
  data=set_table_baseline \
  target=set_table_baseline \
  policy=dit_flow \
  'policy.data_transforms.0.size=[160,288]' \
  training.wandb_project=set-table-droid \
  training.wandb_name=set_table_baseline_dit \
  training.save_every=100000 \
  training.steps=500000 \
  training.pin_fallback_gb=8 \
  out=runs/set_table_baseline_dit \
  "$@"
