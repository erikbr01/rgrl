#!/usr/bin/env bash
# Trains the dit_flow (ABC-DiT + rectified flow) policy on the
# set_table_plusmiddle DROID task -- same recipe as
# scripts/train_set_table_dit.sh, follow-up data: the baseline's 60
# "prepare plate - milk right" + 60 "prepare plate - milk middle" plus 60
# more, disjoint "prepare plate - milk middle" episodes (180 total: 60
# right + 120 middle; episode_ids in
# configs/data/set_table_plusmiddle.yaml), a strict superset of the
# baseline's episode_ids.
#
# training.steps=300000: baseline's 200000 scaled by this split's data
# increase over the baseline (180/120 = 1.5x), not an independently
# chosen number.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CUDA_VISIBLE_DEVICES=1 exec uv run --extra video --extra wandb python train.py \
  data=set_table_plusmiddle \
  target=set_table_baseline \
  policy=dit_flow \
  'policy.data_transforms.0.size=[160,288]' \
  training.wandb_project=set-table-droid \
  training.wandb_name=set_table_plusmiddle_dit \
  training.save_every=100000 \
  training.steps=300000 \
  training.pin_fallback_gb=8 \
  training.dataloader_workers=2 \
  out=runs/set_table_plusmiddle_dit \
  "$@"
