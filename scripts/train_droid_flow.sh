#!/usr/bin/env bash
# Shared DROID DiT flow-matching recipe.
#
# Usage:
#   train_droid_flow.sh DATA TARGET STEPS SAVE_EVERY \
#     WANDB_PROJECT WANDB_NAME OUT [hydra overrides...]
#
# STEPS / SAVE_EVERY are optimizer steps. They are sized for the default
# DROID_BATCH_SIZE; if you override the batch size, rescale them so the model
# still sees the intended amount of data.
#
# Environment overrides:
#   DROID_BATCH_SIZE=64
#   DROID_IMAGE_CROP='[160,288]'
#   DROID_SHARE_IMAGE_ENCODER=false
#
# Camera-weight sharing is deliberately opt-in: it is faster and substantially
# smaller, but changes the model class and needs a robot-quality ablation.
set -euo pipefail

if (( $# < 7 )); then
  echo "usage: $0 DATA TARGET STEPS SAVE_EVERY WANDB_PROJECT WANDB_NAME OUT [overrides...]" >&2
  exit 2
fi

data_config=$1
target_config=$2
steps=$3
save_every=$4
wandb_project=$5
wandb_name=$6
run_out=$7
shift 7

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

exec uv run --extra video --extra wandb python train.py \
  data="$data_config" \
  target="$target_config" \
  policy=dit_flow \
  "policy.data_transforms.0.size=${DROID_IMAGE_CROP:-[160,288]}" \
  "policy.share_image_encoder=${DROID_SHARE_IMAGE_ENCODER:-false}" \
  "training.batch_size=${DROID_BATCH_SIZE:-64}" \
  training.data_mode=auto \
  training.dataloader_workers=0 \
  training.wandb_project="$wandb_project" \
  training.wandb_name="$wandb_name" \
  training.steps="$steps" \
  training.save_every="$save_every" \
  out="$run_out" \
  "$@"
