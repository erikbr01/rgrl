#!/usr/bin/env bash
# Shared DROID DiT flow-matching recipe.
#
# Usage:
#   train_droid_flow.sh DATA TARGET TRAINING \
#     WANDB_PROJECT WANDB_NAME OUT [hydra overrides...]
#
# TRAINING selects a configs/training/*.yaml recipe (e.g. droid, droid_plus)
# that owns the batch size, data path, step budget, and save cadence. To change
# steps/save_every/batch_size, edit that config rather than this script.
#
# Environment overrides:
#   DROID_BATCH_SIZE           # override training.batch_size (else the config's)
#   DROID_IMAGE_CROP='[160,288]'
#   DROID_SHARE_IMAGE_ENCODER=false
#
# Camera-weight sharing is deliberately opt-in: it is faster and substantially
# smaller, but changes the model class and needs a robot-quality ablation.
set -euo pipefail

if (( $# < 6 )); then
  echo "usage: $0 DATA TARGET TRAINING WANDB_PROJECT WANDB_NAME OUT [overrides...]" >&2
  exit 2
fi

data_config=$1
target_config=$2
training_config=$3
wandb_project=$4
wandb_name=$5
run_out=$6
shift 6

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

extra_overrides=()
if [[ -n "${DROID_BATCH_SIZE:-}" ]]; then
  extra_overrides+=("training.batch_size=${DROID_BATCH_SIZE}")
fi

exec uv run --extra video --extra wandb python train.py \
  data="$data_config" \
  target="$target_config" \
  policy=dit_flow \
  training="$training_config" \
  "policy.data_transforms.0.size=${DROID_IMAGE_CROP:-[160,288]}" \
  "policy.share_image_encoder=${DROID_SHARE_IMAGE_ENCODER:-false}" \
  training.wandb_project="$wandb_project" \
  training.wandb_name="$wandb_name" \
  out="$run_out" \
  ${extra_overrides[@]+"${extra_overrides[@]}"} \
  "$@"
