#!/usr/bin/env bash
# Shared DROID DiT flow launcher.
#
# Usage:
#   train_droid_flow.sh EXPERIMENT [hydra overrides...]
#
# EXPERIMENT names a configs/experiment/*.yaml that owns everything --
# data/target/policy/training groups, image crop, wandb project/name, and out.
# To change any of those, edit the experiment (or its training config), not
# this script. This wrapper only supplies the uv extras and repo cwd; extra
# hydra overrides pass straight through (e.g. training.batch_size=32,
# policy.share_image_encoder=true).
set -euo pipefail

if (( $# < 1 )); then
  echo "usage: $0 EXPERIMENT [hydra overrides...]" >&2
  exit 2
fi

experiment=$1
shift

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

exec uv run --extra video --extra wandb python train.py \
  +experiment="$experiment" \
  "$@"
