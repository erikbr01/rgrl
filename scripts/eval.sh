#!/usr/bin/env bash
# Thin wrapper so eval.py can be invoked from anywhere:
#   scripts/eval.sh ckpt=runs/x/latest.pt inference=diffusion_dp env=robomimic_tool_hang
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec uv run python eval.py "$@"
