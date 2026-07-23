#!/usr/bin/env bash
# Thin wrapper so train.py can be invoked from anywhere:
#   scripts/train.sh data=tool_hang_state target=tool_hang_state
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec uv run python train.py "$@"
