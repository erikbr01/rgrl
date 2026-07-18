#!/usr/bin/env bash
# Auto-restart watchdog for scripts/train_prepare_tray_dit.sh. Training
# auto-resumes from the last checkpoint in the same out dir (see
# rgrl/train_loop.py's resume logic), so a crash-and-restart continues
# from where it left off rather than discarding progress.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG=runs/prepare_tray_baseline_dit.log
attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "=== watchdog: attempt $attempt $(date -Iseconds) ===" >>"$LOG"
  ./scripts/train_prepare_tray_dit.sh >>"$LOG" 2>&1
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "=== watchdog: training finished successfully ===" >>"$LOG"
    break
  fi
  echo "=== watchdog: exited with code $code, restarting in 30s ===" >>"$LOG"
  sleep 30
done
