#!/usr/bin/env bash
# Auto-restart watchdog for scripts/train_set_table_plusright_dit.sh -- same
# pattern as scripts/watch_set_table_dit.sh. Training auto-resumes from the
# last checkpoint in the same out dir (see rgrl/train_loop.py's resume
# logic), so a crash-and-restart continues from where it left off rather
# than discarding progress.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG=runs/set_table_plusright_dit.log
attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "=== watchdog: attempt $attempt $(date -Iseconds) ===" >>"$LOG"
  ./scripts/train_set_table_plusright_dit.sh >>"$LOG" 2>&1
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "=== watchdog: training finished successfully ===" >>"$LOG"
    break
  fi
  echo "=== watchdog: exited with code $code, restarting in 30s ===" >>"$LOG"
  sleep 30
done
