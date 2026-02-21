#!/usr/bin/env bash
# Usage:
#   scripts/launch_detached.sh <run_name> -- <python args...>
# Example:
#   scripts/launch_detached.sh cartpole_run -- --episodes 500 --surrogate-topk 2

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <run_name> -- <python args...>"
  exit 1
fi

RUN_NAME="$1"
shift
if [ "$1" != "--" ]; then
  echo "Second argument must be --"
  exit 1
fi
shift

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p figures

LOG="figures/${RUN_NAME}.log"
CMD=(python3 -u scripts/cartpole_lut_dqn_v4.py --run-name "$RUN_NAME" --figures-dir figures "$@")

echo "Launching: ${CMD[*]}"
echo "Log: $LOG"

echo "[$(date '+%F %T')] START ${RUN_NAME}" >> "$LOG"
nohup env PYTHONPATH=. "${CMD[@]}" >> "$LOG" 2>&1 &
PID=$!

echo "PID=$PID"
echo "$PID" > "figures/${RUN_NAME}.pid"
echo "[$(date '+%F %T')] PID ${PID}" >> "$LOG"
