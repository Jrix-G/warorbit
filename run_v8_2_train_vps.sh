#!/usr/bin/env bash
# V8.2 ranker — VPS 2 vCPU, 10h calibrated.
# Usage: ./run_v8_2_train_vps.sh [extra_train_args...]
set -euo pipefail
cd "$(dirname "$0")"

LOG_DIR="logs"
mkdir -p "$LOG_DIR" evaluations
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/v8_2_train_${STAMP}.log"

echo "[run_v8_2_train_vps] starting | log=$LOG"
python3 train_v8_2.py \
  --minutes 600 \
  --pairs 5 \
  --games-per-eval 3 \
  --eval-games 40 \
  --eval-every 5 \
  --max-steps 220 \
  --eval-max-steps 400 \
  --four-player-ratio 0.65 \
  --eval-four-player-ratio 0.70 \
  --workers 2 \
  --pool-limit 6 \
  --min-improvement 0.015 \
  --min-mode-floor 0.05 \
  --log-jsonl "evaluations/v8_2_train_${STAMP}.jsonl" \
  "$@" \
  2>&1 | tee "$LOG"
