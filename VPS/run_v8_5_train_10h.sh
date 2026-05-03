#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="VPS/logs"
EVAL_DIR="VPS/evaluations"
mkdir -p "$LOG_DIR" "$EVAL_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/v8_5_vps_10h_${STAMP}.log"
PIDFILE="$LOG_DIR/v8_5_vps_10h_${STAMP}.pid"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-python3}"
RUNNER=(
  "$PYTHON"
  ./train_v8_5.py
  --minutes 600
  --pairs 2
  --games-per-eval 1
  --eval-games 6
  --eval-every 5
  --max-steps 160
  --eval-max-steps 220
  --four-player-ratio 0.50
  --eval-four-player-ratio 0.50
  --sigma 0.07
  --lr 0.020
  --l2 0.0005
  --workers 1
  --pool-limit 3
  --min-improvement 0.015
  --min-mode-floor 0.05
  --checkpoint "$EVAL_DIR/v8_5_ranker_train_latest.npz"
  --best-checkpoint "$EVAL_DIR/v8_5_ranker_train_best.npz"
  --export-bot-checkpoint "$EVAL_DIR/v8_5_ranker.npz"
  --log-jsonl "$EVAL_DIR/v8_5_train_${STAMP}.jsonl"
)

if ! command -v cpulimit >/dev/null 2>&1; then
  echo "[VPS] cpulimit is required for the 80% CPU cap."
  echo "[VPS] install it with: sudo apt-get update && sudo apt-get install -y cpulimit"
  exit 1
fi

RUNNER=(cpulimit -f -l 80 -- "${RUNNER[@]}")

echo "[VPS] starting | log=$LOG | pidfile=$PIDFILE"
nohup nice -n 10 "${RUNNER[@]}" >"$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "[VPS] pid=$(cat "$PIDFILE")"
echo "[VPS] log=$LOG"
echo "[VPS] tail -f $LOG"
