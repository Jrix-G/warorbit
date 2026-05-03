#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="VPS/logs"
EVAL_DIR="VPS/evaluations"
mkdir -p "$LOG_DIR" "$EVAL_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/v9_vps_10h_${STAMP}.log"
PIDFILE="$LOG_DIR/v9_vps_10h_${STAMP}.pid"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [[ -n "${VPS_MAX_VMEM_KB:-}" ]]; then
  ulimit -Sv "$VPS_MAX_VMEM_KB"
fi

PYTHON="${PYTHON:-python3}"
RUNNER=(
  "$PYTHON"
  ./run_v9.py
  --minutes 600
  --hard-timeout-minutes 600
  --train-only
  --skip-eval
  --workers 1
  --pairs 2
  --games-per-eval 1
  --eval-every 0
  --benchmark-every 0
  --max-steps 100
  --four-player-ratio 0.50
  --train-search-width 2
  --train-simulation-depth 0
  --train-simulation-rollouts 0
  --front-lock-turns 15
  --pool-limit 8
  --checkpoint "$EVAL_DIR/v9_vps_10h_latest.npz"
  --best-checkpoint "$EVAL_DIR/v9_vps_10h_best.npz"
  --export-checkpoint "$EVAL_DIR/v9_vps_10h_policy.npz"
  --log-jsonl "$EVAL_DIR/v9_vps_10h_${STAMP}.jsonl"
  --train-opponents
  random noisy_greedy starter distance sun_dodge structured orbit_stars bot_v7
)

if ! command -v cpulimit >/dev/null 2>&1; then
  echo "[VPS] cpulimit is required for the 80% CPU cap."
  echo "[VPS] install it with: sudo apt-get update && sudo apt-get install -y cpulimit"
  exit 1
fi

RUNNER=(cpulimit -f -l 80 -- "${RUNNER[@]}")

echo "[VPS] starting V9 | log=$LOG | pidfile=$PIDFILE"
nohup nice -n 10 "${RUNNER[@]}" >"$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "[VPS] pid=$(cat "$PIDFILE")"
echo "[VPS] log=$LOG"
echo "[VPS] tail -f $LOG"
