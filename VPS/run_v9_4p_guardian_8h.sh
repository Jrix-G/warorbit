#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="VPS/logs"
EVAL_DIR="VPS/evaluations"
mkdir -p "$LOG_DIR" "$EVAL_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/v9_4p_guardian_8h_${STAMP}.log"
PIDFILE="$LOG_DIR/v9_4p_guardian_8h_${STAMP}.pid"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-python3}"
RUNNER=(
  "$PYTHON"
  ./run_v9.py
  --minutes 480
  --hard-timeout-minutes 480
  --workers 4
  --pairs 6
  --games-per-eval 3
  --eval-games 12
  --benchmark-games 24
  --min-promotion-benchmark-games 24
  --benchmark-progress-every 4
  --eval-every 1
  --benchmark-every 1
  --max-steps 120
  --eval-max-steps 220
  --four-player-ratio 1.0
  --eval-four-player-ratio 1.0
  --benchmark-four-player-ratio 1.0
  --train-search-width 3
  --train-simulation-depth 0
  --train-simulation-rollouts 0
  --train-opponent-samples 1
  --front-lock-turns 22
  --target-active-fronts 2.0
  --target-backbone-turn-frac 0.15
  --front-penalty-weight 0.055
  --front-penalty-cap 0.12
  --front-ok-bonus 0.070
  --front-partial-bonus 0.035
  --backbone-penalty-weight 0.120
  --backbone-bonus-weight 0.100
  --front-pressure-plan-bias 0.16
  --front-pressure-attack-penalty 0.14
  --guardian-enabled 1
  --guardian-min-benchmark-4p 0.42
  --guardian-min-benchmark-backbone 0.08
  --guardian-max-benchmark-fronts 2.70
  --guardian-max-generalization-gap 0.18
  --export-best-on-finish 1
  --min-benchmark-score 0.35
  --max-generalization-gap 0.18
  --exploration-rate 0.08
  --reward-noise 0.008
  --pool-limit 15
  --checkpoint "$EVAL_DIR/v9_4p_guardian_8h_latest.npz"
  --best-checkpoint "$EVAL_DIR/v9_4p_guardian_8h_best.npz"
  --export-checkpoint "$EVAL_DIR/v9_4p_guardian_8h_policy.npz"
  --log-jsonl "$EVAL_DIR/v9_4p_guardian_8h_${STAMP}.jsonl"
  --train-opponents
  random noisy_greedy starter distance sun_dodge structured orbit_stars bot_v7
  notebook_tactical_heuristic
  notebook_mdmahfuzsumon_how_my_ai_wins_space_wars
  notebook_sigmaborov_orbit_wars_2026_starter
  notebook_sigmaborov_orbit_wars_2026_tactical_heuristic
  notebook_orbitbotnext
  notebook_distance_prioritized
  notebook_physics_accurate
)

if command -v cpulimit >/dev/null 2>&1; then
  RUNNER=(cpulimit -f -l 85 -- "${RUNNER[@]}")
fi

echo "[VPS] starting V9 4p guardian | log=$LOG | pidfile=$PIDFILE"
nohup nice -n 10 "${RUNNER[@]}" >"$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "[VPS] pid=$(cat "$PIDFILE")"
echo "[VPS] log=$LOG"
echo "[VPS] tail -f $LOG"
