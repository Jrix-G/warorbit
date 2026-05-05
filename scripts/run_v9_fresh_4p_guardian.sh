#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_NAME="${RUN_NAME:-v9_fresh_4p_guardian}"
LATEST="evaluations/${RUN_NAME}_latest.npz"
BEST="evaluations/${RUN_NAME}_best.npz"
POLICY="evaluations/${RUN_NAME}_policy.npz"
LOG="evaluations/${RUN_NAME}_train.jsonl"

mkdir -p evaluations

# Force a true fresh start for this run namespace.
rm -f "$LATEST" "$BEST" "$POLICY" "$LOG"

echo "Fresh V9 run"
echo "Run: $RUN_NAME"
echo "Latest: $LATEST"
echo "Best: $BEST"
echo "Policy: $POLICY"
echo "Log: $LOG"

python3 run_v9.py \
  --game-engine official_fast \
  --minutes 480 \
  --hard-timeout-minutes 480 \
  --workers 8 \
  --pairs 4 \
  --games-per-eval 2 \
  --eval-games 16 \
  --benchmark-games 32 \
  --min-promotion-benchmark-games 32 \
  --benchmark-progress-every 2 \
  --eval-every 1 \
  --benchmark-every 1 \
  --max-steps 160 \
  --eval-max-steps 220 \
  --four-player-ratio 1.0 \
  --eval-four-player-ratio 1.0 \
  --benchmark-four-player-ratio 1.0 \
  --four-p-signal-boost 1.4 \
  --train-search-width 3 \
  --train-simulation-depth 0 \
  --train-simulation-rollouts 0 \
  --train-opponent-samples 1 \
  --pool-limit 15 \
  --guardian-enabled 1 \
  --guardian-min-benchmark-4p 0.42 \
  --guardian-min-benchmark-backbone 0.08 \
  --guardian-max-benchmark-fronts 2.70 \
  --guardian-max-generalization-gap 0.18 \
  --min-benchmark-score 0.35 \
  --max-generalization-gap 0.18 \
  --reward-noise 0.008 \
  --checkpoint "$LATEST" \
  --best-checkpoint "$BEST" \
  --export-checkpoint "$POLICY" \
  --log-jsonl "$LOG" \
  --no-resume
