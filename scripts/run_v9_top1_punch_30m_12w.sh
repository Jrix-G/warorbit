#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_NAME="${RUN_NAME:-v9_top1_punch_30m_12w}"
LATEST="evaluations/${RUN_NAME}_latest.npz"
BEST="evaluations/${RUN_NAME}_best.npz"
POLICY="evaluations/${RUN_NAME}_policy.npz"
LOG="evaluations/${RUN_NAME}.console.log"
TRAIN_LOG="evaluations/${RUN_NAME}_train.jsonl"
SNAPSHOTS="evaluations/${RUN_NAME}_snapshots"

mkdir -p evaluations

rm -f "$LATEST" "$BEST" "$POLICY" "$LOG" "$TRAIN_LOG"
rm -rf "$SNAPSHOTS"

python3 run_v9.py \
  --game-engine official_fast \
  --minutes 30 \
  --hard-timeout-minutes 30 \
  --workers 12 \
  --pairs 4 \
  --games-per-eval 2 \
  --eval-games 8 \
  --benchmark-games 16 \
  --min-promotion-benchmark-games 16 \
  --benchmark-progress-every 4 \
  --eval-every 1 \
  --benchmark-every 1 \
  --max-steps 120 \
  --eval-max-steps 180 \
  --four-player-ratio 0.80 \
  --eval-four-player-ratio 0.80 \
  --benchmark-four-player-ratio 0.80 \
  --train-search-width 3 \
  --train-simulation-depth 0 \
  --train-simulation-rollouts 0 \
  --train-opponent-samples 1 \
  --opening-punch-turns 55 \
  --opening-min-capture-send-2p 14 \
  --opening-min-capture-send-4p 16 \
  --midgame-min-capture-send-4p 24 \
  --capture-garrison-margin 0.22 \
  --capture-target-ship-margin 0.15 \
  --midgame-capture-target-margin-4p 0.35 \
  --opening-close-neutral-dist-4p 42.0 \
  --opening-long-attack-risk-dist-4p 55.0 \
  --opening-source-commit-frac 1.0 \
  --front-lock-turns 24 \
  --target-active-fronts 2.7 \
  --front-penalty-weight 0.055 \
  --front-penalty-cap 0.12 \
  --front-ok-bonus 0.045 \
  --front-partial-bonus 0.025 \
  --backbone-penalty-weight 0.080 \
  --backbone-bonus-weight 0.060 \
  --front-pressure-plan-bias 0.12 \
  --front-pressure-attack-penalty 0.12 \
  --guardian-enabled 1 \
  --guardian-min-benchmark-4p 0.42 \
  --guardian-min-benchmark-backbone 0.08 \
  --guardian-max-benchmark-fronts 2.70 \
  --guardian-max-generalization-gap 0.18 \
  --min-benchmark-score 0.35 \
  --max-generalization-gap 0.18 \
  --reward-noise 0.008 \
  --snapshot-every 1 \
  --snapshot-dir "$SNAPSHOTS" \
  --checkpoint "$LATEST" \
  --best-checkpoint "$BEST" \
  --export-checkpoint "$POLICY" \
  --log-jsonl "$TRAIN_LOG" \
  --no-resume \
  2>&1 | tee "$LOG"
