#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
RUNNER_PATH=""

if [[ -f "$REPO_ROOT/../neural_network/src/model.py" && -f "$REPO_ROOT/scripts/run_gpu.py" ]]; then
  REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
  RUNNER_PATH="$REPO_ROOT/neural_network_gpu/scripts/run_gpu.py"
elif [[ -f "$REPO_ROOT/neural_network_gpu/scripts/run_gpu.py" ]]; then
  REPO_ROOT="$(cd "$REPO_ROOT/neural_network_gpu/.." && pwd)"
  RUNNER_PATH="$REPO_ROOT/neural_network_gpu/scripts/run_gpu.py"
elif [[ -f "$REPO_ROOT/scripts/run_gpu.py" ]]; then
  RUNNER_PATH="$REPO_ROOT/scripts/run_gpu.py"
else
  echo "Could not locate scripts/run_gpu.py from: $SCRIPT_DIR" >&2
  exit 1
fi

cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

python - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print(f"torch import failed: {exc}", file=sys.stderr)
    raise SystemExit(1)

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", getattr(torch.version, "cuda", None))
if not torch.cuda.is_available():
    raise SystemExit(1)
PY

RUN_NAME="gpu_2p_lightning_event_10h"
SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-gpu_2p_lightning_guarded_v2}"
SOURCE_RUN_DIR="runs/${SOURCE_RUN_NAME}"
SOURCE_BEST="${SOURCE_RUN_DIR}/best_validated.npz"
SOURCE_LATEST="${SOURCE_RUN_DIR}/latest.npz"
CURRENT_RUN_DIR="runs/${RUN_NAME}"
CURRENT_LATEST="${CURRENT_RUN_DIR}/latest.npz"

RESUME_ARGS=()
TEACHER_ARGS=(--teacher-kl-coef 0.010)

if [[ -f "$CURRENT_LATEST" ]]; then
  RESUME_ARGS=(--resume-checkpoint "$CURRENT_LATEST")
  TEACHER_ARGS=(--teacher-checkpoint "$CURRENT_LATEST" --teacher-kl-coef 0.010)
  echo "Resuming current event run from $CURRENT_LATEST"
elif [[ -f "$SOURCE_BEST" ]]; then
  RESUME_ARGS=(--resume-checkpoint "$SOURCE_BEST")
  TEACHER_ARGS=(--teacher-checkpoint "$SOURCE_BEST" --teacher-kl-coef 0.010)
  echo "Starting event run from source best $SOURCE_BEST"
elif [[ -f "$SOURCE_LATEST" ]]; then
  RESUME_ARGS=(--resume-checkpoint "$SOURCE_LATEST")
  TEACHER_ARGS=(--teacher-checkpoint "$SOURCE_LATEST" --teacher-kl-coef 0.010)
  echo "Source best missing; starting event run from source latest $SOURCE_LATEST"
else
  echo "No checkpoint found; starting event run from scratch"
fi

python "$RUNNER_PATH" \
  "${RESUME_ARGS[@]}" \
  "${TEACHER_ARGS[@]}" \
  --run-name "$RUN_NAME" \
  --device cuda \
  --duration-minutes 600 \
  --workers 8 \
  --train-every 32 \
  --eval-every 512 \
  --eval-episodes 32 \
  --batch-size 48 \
  --batch-timeout 0.010 \
  --ppo-minibatch-size 128 \
  --learning-rate 0.000018 \
  --min-lr 0.000006 \
  --max-lr 0.000035 \
  --ppo-epochs 2 \
  --n-players 2 \
  --simple-opponents "random,random,greedy,greedy,greedy,starter,starter,distance,distance" \
  --eval-opponents "random,greedy,starter,distance" \
  --auto-tune-training \
  --policy-prior-strength 0.12 \
  --train-return-gamma 0.997 \
  --train-return-clip 2.0 \
  --event-capture-bonus 0.10 \
  --event-enemy-hit-bonus 0.045 \
  --event-hit-bonus 0.035 \
  --event-support-bonus 0.015 \
  --event-lost-penalty 0.045 \
  --event-pending-penalty 0.0 \
  --event-min-shape-clip 0.12 \
  --event-max-flat-action-bonus 0.002 \
  --event-max-ship-volume-bonus 0.0 \
  --event-max-activity-action-bonus 0.03 \
  --event-max-activity-ships-bonus 0.0 \
  --per-step-real-action-bonus 0.001 \
  --per-step-ship-volume-bonus 0.0 \
  --per-step-legal-noop-penalty 0.006 \
  --per-step-shape-clip 0.04 \
  --train-mission-mix-bonus-coef 0.04 \
  --train-target-support-ratio 0.25 \
  --train-support-ratio-band 0.20 \
  --train-min-support-ratio 0.08 \
  --train-max-attack-ratio 0.62 \
  --train-mission-mix-reward-clip 0.10 \
  --target-winrate 0.80 \
  --max-eval-do-nothing-rate 0.84 \
  --valid-win-max-do-nothing-rate 0.88 \
  --rollback-on-noop-rate 0.0 \
  --min-eval-avg-ships-sent 2.5 \
  --valid-win-min-avg-ships-sent 3.0 \
  --valid-win-min-real-moves-turn 0.75 \
  --max-eval-legal-noop-rate 0.35 \
  --valid-win-max-legal-noop-rate 0.35 \
  --rollback-on-legal-noop-rate 0.55 \
  --train-target-legal-noop-rate 0.20 \
  --passive-win-legal-noop-rate 0.35 \
  --passive-win-terminal-reward -0.25 \
  --degenerate-noop-rate 0.90 \
  --degenerate-max-avg-ships-sent 1.75 \
  --degenerate-min-winrate 0.65 \
  --collapse-stop-evals 3 \
  --collapse-stop-noop-rate 0.92 \
  --collapse-stop-max-avg-ships-sent 1.75 \
  --collapse-max-recoveries 4 \
  --max-consecutive-regressions 5 \
  --rollback-margin 0.16 \
  --max-opponent-regression 0.18 \
  --min-ci-promotion-games 96 \
  --stabilizer-target-legal-noop 0.20 \
  --stabilizer-target-legal-passivity 0.20 \
  --stabilizer-target-real-moves-turn 1.00 \
  --stabilizer-target-avg-ships-sent 5.0 \
  --stabilizer-min-weighted-score 0.40 \
  --stabilizer-max-ship-bonus 0.35 \
  --stabilizer-max-min-ships 6 \
  --stabilizer-max-starter-count 6 \
  --stabilizer-ratio-floor-step 0.04 \
  --stabilizer-ratio-floor-cap 0.55 \
  --supervisor-max-avg-ships-sent 18.0 \
  --supervisor-ratio-ceiling-step 0.06 \
  --supervisor-ratio-ceiling-floor 0.65
