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

RUN_NAME="${RUN_NAME:-gpu_2p_lightning_counterfactual_10h}"
SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-gpu_2p_lightning_nn_dagger_10h}"
SOURCE_RUN_DIR="runs/${SOURCE_RUN_NAME}"
FALLBACK_RUN_DIR="runs/gpu_2p_lightning_event_10h"
CURRENT_RUN_DIR="runs/${RUN_NAME}"
CURRENT_LATEST="${CURRENT_RUN_DIR}/latest.npz"

WORKERS="${WORKERS:-16}"
EVAL_EVERY="${EVAL_EVERY:-512}"
EVAL_EPISODES="${EVAL_EPISODES:-12}"
EVAL_OPPONENTS="${EVAL_OPPONENTS:-random,greedy}"
BEST_EVAL_EVERY="${BEST_EVAL_EVERY:-2}"
ON_POLICY_IMITATION_COEF="${ON_POLICY_IMITATION_COEF:-0.025}"
COUNTERFACTUAL_IMITATION_COEF="${COUNTERFACTUAL_IMITATION_COEF:-0.085}"
COUNTERFACTUAL_TEMPERATURE="${COUNTERFACTUAL_TEMPERATURE:-0.75}"
COUNTERFACTUAL_MIN_MARGIN="${COUNTERFACTUAL_MIN_MARGIN:-0.05}"
COUNTERFACTUAL_MAX_WEIGHT="${COUNTERFACTUAL_MAX_WEIGHT:-2.5}"

RESUME_ARGS=()
if [[ -f "$CURRENT_LATEST" ]]; then
  RESUME_ARGS=(--resume-checkpoint "$CURRENT_LATEST")
  echo "Resuming counterfactual NN run from $CURRENT_LATEST"
elif [[ -f "$SOURCE_RUN_DIR/best_validated.npz" ]]; then
  RESUME_ARGS=(--resume-checkpoint "$SOURCE_RUN_DIR/best_validated.npz")
  echo "Starting counterfactual NN run from source best $SOURCE_RUN_DIR/best_validated.npz"
elif [[ -f "$SOURCE_RUN_DIR/latest.npz" ]]; then
  RESUME_ARGS=(--resume-checkpoint "$SOURCE_RUN_DIR/latest.npz")
  echo "Source best missing; starting counterfactual NN run from source latest $SOURCE_RUN_DIR/latest.npz"
elif [[ -f "$FALLBACK_RUN_DIR/best_validated.npz" ]]; then
  RESUME_ARGS=(--resume-checkpoint "$FALLBACK_RUN_DIR/best_validated.npz")
  echo "Starting counterfactual NN run from fallback best $FALLBACK_RUN_DIR/best_validated.npz"
else
  echo "No checkpoint found; starting counterfactual NN run from scratch"
fi

python "$RUNNER_PATH" \
  "${RESUME_ARGS[@]}" \
  --reset-shaping-coefs \
  --run-name "$RUN_NAME" \
  --device cuda \
  --duration-minutes 600 \
  --workers "$WORKERS" \
  --train-every 32 \
  --eval-every "$EVAL_EVERY" \
  --eval-episodes "$EVAL_EPISODES" \
  --best-eval-every "$BEST_EVAL_EVERY" \
  --batch-size 64 \
  --batch-timeout 0.010 \
  --ppo-minibatch-size 256 \
  --learning-rate 0.000018 \
  --min-lr 0.000006 \
  --max-lr 0.000032 \
  --ppo-epochs 2 \
  --on-policy-imitation-coef "$ON_POLICY_IMITATION_COEF" \
  --on-policy-imitation-min-margin 0.20 \
  --on-policy-imitation-max-weight 1.8 \
  --counterfactual-imitation-coef "$COUNTERFACTUAL_IMITATION_COEF" \
  --counterfactual-temperature "$COUNTERFACTUAL_TEMPERATURE" \
  --counterfactual-min-margin "$COUNTERFACTUAL_MIN_MARGIN" \
  --counterfactual-max-weight "$COUNTERFACTUAL_MAX_WEIGHT" \
  --counterfactual-selected-outcome-coef 1.0 \
  --counterfactual-selected-episode-coef 0.10 \
  --counterfactual-selected-step-shape-coef 0.30 \
  --counterfactual-prior-weight 0.30 \
  --counterfactual-tactical-weight 1.0 \
  --counterfactual-oracle-scale 0.45 \
  --n-players 2 \
  --max-actions-per-turn 4 \
  --min-expand-attack-ships 4 \
  --send-ratios "0.35,0.50,0.65,0.80,0.95" \
  --simple-opponents "random,greedy,greedy,greedy,greedy" \
  --eval-opponents "$EVAL_OPPONENTS" \
  --policy-prior-strength 0.05 \
  --entropy-coef-start 0.038 \
  --train-return-gamma 0.999 \
  --train-return-clip 1.5 \
  --event-capture-bonus 0.08 \
  --event-enemy-hit-bonus 0.008 \
  --event-hit-bonus 0.004 \
  --event-support-bonus 0.000 \
  --event-lost-penalty 0.080 \
  --event-pending-penalty 0.0 \
  --event-min-shape-clip 0.08 \
  --event-max-flat-action-bonus 0.0 \
  --event-max-ship-volume-bonus 0.0 \
  --event-max-activity-action-bonus 0.0 \
  --event-max-activity-ships-bonus 0.0 \
  --per-step-real-action-bonus 0.0 \
  --per-step-ship-volume-bonus 0.0 \
  --per-step-legal-noop-penalty 0.003 \
  --per-step-shape-clip 0.03 \
  --train-mission-mix-bonus-coef 0.0 \
  --target-winrate 0.80 \
  --max-eval-do-nothing-rate 0.88 \
  --valid-win-max-do-nothing-rate 0.90 \
  --rollback-on-noop-rate 0.0 \
  --min-eval-avg-ships-sent 3.0 \
  --valid-win-min-avg-ships-sent 3.0 \
  --valid-win-min-real-moves-turn 0.50 \
  --max-eval-legal-noop-rate 0.45 \
  --valid-win-max-legal-noop-rate 0.45 \
  --rollback-on-legal-noop-rate 0.0 \
  --train-target-legal-noop-rate 0.25 \
  --passive-win-legal-noop-rate 0.45 \
  --passive-win-terminal-reward -0.25 \
  --degenerate-noop-rate 0.92 \
  --degenerate-max-avg-ships-sent 1.75 \
  --degenerate-min-winrate 0.65 \
  --collapse-stop-evals 3 \
  --collapse-stop-noop-rate 0.94 \
  --collapse-stop-max-avg-ships-sent 1.75 \
  --collapse-max-recoveries 2 \
  --max-consecutive-regressions 4 \
  --rollback-margin 0.16 \
  --max-opponent-regression 0.20 \
  --min-ci-promotion-games 96 \
  --disable-behavior-supervisor
