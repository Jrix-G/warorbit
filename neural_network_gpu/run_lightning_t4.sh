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

RUN_NAME="gpu_2p_lightning_t4_guarded"
SOURCE_RUN_NAME="gpu_2p_rg_nosupport_local"
SOURCE_RUN_DIR="runs/${SOURCE_RUN_NAME}"
SOURCE_BEST="${SOURCE_RUN_DIR}/best_validated.npz"
SOURCE_LATEST="${SOURCE_RUN_DIR}/latest.npz"
CURRENT_RUN_DIR="runs/${RUN_NAME}"
CURRENT_LATEST="${CURRENT_RUN_DIR}/latest.npz"

RESUME_ARGS=()
TEACHER_CHECKPOINT=""

if [[ -f "$CURRENT_LATEST" ]]; then
  RESUME_ARGS=(--resume-checkpoint "$CURRENT_LATEST")
  TEACHER_CHECKPOINT="$CURRENT_LATEST"
  echo "Resuming current Lightning run from $CURRENT_LATEST"
elif [[ -f "$SOURCE_BEST" ]]; then
  RESUME_ARGS=(--resume-checkpoint "$SOURCE_BEST")
  TEACHER_CHECKPOINT="$SOURCE_BEST"
  echo "Starting Lightning run from source best $SOURCE_BEST"
elif [[ -f "$SOURCE_LATEST" ]]; then
  RESUME_ARGS=(--resume-checkpoint "$SOURCE_LATEST")
  TEACHER_CHECKPOINT="$SOURCE_LATEST"
  echo "Source best missing; starting from source latest $SOURCE_LATEST"
else
  echo "No checkpoint found; starting Lightning run from scratch"
fi

if [[ -n "$TEACHER_CHECKPOINT" ]]; then
  TEACHER_ARGS=(--teacher-checkpoint "$TEACHER_CHECKPOINT" --teacher-kl-coef 0.020)
else
  TEACHER_ARGS=(--teacher-kl-coef 0.000)
fi

python "$RUNNER_PATH" \
  "${RESUME_ARGS[@]}" \
  "${TEACHER_ARGS[@]}" \
  --run-name "$RUN_NAME" \
  --device cuda \
  --duration-minutes 540 \
  --workers 4 \
  --train-every 32 \
  --eval-every 512 \
  --eval-episodes 32 \
  --batch-size 32 \
  --batch-timeout 0.010 \
  --ppo-minibatch-size 64 \
  --learning-rate 0.000025 \
  --min-lr 0.000010 \
  --max-lr 0.000050 \
  --ppo-epochs 3 \
  --n-players 2 \
  --simple-opponents "random,random,greedy,greedy,greedy,starter,starter,starter,distance,distance,distance" \
  --eval-opponents "random,greedy,starter,distance" \
  --auto-tune-training \
  --policy-prior-strength 0.10 \
  --train-mission-mix-bonus-coef 0.16 \
  --train-target-support-ratio 0.30 \
  --train-support-ratio-band 0.20 \
  --train-min-support-ratio 0.12 \
  --train-max-attack-ratio 0.58 \
  --train-mission-mix-reward-clip 0.20 \
  --target-winrate 0.80 \
  --max-eval-do-nothing-rate 0.74 \
  --rollback-on-noop-rate 0.84 \
  --min-eval-avg-ships-sent 4.0 \
  --degenerate-noop-rate 0.90 \
  --degenerate-max-avg-ships-sent 1.75 \
  --degenerate-min-winrate 0.65 \
  --rollback-margin 0.22 \
  --max-opponent-regression 0.16 \
  --min-ci-promotion-games 96 \
  --stabilizer-target-noop 0.68 \
  --stabilizer-target-passivity 0.68 \
  --stabilizer-target-real-moves-turn 1.10 \
  --stabilizer-target-avg-ships-sent 10.0 \
  --stabilizer-min-weighted-score 0.58 \
  --stabilizer-max-ship-bonus 0.80 \
  --stabilizer-max-min-ships 8 \
  --stabilizer-max-starter-count 8 \
  --stabilizer-ratio-floor-step 0.06 \
  --stabilizer-ratio-floor-cap 0.65 \
  --supervisor-max-avg-ships-sent 24.0 \
  --supervisor-ratio-ceiling-step 0.08 \
  --supervisor-ratio-ceiling-floor 0.65
