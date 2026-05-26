#!/usr/bin/env bash
# Kaggle T4 — PPO training with v15 in pool.
# Usage inside Kaggle notebook:
#   cp -r /kaggle/input/warorbit /kaggle/working/warorbit
#   bash /kaggle/working/warorbit/neural_network_gpu/run_kaggle_v15_climb.sh
set -euo pipefail

REPO=/kaggle/working/warorbit
RUNS=/kaggle/working/runs
PYTHON="$REPO/.venv/bin/python"

# Fall back to system python if no venv (Kaggle already has torch installed)
if [[ ! -f "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

export PYTHONPATH="$REPO:$REPO/neural_network_gpu/kaggle_submission_stage"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd "$REPO"

echo "[kaggle] python = $PYTHON"
echo "[kaggle] PYTHONPATH = $PYTHONPATH"
$PYTHON -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
$PYTHON -c "from neural_network.src.model import NeuralNetworkModel; print('imports OK')"

# --- checkpoint selection ---
# Priority: 1) attached dataset checkpoint  2) overnight_v2 in repo  3) scratch
RESUME=""
for candidate in \
  /kaggle/input/warorbit-checkpoints/best_validated.npz \
  /kaggle/input/warorbit-checkpoints/gpu_2p_overnight_v2_best_validated.npz \
  "$REPO/runs/gpu_2p_overnight_v2/best_validated.npz" \
  "$REPO/runs/gpu_2p_overnight_9h/best_validated.npz"; do
  if [[ -f "$candidate" ]]; then
    RESUME="$candidate"
    echo "[kaggle] resuming from $RESUME"
    break
  fi
done

RESUME_ARGS=()
if [[ -n "$RESUME" ]]; then
  RESUME_ARGS=(--resume-checkpoint "$RESUME" --reset-shaping-coefs)
fi

RUN_NAME="kaggle_v15_climb_$(date -u +%Y%m%d_%H%M%S)"

$PYTHON neural_network_gpu/scripts/run_gpu.py \
  "${RESUME_ARGS[@]}" \
  --run-name "$RUN_NAME" \
  --runs-root "$RUNS" \
  --device cuda \
  --duration-minutes 690 \
  --workers 12 \
  --train-every 32 \
  --eval-every 256 \
  --eval-episodes 32 \
  --batch-size 64 \
  --batch-timeout 0.010 \
  --ppo-minibatch-size 256 \
  --learning-rate 0.00005 \
  --min-lr 0.00001 \
  --max-lr 0.00010 \
  --ppo-epochs 3 \
  --entropy-coef-start 0.02 \
  --n-players 2 \
  --simple-opponents "random,greedy,greedy,starter,starter,distance,distance,v15" \
  --eval-opponents "random,greedy,starter,distance,v15" \
  --auto-tune-training \
  --policy-prior-strength 0.20 \
  --rollback-margin -0.10 \
  --max-opponent-regression 0.25 \
  --dense-planet-coef 0.15 \
  --dense-production-coef 0.12 \
  --dense-ship-share-coef 0.14 \
  --dense-reward-clip 0.50 \
  --target-winrate 0.85 \
  --train-target-legal-noop-rate 0.15 \
  --max-eval-legal-noop-rate 0.35 \
  --valid-win-max-legal-noop-rate 0.35 \
  --passive-win-legal-noop-rate 0.35 \
  --per-step-legal-noop-penalty 0.012 \
  --per-step-real-action-bonus 0.001 \
  --valid-win-min-avg-ships-sent 4.0 \
  --valid-win-min-real-moves-turn 0.90 \
  --degenerate-noop-rate 0.90 \
  --degenerate-max-avg-ships-sent 1.50 \
  --collapse-stop-evals 3 \
  --collapse-max-recoveries 4 \
  --max-consecutive-regressions 5 \
  --stabilizer-min-weighted-score 0.35 \
  --min-ci-promotion-games 64 \
  2>&1 | tee "$RUNS/kaggle_v15_climb.log"

echo "[kaggle] run finished. Saving outputs..."
ls -lh "$RUNS/$RUN_NAME/" 2>/dev/null || true
