$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")

$InitCheckpoint = Join-Path $RepoRoot.Path "runs\imitation_4p_top10_v1\bc_4p_top10_best.npz"
$CompatCheckpoint = Join-Path $RepoRoot.Path ".tmp\bc_4p_top10_best_compat.npz"
$TeacherCheckpoint = Join-Path $RepoRoot.Path "runs\gpu_2p_top1_distance_guarded_local_v2\best_validated.npz"
$RunName = "gpu_2p_from_imitation_4p_local"

if (-not (Test-Path $InitCheckpoint)) {
  throw "Missing imitation checkpoint: $InitCheckpoint"
}
if (-not (Test-Path $TeacherCheckpoint)) {
  throw "Missing teacher checkpoint: $TeacherCheckpoint"
}

@"
import json
from pathlib import Path
import numpy as np

src = Path(r"$InitCheckpoint")
dst = Path(r"$CompatCheckpoint")
data = np.load(src, allow_pickle=False)
state = {k: data[k] for k in data.files}
state["metadata"] = json.dumps({
    "source": "imitation_4p_top10",
    "base_checkpoint": str(src),
    "compat_mode": "metadata_added_for_run_gpu_resume",
})
dst.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(dst, **state)
"@ | python -

if (-not (Test-Path $CompatCheckpoint)) {
  throw "Failed to create compat checkpoint: $CompatCheckpoint"
}

$TrainOpponents = "random,random,greedy,greedy,greedy,greedy,greedy,greedy,starter,starter,starter,starter,starter,starter,starter,starter,distance,distance,distance,distance"
$EvalOpponents = "random,greedy,starter,distance"

Set-Location $RepoRoot
$env:PYTHONPATH = $RepoRoot.Path
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

python -c "import torch, sys; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', getattr(torch.version, 'cuda', None)); sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) {
  throw "PyTorch cannot see CUDA. Install a CUDA-enabled PyTorch build or run with --device cpu."
}

python neural_network_gpu\scripts\run_gpu.py `
  --resume-checkpoint $CompatCheckpoint `
  --teacher-checkpoint $TeacherCheckpoint `
  --teacher-kl-coef 0.020 `
  --device cuda `
  --duration-minutes 540 `
  --workers 8 `
  --train-every 32 `
  --eval-every 512 `
  --eval-episodes 32 `
  --batch-size 48 `
  --batch-timeout 0.010 `
  --ppo-minibatch-size 128 `
  --learning-rate 0.000025 `
  --min-lr 0.000010 `
  --max-lr 0.000050 `
  --ppo-epochs 3 `
  --n-players 2 `
  --simple-opponents $TrainOpponents `
  --eval-opponents $EvalOpponents `
  --auto-tune-training `
  --policy-prior-strength 0.10 `
  --train-mission-mix-bonus-coef 0.16 `
  --train-target-support-ratio 0.30 `
  --train-support-ratio-band 0.20 `
  --train-min-support-ratio 0.12 `
  --train-max-attack-ratio 0.58 `
  --train-mission-mix-reward-clip 0.20 `
  --target-winrate 0.80 `
  --max-eval-do-nothing-rate 0.74 `
  --rollback-on-noop-rate 0.84 `
  --min-eval-avg-ships-sent 4.0 `
  --valid-win-max-do-nothing-rate 0.84 `
  --valid-win-min-avg-ships-sent 4.0 `
  --valid-win-min-real-moves-turn 1.0 `
  --passive-win-terminal-reward -0.25 `
  --degenerate-noop-rate 0.90 `
  --degenerate-max-avg-ships-sent 1.75 `
  --degenerate-min-winrate 0.65 `
  --collapse-stop-evals 2 `
  --collapse-stop-noop-rate 0.92 `
  --collapse-stop-max-avg-ships-sent 2.0 `
  --rollback-margin 0.22 `
  --max-opponent-regression 0.16 `
  --min-ci-promotion-games 96 `
  --stabilizer-target-noop 0.68 `
  --stabilizer-target-passivity 0.40 `
  --stabilizer-target-real-moves-turn 1.10 `
  --stabilizer-target-avg-ships-sent 10.0 `
  --stabilizer-min-weighted-score 0.58 `
  --stabilizer-max-ship-bonus 0.80 `
  --stabilizer-max-min-ships 8 `
  --stabilizer-max-starter-count 8 `
  --stabilizer-ratio-floor-step 0.06 `
  --stabilizer-ratio-floor-cap 0.65 `
  --supervisor-max-avg-ships-sent 24.0 `
  --supervisor-ratio-ceiling-step 0.08 `
  --supervisor-ratio-ceiling-floor 0.65 `
  --run-name $RunName
