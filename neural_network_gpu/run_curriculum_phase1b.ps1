$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")

$InitCheckpoint = Join-Path $RepoRoot.Path "runs\imitation_4p_top10_v1\bc_4p_top10_best.npz"
$CompatCheckpoint = Join-Path $RepoRoot.Path ".tmp\bc_4p_top10_best_compat.npz"
$TeacherCheckpoint = Join-Path $RepoRoot.Path "runs\gpu_2p_top1_distance_guarded_local_v2\best_validated.npz"
$RunName = "gpu_2p_curriculum_phase1b_random_only"

if (-not (Test-Path $InitCheckpoint)) {
  throw "Missing imitation checkpoint: $InitCheckpoint"
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
    "curriculum_phase": "phase1b_random_only",
})
dst.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(dst, **state)
"@ | python -

if (-not (Test-Path $CompatCheckpoint)) {
  throw "Failed to create compat checkpoint: $CompatCheckpoint"
}

# Phase 1b: 100% random only — agent must master random before seeing greedy.
# Fixes vs phase1: shaping coefs halved, teacher_kl x5, greedy removed from pool.
$TrainOpponents = "random,random,random,random,random,random,random,random,random,random"
$EvalOpponents = "random,greedy,starter,distance"
$UseTeacher = Test-Path $TeacherCheckpoint

Set-Location $RepoRoot
$env:PYTHONPATH = $RepoRoot.Path
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

python -c "import torch, sys; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', getattr(torch.version, 'cuda', None)); sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) {
  throw "PyTorch cannot see CUDA. Install a CUDA-enabled PyTorch build or run with --device cpu."
}

$Args = @(
  "neural_network_gpu\scripts\run_gpu.py",
  "--resume-checkpoint", $CompatCheckpoint,
  "--reset-shaping-coefs",
  "--device", "cuda",
  "--duration-minutes", "240",
  "--workers", "8",
  "--train-every", "32",
  "--eval-every", "512",
  "--eval-episodes", "32",
  "--batch-size", "48",
  "--batch-timeout", "0.010",
  "--ppo-minibatch-size", "128",
  "--learning-rate", "0.000025",
  "--min-lr", "0.000010",
  "--max-lr", "0.000040",
  "--ppo-epochs", "3",
  "--n-players", "2",
  "--simple-opponents", $TrainOpponents,
  "--eval-opponents", $EvalOpponents,
  "--policy-prior-strength", "0.08",
  "--train-mission-mix-bonus-coef", "0.14",
  "--train-target-support-ratio", "0.22",
  "--train-support-ratio-band", "0.18",
  "--train-min-support-ratio", "0.08",
  "--train-max-attack-ratio", "0.62",
  "--train-mission-mix-reward-clip", "0.18",
  "--target-winrate", "0.80",

  # FIX #1: Per-step shaping coefs halved vs phase1.
  # Rationale: step_shape was +1.0-1.4 while terminal was -0.5 -> reward hacking.
  # The shaping must not overpower the win/loss signal.
  "--train-target-legal-noop-rate", "0.20",
  "--max-eval-legal-noop-rate", "0.35",
  "--valid-win-max-legal-noop-rate", "0.35",
  "--passive-win-legal-noop-rate", "0.35",
  "--rollback-on-legal-noop-rate", "0.50",
  "--stabilizer-target-legal-noop", "0.20",
  "--stabilizer-target-legal-passivity", "0.20",
  "--per-step-legal-noop-penalty", "0.006",
  "--per-step-real-action-bonus", "0.004",
  "--per-step-shape-clip", "0.02",

  # Raw no-op backstops stay loose.
  "--max-eval-do-nothing-rate", "0.86",
  "--valid-win-max-do-nothing-rate", "0.90",
  "--rollback-on-noop-rate", "0.0",

  # Anti-exploit gates (unchanged from phase1).
  "--min-eval-avg-ships-sent", "2.5",
  "--valid-win-min-avg-ships-sent", "2.5",
  "--valid-win-min-real-moves-turn", "0.85",
  "--passive-win-terminal-reward", "-0.25",
  "--degenerate-noop-rate", "0.90",
  "--degenerate-max-avg-ships-sent", "1.50",
  "--degenerate-min-winrate", "0.60",
  "--collapse-stop-evals", "2",
  "--collapse-stop-noop-rate", "0.92",
  "--collapse-stop-max-avg-ships-sent", "1.75",

  # Promotion/regression guard.
  "--rollback-margin", "0.12",
  "--max-opponent-regression", "0.25",
  "--min-ci-promotion-games", "96",
  "--stabilizer-target-real-moves-turn", "1.05",
  "--stabilizer-target-avg-ships-sent", "5.0",
  "--stabilizer-min-weighted-score", "0.35",
  "--stabilizer-max-ship-bonus", "0.35",
  "--stabilizer-max-min-ships", "5",
  "--stabilizer-max-starter-count", "0",
  "--stabilizer-ratio-floor-step", "0.04",
  "--stabilizer-ratio-floor-cap", "0.55",
  "--supervisor-max-avg-ships-sent", "18.0",
  "--supervisor-ratio-ceiling-step", "0.06",
  "--supervisor-ratio-ceiling-floor", "0.65",
  "--run-name", $RunName
)

# FIX #2: teacher_kl_coef x5 (0.010 -> 0.050) to re-anchor policy to BC.
# In phase1 the BC signal was erased in <1h (teacher_kl rose to 0.48).
if ($UseTeacher) {
  $Args += @("--teacher-checkpoint", $TeacherCheckpoint, "--teacher-kl-coef", "0.050")
} else {
  Write-Host "Teacher checkpoint not found; running without KL distillation: $TeacherCheckpoint"
}

python @Args
