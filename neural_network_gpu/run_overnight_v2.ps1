$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")

# Resume from best_validated of overnight_9h (ep 512 model), else phase1b, else BC.
$PreviousRunBest = Join-Path $RepoRoot.Path "runs\gpu_2p_overnight_9h\best_validated.npz"
$Phase1bBest    = Join-Path $RepoRoot.Path "runs\gpu_2p_curriculum_phase1b_random_only\best_validated.npz"
$InitCheckpoint = Join-Path $RepoRoot.Path "runs\imitation_4p_top10_v1\bc_4p_top10_best.npz"
$CompatCheckpoint = Join-Path $RepoRoot.Path ".tmp\bc_4p_top10_best_compat_v2.npz"
$TeacherCheckpoint = Join-Path $RepoRoot.Path "runs\gpu_2p_top1_distance_guarded_local_v2\best_validated.npz"
$RunName = "gpu_2p_overnight_v2"

if (Test-Path $PreviousRunBest) {
  Write-Host "Resuming from overnight_9h best: $PreviousRunBest"
  $ResumeCheckpoint = $PreviousRunBest
} elseif (Test-Path $Phase1bBest) {
  Write-Host "Resuming from phase1b best: $Phase1bBest"
  $ResumeCheckpoint = $Phase1bBest
} elseif (Test-Path $InitCheckpoint) {
  Write-Host "No previous best found. Creating compat BC checkpoint."
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
    "compat_mode": "metadata_added",
    "curriculum_phase": "overnight_v2",
})
dst.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(dst, **state)
"@ | python -
  if (-not (Test-Path $CompatCheckpoint)) { throw "Failed to create compat checkpoint." }
  $ResumeCheckpoint = $CompatCheckpoint
} else {
  throw "No checkpoint to start from."
}

# Train pool: 100% random (curriculum phase — master random before adding harder opponents)
$TrainOpponents = "random,random,random,random,random,random,random,greedy,greedy,greedy"
$EvalOpponents  = "random,greedy"
$UseTeacher = Test-Path $TeacherCheckpoint

Set-Location $RepoRoot
$env:PYTHONPATH = $RepoRoot.Path
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

python -c "import torch, sys; print('torch', torch.__version__); sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { throw "CUDA not available." }

$Args = @(
  "neural_network_gpu\scripts\run_gpu.py",
  "--resume-checkpoint", $ResumeCheckpoint,
  "--reset-shaping-coefs",
  "--auto-tune-training",
  "--device", "cuda",
  "--duration-minutes", "540",
  "--workers", "8",
  "--train-every", "32",
  "--eval-every", "512",
  "--eval-episodes", "32",
  "--batch-size", "48",
  "--batch-timeout", "0.010",
  "--ppo-minibatch-size", "128",
  "--learning-rate", "0.000025",
  "--min-lr", "0.000008",
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

  "--train-target-legal-noop-rate", "0.20",
  "--max-eval-legal-noop-rate", "0.35",
  "--valid-win-max-legal-noop-rate", "0.35",
  "--passive-win-legal-noop-rate", "0.35",
  "--rollback-on-legal-noop-rate", "0.50",
  "--stabilizer-target-legal-noop", "0.20",
  "--stabilizer-target-legal-passivity", "0.20",
  "--per-step-legal-noop-penalty", "0.0012",
  "--per-step-real-action-bonus", "0.0008",
  "--per-step-ship-volume-bonus", "0.0006",
  "--per-step-ship-volume-target", "8.0",
  "--per-step-shape-clip", "0.005",

  "--max-eval-do-nothing-rate", "0.86",
  "--valid-win-max-do-nothing-rate", "0.90",
  "--rollback-on-noop-rate", "0.0",

  "--min-eval-avg-ships-sent", "3.0",
  "--valid-win-min-avg-ships-sent", "4.0",
  "--valid-win-min-real-moves-turn", "0.90",
  "--passive-win-terminal-reward", "-0.25",
  "--degenerate-noop-rate", "0.90",
  "--degenerate-max-avg-ships-sent", "1.50",
  "--degenerate-min-winrate", "0.60",

  "--collapse-stop-evals", "3",
  "--collapse-max-recoveries", "4",
  "--collapse-stop-noop-rate", "0.92",
  "--collapse-stop-max-avg-ships-sent", "1.75",
  "--max-consecutive-regressions", "5",

  # FIX #3: rollback_margin 0.06 -> -0.10
  # Rationale: candidate NEVER beats best_validated (5 evals, 0 promotions). PPO stuck
  # at local optimum. Negative margin allows exploration: accept candidate even if -10%
  # worse so the policy can drift away from the BC starting point.
  "--rollback-margin", "-0.10",
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
  # Lever 3: stronger dense reward toward planet capture and production growth
  # Rationale: default coefs (0.05/0.04) too weak vs per-step shaping.
  # 3x increase makes strategic outcomes (captures, production) more valuable.
  # Bug fix 1: entropy trap — entropy_coef_start 0.10 -> 0.02
  # With coef=0.10 and entropy=4.0: entropy gradient=0.40 >> policy gradient~=0 (clip_frac=0)
  # AUTO_TUNE never fires (requires clip_frac>0.020, impossible at entropy=4.0)
  # With coef=0.02: entropy gradient=0.08 ~= policy_loss=0.09, PPO can learn.
  "--entropy-coef-start", "0.02",
  "--dense-planet-coef", "0.15",
  "--dense-production-coef", "0.12",
  "--dense-ship-share-coef", "0.14",
  "--dense-score-coef", "0.10",
  "--dense-reward-clip", "0.50",
  "--run-name", $RunName
)

# FIX #2: teacher_kl_coef 0.050 -> 0.10
# Rationale: KL rose from 0.05 to 0.37 (x7), BC anchor was overwhelmed by PPO.
# Doubling coef raises KL penalty ratio from 1:8 to 1:4 vs policy_loss.
if ($UseTeacher) {
  $Args += @("--teacher-checkpoint", $TeacherCheckpoint, "--teacher-kl-coef", "0.10")
} else {
  Write-Host "Teacher checkpoint not found."
}

python @Args
