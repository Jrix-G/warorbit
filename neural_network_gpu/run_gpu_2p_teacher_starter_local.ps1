$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$SourceRunName = "gpu_2p_rg_nosupport_local"
$RunName = "gpu_2p_teacher_starter_local"
$SourceRunDir = Join-Path $RepoRoot.Path "runs\$SourceRunName"
$RunDir = Join-Path $RepoRoot.Path "runs\$RunName"
$SourceBest = Join-Path $SourceRunDir "best_validated.npz"
$SourceLatest = Join-Path $SourceRunDir "latest.npz"
$LatestCheckpoint = Join-Path $RunDir "latest.npz"

$TrainOpponents = "random,random,random,random,greedy,greedy,greedy,greedy,greedy,greedy,greedy,starter,starter,starter,starter,starter,starter,starter,starter,starter"
$EvalOpponents = "random,greedy,starter"
$ResumeArgs = @()
$TeacherCheckpoint = $SourceBest

if (Test-Path $LatestCheckpoint) {
  $ResumeArgs = @("--resume-checkpoint", $LatestCheckpoint)
  Write-Host "Resuming teacher-starter run from $LatestCheckpoint"
} elseif (Test-Path $SourceBest) {
  $ResumeArgs = @("--resume-checkpoint", $SourceBest)
  Write-Host "Starting teacher-starter run from source best $SourceBest"
} elseif (Test-Path $SourceLatest) {
  $ResumeArgs = @("--resume-checkpoint", $SourceLatest)
  $TeacherCheckpoint = $SourceLatest
  Write-Host "Source best missing; starting from source latest $SourceLatest"
} else {
  throw "No source checkpoint found in $SourceRunDir"
}

Set-Location $RepoRoot
$env:PYTHONPATH = $RepoRoot.Path
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

python -c "import torch, sys; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'cuda', getattr(torch.version, 'cuda', None)); sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) {
  throw "PyTorch cannot see CUDA. Install a CUDA-enabled PyTorch build or run with --device cpu."
}

python neural_network_gpu\scripts\run_gpu.py `
  @ResumeArgs `
  --teacher-checkpoint $TeacherCheckpoint `
  --teacher-kl-coef 0.050 `
  --device cuda `
  --duration-minutes 360 `
  --workers 10 `
  --train-every 32 `
  --eval-every 512 `
  --eval-episodes 32 `
  --batch-size 64 `
  --batch-timeout 0.010 `
  --ppo-minibatch-size 96 `
  --learning-rate 0.00003 `
  --min-lr 0.00001 `
  --max-lr 0.00006 `
  --ppo-epochs 3 `
  --n-players 2 `
  --simple-opponents $TrainOpponents `
  --eval-opponents $EvalOpponents `
  --disable-support-actions `
  --auto-tune-training `
  --policy-prior-strength 0.20 `
  --target-winrate 0.80 `
  --max-eval-do-nothing-rate 0.97 `
  --rollback-on-noop-rate 0.985 `
  --min-eval-avg-ships-sent 10.0 `
  --rollback-margin 0.22 `
  --max-opponent-regression 0.16 `
  --min-ci-promotion-games 96 `
  --stabilizer-target-noop 0.94 `
  --stabilizer-target-passivity 0.80 `
  --stabilizer-target-real-moves-turn 0.45 `
  --stabilizer-target-avg-ships-sent 18.0 `
  --stabilizer-min-weighted-score 0.62 `
  --stabilizer-max-ship-bonus 2.20 `
  --stabilizer-max-min-ships 14 `
  --stabilizer-max-starter-count 10 `
  --stabilizer-ratio-floor-step 0.10 `
  --stabilizer-ratio-floor-cap 0.85 `
  --run-name $RunName
