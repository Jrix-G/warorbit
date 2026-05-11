$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$Stage1RunName = "gpu_2p_rg_active_local"
$RunName = "gpu_2p_starter_local"
$Stage1RunDir = Join-Path $RepoRoot.Path "neural_network_gpu\runs\$Stage1RunName"
$Stage1Best = Join-Path $Stage1RunDir "best_validated.npz"
$Stage1Latest = Join-Path $Stage1RunDir "latest.npz"

$ResumeCheckpoint = ""
if (Test-Path $Stage1Best) {
  $ResumeCheckpoint = $Stage1Best
} elseif (Test-Path $Stage1Latest) {
  $ResumeCheckpoint = $Stage1Latest
} else {
  throw "No stage-1 checkpoint found. Run .\run_gpu_2p_simple_local.ps1 first."
}

Set-Location $RepoRoot
$env:PYTHONPATH = $RepoRoot.Path
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

python -c "import torch, sys; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'cuda', getattr(torch.version, 'cuda', None)); sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) {
  throw "PyTorch cannot see CUDA. Install a CUDA-enabled PyTorch build or run with --device cpu."
}

Write-Host "Starting starter stage from $ResumeCheckpoint"

python neural_network_gpu\scripts\run_gpu.py `
  --resume-checkpoint $ResumeCheckpoint `
  --device cuda `
  --duration-minutes 720 `
  --workers 4 `
  --train-every 32 `
  --eval-every 128 `
  --eval-episodes 8 `
  --batch-size 32 `
  --batch-timeout 0.010 `
  --ppo-minibatch-size 64 `
  --learning-rate 0.00006 `
  --min-lr 0.00002 `
  --ppo-epochs 3 `
  --n-players 2 `
  --simple-opponents random,greedy,starter `
  --target-winrate 0.85 `
  --rollback-margin 1.0 `
  --max-opponent-regression 1.0 `
  --min-ci-promotion-games 96 `
  --run-name $RunName
