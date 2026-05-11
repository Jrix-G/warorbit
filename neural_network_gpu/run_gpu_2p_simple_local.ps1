$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$RunName = "gpu_2p_rg_nosupport_local"
$RunDir = Join-Path $RepoRoot.Path "runs\$RunName"
$LatestCheckpoint = Join-Path $RunDir "latest.npz"
$TrainOpponents = "random,random,random,random,random,random,greedy,greedy,greedy,greedy,greedy,greedy,greedy,greedy,greedy,greedy,greedy,greedy,greedy,starter"
$EvalOpponents = "random,greedy,starter"
$ResumeArgs = @()
if (Test-Path $LatestCheckpoint) {
  $ResumeArgs = @("--resume-checkpoint", $LatestCheckpoint)
  Write-Host "Resuming from $LatestCheckpoint"
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
  --device cuda `
  --duration-minutes 720 `
  --workers 6 `
  --train-every 32 `
  --eval-every 256 `
  --eval-episodes 16 `
  --batch-size 48 `
  --batch-timeout 0.010 `
  --ppo-minibatch-size 96 `
  --learning-rate 0.00008 `
  --min-lr 0.00002 `
  --ppo-epochs 3 `
  --n-players 2 `
  --simple-opponents $TrainOpponents `
  --eval-opponents $EvalOpponents `
  --disable-support-actions `
  --auto-tune-training `
  --target-winrate 0.80 `
  --max-eval-do-nothing-rate 0.55 `
  --min-eval-avg-ships-sent 3.0 `
  --rollback-margin 1.0 `
  --max-opponent-regression 1.0 `
  --min-ci-promotion-games 96 `
  --run-name $RunName
