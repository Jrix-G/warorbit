$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$RunName = "gpu_2p_simple_local"

Set-Location $RepoRoot
$env:PYTHONPATH = $RepoRoot.Path
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

python -c "import torch, sys; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'cuda', getattr(torch.version, 'cuda', None)); sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) {
  throw "PyTorch cannot see CUDA. Install a CUDA-enabled PyTorch build or run with --device cpu."
}

python neural_network_gpu\scripts\run_gpu.py `
  --device cuda `
  --duration-minutes 360 `
  --workers 4 `
  --train-every 32 `
  --eval-every 256 `
  --eval-episodes 64 `
  --batch-size 32 `
  --batch-timeout 0.010 `
  --ppo-minibatch-size 64 `
  --learning-rate 0.00008 `
  --ppo-epochs 3 `
  --n-players 2 `
  --target-winrate 0.85 `
  --run-name $RunName
