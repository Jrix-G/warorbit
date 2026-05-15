$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")

Set-Location $RepoRoot
$env:PYTHONPATH = $RepoRoot.Path
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

python -c "import torch, sys; print('CUDA:', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { throw "CUDA not available." }

python neural_network_gpu\scripts\eval_checkpoints.py `
    --games 64 `
    --opponents "random,greedy" `
    --n-players 2 `
    --device cuda `
    --config ..\runs\gpu_2p_curriculum_phase1_from_imitation\config.json
