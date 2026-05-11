$ErrorActionPreference = "Stop"

$env:KMP_DUPLICATE_LIB_OK = "TRUE"

New-Item -ItemType Directory -Force -Path "runs", "evaluations" | Out-Null

$pythonCandidates = @(
    ".\.venv-win\Scripts\python.exe",
    "python",
    "py"
)

$python = $null
foreach ($candidate in $pythonCandidates) {
    if ($candidate -like ".\*" -and -not (Test-Path $candidate)) {
        continue
    }
    try {
        & $candidate -c "import numpy" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $python = $candidate
            break
        }
    } catch {
        continue
    }
}

if ($null -eq $python) {
    throw "No Python with numpy found. Install numpy in .venv-win or use a Python environment that has numpy."
}

$base = "evaluations\scorer_v14_4p_strict_nobc_25m.best4p.npz"
if (-not (Test-Path $base)) {
    throw "Missing $base. Run .\run_v14_strict4p_nobc_25m.ps1 first."
}

Write-Host "Using Python: $python"
Write-Host "Loading best 4p checkpoint: $base"

& $python -u .\train_v14_finetune_v2.py `
    --minutes 20 `
    --workers 8 `
    --batch-size 16 `
    --max-steps 160 `
    --load $base `
    --out evaluations\scorer_v14_4p_strict_nobc_continue20m.npz `
    --out-critic evaluations\critic_v14_4p_strict_nobc_continue20m.npz `
    --bc-data replay_dataset\v14_bc_top1.npz `
    --no-bc `
    --lr 1.5e-4 `
    --lr-min 8e-5 `
    --temperature-start 1.15 `
    --temperature-end 1.00 `
    --modes 4p 4p 4p 4p `
    --rank-reward-4p 1.0 0.0 -0.5 -1.0 `
    --value-coef 0.05 `
    --entropy-beta 0.025 `
    --grad-clip 2.0 `
    --ppo-epochs 1 `
    --advantage-scale 1.75 `
    --selfplay-pool 4 `
    --selfplay-every 2 2>&1 | Tee-Object -FilePath runs\v14_strict4p_nobc_continue20m.log
