$ErrorActionPreference = "Stop"

$env:KMP_DUPLICATE_LIB_OK = "TRUE"

New-Item -ItemType Directory -Force -Path "runs", "evaluations" | Out-Null

$candidates = @(
    ".\.venv-win\Scripts\python.exe",
    "python",
    "py"
)

$python = $null
foreach ($candidate in $candidates) {
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

Write-Host "Using Python: $python"

& $python -u .\train_v14_finetune_v2.py `
    --minutes 25 `
    --workers 8 `
    --batch-size 16 `
    --max-steps 160 `
    --load evaluations\scorer_v14.npz `
    --out evaluations\scorer_v14_4p_strict_nobc_25m.npz `
    --out-critic evaluations\critic_v14_4p_strict_nobc_25m.npz `
    --bc-data replay_dataset\v14_bc_top1.npz `
    --no-bc `
    --lr 2e-4 `
    --lr-min 5e-5 `
    --temperature-start 1.25 `
    --temperature-end 0.95 `
    --modes 4p 4p 4p 4p `
    --rank-reward-4p 1.0 0.0 -0.5 -1.0 `
    --value-coef 0.10 `
    --entropy-beta 0.02 `
    --grad-clip 2.0 `
    --ppo-epochs 1 `
    --advantage-scale 1.5 `
    --selfplay-pool 4 `
    --selfplay-every 2 2>&1 | Tee-Object -FilePath runs\v14_strict4p_nobc_25m.log
