$ErrorActionPreference = "Stop"

$env:KMP_DUPLICATE_LIB_OK = "TRUE"

New-Item -ItemType Directory -Force -Path "runs" | Out-Null

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

$candidates = @(
    @{ Name = "base"; Path = "evaluations\scorer_v14.npz" },
    @{ Name = "strict_nobc_25m_best4p"; Path = "evaluations\scorer_v14_4p_strict_nobc_25m.best4p.npz" },
    @{ Name = "continue20m_best4p"; Path = "evaluations\scorer_v14_4p_strict_nobc_continue20m.best4p.npz" },
    @{ Name = "continue20m_latest"; Path = "evaluations\scorer_v14_4p_strict_nobc_continue20m.npz" }
)

foreach ($candidate in $candidates) {
    if (-not (Test-Path $candidate.Path)) {
        Write-Host "Skipping missing checkpoint: $($candidate.Path)"
        continue
    }
    $log = "runs\bench_v14_$($candidate.Name)_4p.log"
    Write-Host "Benchmarking $($candidate.Name): $($candidate.Path)"
    & $python -u .\benchmark_v14.py `
        --v14-weights $candidate.Path `
        --games 32 `
        --workers 8 `
        --max-steps 220 `
        --modes 4p `
        --bots v14 `
        --opponents notebook_orbitbotnext notebook_distance_prioritized notebook_physics_accurate notebook_pascalledesma_orbitwork_v14 `
        2>&1 | Tee-Object -FilePath $log
}
