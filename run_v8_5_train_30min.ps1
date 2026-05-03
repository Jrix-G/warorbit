Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Set-Location $PSScriptRoot

$logDir = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot 'evaluations') | Out-Null

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logPath = Join-Path $logDir "v8_5_train_30min_$stamp.log"

$pythonCandidates = @(
    (Join-Path $env:USERPROFILE 'miniconda3\python.exe'),
    (Join-Path $PSScriptRoot '.venv-win\Scripts\python.exe'),
    (Join-Path $PSScriptRoot '.venv\Scripts\python.exe'),
    'python'
)

$python = $null
foreach ($candidate in $pythonCandidates) {
    if ($candidate -eq 'python' -or (Test-Path -LiteralPath $candidate)) {
        $python = $candidate
        break
    }
}

if (-not $python) {
    throw "No usable Python interpreter found."
}

Write-Host "[run_v8_5_train_30min] starting | log=$logPath"
& $python .\train_v8_5.py `
    --minutes 30 `
    --four-player-ratio 1.0 `
    --eval-four-player-ratio 1.0 `
    --pairs 4 `
    --games-per-eval 2 `
    --eval-games 12 `
    --max-steps 120 `
    --eval-max-steps 260 `
    --eval-every 1 `
    --workers 8 `
    --pool-limit 4 `
    2>&1 | Tee-Object -FilePath $logPath
