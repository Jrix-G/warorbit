param(
    [switch]$Fresh
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Set-Location $PSScriptRoot

$logDir = Join-Path $PSScriptRoot 'logs'
$evalDir = Join-Path $PSScriptRoot 'evaluations'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $evalDir | Out-Null

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logPath = Join-Path $logDir "v9_frontfix_60min_$stamp.log"

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
    throw 'No usable Python interpreter found.'
}

$trainOpponents = @(
    'random',
    'noisy_greedy',
    'starter',
    'distance',
    'sun_dodge',
    'structured',
    'orbit_stars',
    'bot_v7',
    'notebook_tactical_heuristic',
    'notebook_mdmahfuzsumon_how_my_ai_wins_space_wars',
    'notebook_sigmaborov_orbit_wars_2026_starter',
    'notebook_sigmaborov_orbit_wars_2026_tactical_heuristic',
    'notebook_orbitbotnext',
    'notebook_distance_prioritized',
    'notebook_physics_accurate'
)

$arguments = @(
    '.\run_v9.py',
    '--minutes', '60',
    '--hard-timeout-minutes', '60',
    '--train-only',
    '--skip-eval',
    '--workers', '8',
    '--pairs', '12',
    '--games-per-eval', '4',
    '--eval-every', '0',
    '--benchmark-every', '0',
    '--max-steps', '100',
    '--four-player-ratio', '0.86',
    '--train-search-width', '3',
    '--train-simulation-depth', '0',
    '--train-simulation-rollouts', '0',
    '--front-lock-turns', '20',
    '--target-active-fronts', '2.0',
    '--front-penalty-weight', '0.070',
    '--front-penalty-cap', '0.14',
    '--front-ok-bonus', '0.055',
    '--front-partial-bonus', '0.030',
    '--front-pressure-plan-bias', '0.16',
    '--front-pressure-attack-penalty', '0.15',
    '--exploration-rate', '0.06',
    '--reward-noise', '0.010',
    '--pool-limit', '15',
    '--checkpoint', 'evaluations\v9_frontfix_60m_latest.npz',
    '--best-checkpoint', 'evaluations\v9_frontfix_60m_best.npz',
    '--export-checkpoint', 'evaluations\v9_frontfix_60m_policy.npz',
    '--log-jsonl', 'evaluations\v9_frontfix_60m_train.jsonl',
    '--train-opponents'
) + $trainOpponents

if ($Fresh) {
    $arguments += '--no-resume'
}

Write-Host "[run_v9_frontfix_60min] starting | workers=8 | train_opponents=$($trainOpponents.Count) | train_only=1 | front_target=2.0 | log=$logPath"
& $python @arguments 2>&1 | Tee-Object -FilePath $logPath
