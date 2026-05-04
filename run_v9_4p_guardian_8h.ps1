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
$logPath = Join-Path $logDir "v9_4p_guardian_8h_$stamp.log"

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
    '--minutes', '480',
    '--hard-timeout-minutes', '480',
    '--workers', '8',
    '--pairs', '8',
    '--games-per-eval', '3',
    '--eval-games', '12',
    '--benchmark-games', '24',
    '--min-promotion-benchmark-games', '24',
    '--benchmark-progress-every', '4',
    '--eval-every', '1',
    '--benchmark-every', '1',
    '--max-steps', '120',
    '--eval-max-steps', '220',
    '--four-player-ratio', '1.0',
    '--eval-four-player-ratio', '1.0',
    '--benchmark-four-player-ratio', '1.0',
    '--train-search-width', '3',
    '--train-simulation-depth', '0',
    '--train-simulation-rollouts', '0',
    '--train-opponent-samples', '1',
    '--front-lock-turns', '22',
    '--target-active-fronts', '2.0',
    '--target-backbone-turn-frac', '0.15',
    '--front-penalty-weight', '0.055',
    '--front-penalty-cap', '0.12',
    '--front-ok-bonus', '0.070',
    '--front-partial-bonus', '0.035',
    '--backbone-penalty-weight', '0.120',
    '--backbone-bonus-weight', '0.100',
    '--front-pressure-plan-bias', '0.16',
    '--front-pressure-attack-penalty', '0.14',
    '--guardian-enabled', '1',
    '--guardian-min-benchmark-4p', '0.42',
    '--guardian-min-benchmark-backbone', '0.08',
    '--guardian-max-benchmark-fronts', '2.70',
    '--guardian-max-generalization-gap', '0.18',
    '--export-best-on-finish', '1',
    '--min-benchmark-score', '0.35',
    '--max-generalization-gap', '0.18',
    '--exploration-rate', '0.08',
    '--reward-noise', '0.008',
    '--pool-limit', '15',
    '--checkpoint', 'evaluations\v9_4p_guardian_8h_latest.npz',
    '--best-checkpoint', 'evaluations\v9_4p_guardian_8h_best.npz',
    '--export-checkpoint', 'evaluations\v9_4p_guardian_8h_policy.npz',
    '--log-jsonl', 'evaluations\v9_4p_guardian_8h_train.jsonl',
    '--train-opponents'
) + $trainOpponents

if ($Fresh) {
    $arguments += '--no-resume'
}

Write-Host "[run_v9_4p_guardian_8h] starting | workers=8 | 4p_only=1 | log=$logPath"
& $python @arguments 2>&1 | Tee-Object -FilePath $logPath
