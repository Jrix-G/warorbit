param(
    [double]$DurationMinutes = 10.0,
    [int]$EvalEpisodes = 32,
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$argsList = @(
    ".\scripts\run_notebook_4p_training.py",
    "--config", ".\configs\default_config.json",
    "--duration-minutes", "$DurationMinutes",
    "--eval-episodes", "$EvalEpisodes"
)

if ($NoResume) {
    $argsList += "--no-resume"
}

python @argsList
