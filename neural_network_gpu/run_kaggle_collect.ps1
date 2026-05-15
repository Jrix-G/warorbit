param(
    [string]$SourceType = "local",
    [string]$SourceRoot = ".\replays",
    [string]$OutputRoot = ".\replay_corpus\kaggle_top123_2p",
    [string]$KaggleId = "",
    [string[]]$KaggleIds = @(),
    [string]$DownloadRoot = ".\.tmp\kaggle_downloads",
    [int]$MaxTurns = 250,
    [double]$MaxFileMb = 50.0,
    [double]$MaxSourceMb = 100.0,
    [int]$DefaultRank = 0,
    [int]$Limit = 5000
)

$ErrorActionPreference = "Stop"

$args = @(
    "scripts\collect_kaggle_replays.py",
    "--source-type", $SourceType,
    "--source-root", $SourceRoot,
    "--output-root", $OutputRoot,
    "--download-root", $DownloadRoot,
    "--max-turns", "$MaxTurns",
    "--max-file-mb", "$MaxFileMb",
    "--max-source-mb", "$MaxSourceMb",
    "--default-rank", "$DefaultRank",
    "--limit", "$Limit",
    "--top-ranks", "1,2,3"
)

if ($KaggleId) {
    $args += @("--kaggle-id", $KaggleId)
}

if ($KaggleIds -and $KaggleIds.Count -gt 0) {
    $args += @("--kaggle-ids", ($KaggleIds -join ","))
}

python @args
