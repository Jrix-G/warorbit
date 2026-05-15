param(
    [string]$OutputRoot = ".\replay_corpus\kaggle_top123_2p",
    [string]$DownloadRoot = ".\.tmp\kaggle_downloads",
    [int]$MaxTurns = 250,
    [double]$MaxFileMb = 50.0,
    [double]$MaxSourceMb = 100.0,
    [int]$Limit = 5000
)

$ErrorActionPreference = "Stop"

$KaggleIds = @(
    "bovard/orbit-wars-top10-episodes-2026-04-16",
    "bovard/orbit-wars-top10-episodes-2026-04-17",
    "bovard/orbit-wars-top10-episodes-2026-04-18",
    "bovard/orbit-wars-top10-episodes-2026-04-19",
    "bovard/orbit-wars-top10-episodes-2026-04-20",
    "bovard/orbit-wars-top10-episodes-2026-04-21",
    "bovard/orbit-wars-top10-episodes-2026-04-22",
    "bovard/orbit-wars-top10-episodes-2026-04-23",
    "bovard/orbit-wars-top10-episodes-2026-04-24",
    "bovard/orbit-wars-top10-episodes-2026-04-25",
    "bovard/orbit-wars-top10-episodes-2026-04-26",
    "bovard/orbit-wars-top10-episodes-2026-04-27",
    "bovard/orbit-wars-top10-episodes-2026-04-28",
    "bovard/orbit-wars-top10-episodes-2026-04-29",
    "bovard/orbit-wars-top10-episodes-2026-04-30",
    "bovard/orbit-wars-top10-episodes-2026-05-01",
    "bovard/orbit-wars-top10-episodes-2026-05-02",
    "bovard/orbit-wars-top10-episodes-2026-05-03",
    "bovard/orbit-wars-top10-episodes-2026-05-04"
)

& .\run_kaggle_collect.ps1 `
    -SourceType dataset `
    -OutputRoot $OutputRoot `
    -DownloadRoot $DownloadRoot `
    -KaggleIds $KaggleIds `
    -MaxTurns $MaxTurns `
    -MaxFileMb $MaxFileMb `
    -MaxSourceMb $MaxSourceMb `
    -Limit $Limit
