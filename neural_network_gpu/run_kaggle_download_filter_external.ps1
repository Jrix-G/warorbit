param(
    [string]$RawRoot = "D:\warorbit_kaggle_raw",
    [string]$OutputRoot = ".\replay_corpus\kaggle_top123_2p",
    [int]$MaxTurns = 250,
    [double]$MaxFileMb = 50.0,
    [int]$Limit = 5000,
    [switch]$KeepRaw
)

$ErrorActionPreference = "Stop"

$Datasets = @(
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

New-Item -ItemType Directory -Force -Path $RawRoot | Out-Null

foreach ($Dataset in $Datasets) {
    $Slug = ($Dataset -split "/")[-1]
    $DayRoot = Join-Path $RawRoot $Slug
    New-Item -ItemType Directory -Force -Path $DayRoot | Out-Null

    kaggle datasets download $Dataset --unzip -p $DayRoot

    $Tar = Join-Path $DayRoot "episodes.tar.gz"
    if (Test-Path $Tar) {
        tar -xzf $Tar -C $DayRoot
        Remove-Item -LiteralPath $Tar -Force
    }

    & .\run_kaggle_collect.ps1 `
        -SourceType local `
        -SourceRoot $DayRoot `
        -OutputRoot $OutputRoot `
        -MaxTurns $MaxTurns `
        -MaxFileMb $MaxFileMb `
        -DefaultRank 1 `
        -Limit $Limit

    if (-not $KeepRaw) {
        Remove-Item -LiteralPath $DayRoot -Recurse -Force
    }

    $Manifest = Join-Path $OutputRoot "manifest.jsonl"
    if (Test-Path $Manifest) {
        $Accepted = (Get-Content $Manifest | Measure-Object -Line).Lines
        if ($Accepted -ge $Limit) {
            break
        }
    }
}

if (-not $KeepRaw) {
    $Remaining = Get-ChildItem -Path $RawRoot -Force -ErrorAction SilentlyContinue
    if (-not $Remaining) {
        Remove-Item -LiteralPath $RawRoot -Force
    }
}
