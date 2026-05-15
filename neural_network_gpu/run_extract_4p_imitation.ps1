param(
  [string]$SourceRoot = "D:\warorbit_kaggle_raw",
  [string]$OutputDir = ".\replay_corpus\imitation_4p_top10_v1",
  [int]$MaxSamples = 250000,
  [int]$MaxEpisodes = 0
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python .\scripts\extract_4p_imitation_dataset.py `
  --source-root $SourceRoot `
  --output-dir $OutputDir `
  --max-samples $MaxSamples `
  --max-episodes $MaxEpisodes `
  --max-turns 250 `
  --max-angle-error 0.28 `
  --max-candidates 2048 `
  --noop-keep-rate 0.0 `
  --shard-size 4096
