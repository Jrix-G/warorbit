param(
  [string]$DatasetDir = ".\replay_corpus\imitation_4p_top10_v1",
  [string]$InitCheckpoint = "..\runs\gpu_2p_top1_distance_guarded_local_v2\best_validated.npz",
  [string]$OutputDir = "..\runs\imitation_4p_top10_v1",
  [int]$Epochs = 3
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python .\scripts\train_imitation_4p.py `
  --dataset-dir $DatasetDir `
  --init-checkpoint $InitCheckpoint `
  --output-dir $OutputDir `
  --epochs $Epochs `
  --batch-size 128 `
  --val-samples 8192 `
  --lr 0.000020 `
  --weight-decay 0.0001
