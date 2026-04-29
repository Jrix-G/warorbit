python -u .\train_v8_offline.py `
    --refresh-dataset `
    --hours 1.0 `
    --dataset-states 16 `
    --target-examples 256 `
    --max-snapshots 2400 `
    --oracle-horizon 80 `
    --min-gap 0.005 `
    --variance-horizon 220 `
    --benchmark-games 12 `
    --benchmark-seconds 900 `
    --save-seconds 600 `
    --skip-initial-benchmark `
    --out .\evaluations\v8_policy_variance.npz `
    2>&1 | Tee-Object -FilePath .\training_v8_variance.log
