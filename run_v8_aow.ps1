python -u .\train_v8_offline.py `
    --refresh-dataset `
    --hours 0.33 `
    --dataset-states 16 `
    --target-examples 128 `
    --max-snapshots 1200 `
    --oracle-horizon 80 `
    --min-gap 0.005 `
    --variance-horizon 200 `
    --benchmark-games 8 `
    --benchmark-seconds 600 `
    --save-seconds 600 `
    --skip-initial-benchmark `
    --out .\evaluations\v8_policy_aow.npz `
    2>&1 | Tee-Object -FilePath .\training_v8_aow.log
