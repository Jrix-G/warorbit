$env:PYTHONIOENCODING = 'utf-8'
python -u .\train_v7_fast.py `
    --minutes 5 `
    --workers 4 `
    --pairs 2 `
    --games-per-eval 1 `
    --match-4p-ratio 0.7 `
    --skip-baseline-eval `
    --eval-every 999 `
    2>&1 | Tee-Object -FilePath .\training_v7_smoke.log
