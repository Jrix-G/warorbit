$env:PYTHONIOENCODING = 'utf-8'
python -u .\train_kaggle.py `
    --minutes 20 `
    --workers 8 `
    --pairs 2 `
    --games-per-eval 1 `
    --eval-games-per-opp 1 `
    --match-4p-ratio 0.7 `
    --episode-steps 150 `
    --skip-baseline-eval `
    --resume-source best `
    --eval-every 999 `
    2>&1 | Tee-Object -FilePath .\training_kaggle_smoke.log
