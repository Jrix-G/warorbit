python -u .\benchmark_v7_aow.py `
    --games-per-opp 20 `
    --seed-offset 200 `
    --workers 8 `
    2>&1 | Tee-Object -FilePath .\benchmark_v7_aow.log
