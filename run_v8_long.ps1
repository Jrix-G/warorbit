Remove-Item .\evaluations\v8_policy.npz, .\evaluations\v8_policy_best.npz -ErrorAction SilentlyContinue
python -u .\train_v8.py --hours 2.0 --sample-stride 8 --rollout-steps 15 --min-oracle-gap 0.01 --benchmark-games 5 --benchmark-seconds 1200 --save-seconds 600 --skip-initial-benchmark 2>&1 | Tee-Object -FilePath .\training.log
