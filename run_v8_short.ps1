Remove-Item .\evaluations\v8_policy.npz, .\evaluations\v8_policy_best.npz -ErrorAction SilentlyContinue
python -u .\train_v8.py --hours 0.2 --sample-stride 8 --rollout-steps 15 --min-oracle-gap 0.01 --benchmark-games 2 --benchmark-seconds 99999 --save-seconds 60 --skip-initial-benchmark 2>&1 | Tee-Object -FilePath .\training_short.log
