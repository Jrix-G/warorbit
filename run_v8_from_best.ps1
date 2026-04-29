Remove-Item .\evaluations\v8_policy_state.pkl -ErrorAction SilentlyContinue
Copy-Item .\evaluations\v8_policy_best.npz .\evaluations\v8_policy.npz -Force
python -u .\train_v8.py --hours 0.33 --sample-stride 8 --rollout-steps 15 --min-oracle-gap 0.01 --benchmark-games 10 --benchmark-seconds 99999 --save-seconds 60 --skip-initial-benchmark --resume .\evaluations\v8_policy.npz 2>&1 | Tee-Object -FilePath .\training_from_best.log
