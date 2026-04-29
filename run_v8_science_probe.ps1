Remove-Item .\evaluations\v8_policy.npz, .\evaluations\v8_policy_best.npz, .\evaluations\v8_policy_state.pkl -ErrorAction SilentlyContinue
python -u .\train_v8.py --hours 0.33 --sample-stride 8 --rollout-steps 15 --min-oracle-gap 0.005 --benchmark-games 10 --benchmark-seconds 600 --save-seconds 300 --skip-initial-benchmark 2>&1 | Tee-Object -FilePath .\training_science_probe.log
