python3 -u train_reinforce.py `
    --episodes 300 `
    --batch-size 16 `
    --lr 0.01 `
    --l2 0.0001 `
    --noise-start 0.3 `
    --noise-end 0.1 `
    --zoo-ratio 0.4 `
    --benchmark-every 100 `
    --benchmark-games 6 `
    --out evaluations/reinforce_weights.npy `
    2>&1 | Tee-Object -FilePath training_reinforce.log
