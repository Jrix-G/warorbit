# Kaggle GPU run

## Notebook setup

Use a Kaggle Notebook with accelerator `GPU`. If the repo is attached as a Kaggle dataset, copy it to writable storage first:

```bash
cp -r /kaggle/input/warorbit /kaggle/working/warorbit
cd /kaggle/working/warorbit
```

If the checkpoint is attached as a dataset, set `RESUME` to its path. Example:

```bash
export RESUME=/kaggle/input/warorbit-checkpoints/best_validated.npz
```

## Recommended 2-player run

This preset spends the Kaggle session on 2-player training against only three simple agents: `random`, `greedy`, and `starter`. Checkpoint/archive league opponents are disabled for 2p by `simple_2p_only`.

```bash
cd /kaggle/working/warorbit
PYTHONPATH=/kaggle/working/warorbit python neural_network_gpu/scripts/run_gpu.py \
  --runs-root /kaggle/working/runs \
  --resume-checkpoint "$RESUME" \
  --duration-minutes 690 \
  --workers 12 \
  --train-every 96 \
  --eval-every 384 \
  --eval-episodes 96 \
  --batch-size 128 \
  --batch-timeout 0.010 \
  --ppo-minibatch-size 512 \
  --learning-rate 0.00008 \
  --ppo-epochs 3 \
  --n-players 2 \
  --target-winrate 0.85 \
  --run-name kaggle_2p_$(date -u +%Y%m%d_%H%M%S)
```

## Monitor

```bash
tail -f /kaggle/working/runs/kaggle_2p_*/gpu_train.log
nvidia-smi
```

Good signs:

- `train_s` stays small compared with collection time.
- `eps_per_hour` is much higher than the old run.
- `grad_norm` is not near zero all the time.
- `policy_loss`, `value_loss`, and `entropy` are finite.
- `param_delta` is above zero after each train step; this proves the optimizer moved weights.
- `ratio_std`, `ratio_range`, or `logr_max` rise above exact zero even when rounded `kl` and `clip_frac` are still `0.0000`.
- `clip_frac` remains below about `0.30`; exact zero is acceptable only if the ratio diagnostics and `param_delta` show movement.

After the run, download or save `/kaggle/working/runs/<run_name>/best_validated.npz`, `latest.npz`, `gpu_train.log`, `eval_history.jsonl`, and `result.json`.
