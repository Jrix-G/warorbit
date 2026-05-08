# V14 runbook

V14 is a conservative reset after the V11/V13 failures:

1. Use `c_engine` for training and benchmarking, not the older local simulator.
2. Extract supervised data from strong Kaggle replays.
3. Score tactical candidates with richer features instead of learning raw moves.
4. Pretrain the candidate ranker by behavioral cloning.
5. Run a hybrid bot: V14 scorer plus V12 fallback.
6. Fine-tune with a BC anchor so policy updates do not collapse.
7. Promote only through a reproducible V7/V12 gate.

## Smoke

```bash
python3 -m py_compile v14_core.py bot_v14.py extract_v14_bc_dataset.py train_v14_bc.py train_v14_finetune.py benchmark_v14.py gate_v14.py
python3 extract_v14_bc_dataset.py --inputs replays/top1-05-05 --max-samples 50 --output /tmp/v14_bc_smoke.npz
python3 train_v14_bc.py --data /tmp/v14_bc_smoke.npz --out /tmp/scorer_v14_smoke.npz --epochs 2 --batch-size 16
python3 -u benchmark_v14.py --v14-weights /tmp/scorer_v14_smoke.npz --games 1 --workers 1 --max-steps 40 --modes 2p --bots v14 v12 --opponents greedy
python3 -u gate_v14.py --v14-weights /tmp/scorer_v14_smoke.npz --games 1 --workers 1 --max-steps 40 --opponents greedy --allowed-drop 1.0 --min-avg-delta -1.0
```

## Full supervised pass

Use winner-only extraction first. In the current `replays/top1-05-05` folder,
`bowwowforeach` alone only yields a small dataset, while winner-only gives more
usable supervised samples without mixing every losing player.

```bash
python3 extract_v14_bc_dataset.py \
  --inputs replays/top1-05-05 \
  --output replay_dataset/v14_bc_top1.npz

python3 train_v14_bc.py \
  --data replay_dataset/v14_bc_top1.npz \
  --out evaluations/scorer_v14.npz \
  --epochs 40 \
  --batch-size 128 \
  --lr 0.0003
```

## Conservative fine-tune

```bash
python3 -u train_v14_finetune.py \
  --minutes 120 \
  --load evaluations/scorer_v14.npz \
  --out evaluations/scorer_v14_ft.npz \
  --bc-data replay_dataset/v14_bc_top1.npz \
  --bc-weight 0.35 \
  --batch-size 16 \
  --lr 0.0001 \
  --max-steps 220
```

## Benchmark and promotion

```bash
python3 -u benchmark_v14.py \
  --v14-weights evaluations/scorer_v14_ft.npz \
  --games 16 \
  --workers 8 \
  --max-steps 220 \
  --modes 4p 2p \
  --bots v7 v12 v13 v14

python3 -u gate_v14.py \
  --v14-weights evaluations/scorer_v14_ft.npz \
  --games 32 \
  --workers 8 \
  --max-steps 220 \
  --allowed-drop 0.03 \
  --min-avg-delta 0.02 \
  --json-out evaluations/v14_gate.json
```

If the gate fails, keep the checkpoint for analysis but do not submit or use it
as the next training base.
