# V17 AlphaZero — Run Context

## Goal
Beat V15 (RCC, ~975 ELO) at 75-80% WR → ~1400 ELO. Target: +400 ELO from V15.

## Pipeline Files
- `v17_net.py` — V17Net (d=64, ~35k params). Entity-wise attention, per-planet policy head.
- `v17_encode.py` — state encoder (pf: planet features, gf: global features)
- `v17_mcts.py` — single-agent PUCT MCTS; opponents modeled by net greedy policy
- `v17_selfplay.py` — self-play worker; returns (pf, gf, pol, mask, outcome) per step
- `v17_train.py` — policy CE + value MSE training
- `v17_loop.py` — outer AlphaZero loop: self-play → train → checkpoint, resume-safe
- `v17_agent.py` — Kaggle submission wrapper (has silent `except Exception: return []`)
- `bench_v17.py` — benchmark V17 checkpoint vs V15 (N games, n_sims MCTS)

Checkpoints saved to `analysis/v17_loop.pt` (latest) and `analysis/v17_iter{N}.pt` (per-iter).

## Critical Bugs Fixed
1. **Warmstart always-pass**: BC training has 89% pass labels → network converges to all-pass.
   Fix: use `--fresh` (random init). Do NOT use `analysis/v17_warmstart.pt` as starting point.

2. **--vs-v15-frac causes infinite runtime**: V15 RCC search has no time limit locally
   (Kaggle enforces 1s kill, local doesn't). With 40% V15 opponents, iter 1 takes ~200h+ not 5-7h.
   Fix: always use `--vs-v15-frac 0.0` for training. Use bench_v17.py for V15 evaluation.

## VPS State (as of 2026-05-17 ~22h)
- Shape: VM.Standard2.4 = 4 OCPUs = **8 vCPUs** (1 OCPU = 1 core / 2 threads), 60GB RAM
- Run launched at ~14:40 with: `python -u v17_loop.py --fresh --iterations 18 --games 100 --n-sims 100 --workers 14`
- **STUCK**: default --vs-v15-frac=0.4 triggers the unlimited-search bug
- Also: 14 workers on 8 vCPUs = 1.75x oversubscribed; each worker left BLAS
  multi-threaded -> ~112 threads on 8 cores, heavy thrash
- `analysis/v17_iter1.pt` and `analysis/v17_loop.pt` on VPS are OLD (from PC smoke tests, timestamp 14:46)
- Iter 1 has NOT completed after 8+ hours

## Speed Fixes Applied (2026-05-17)
- `v17_loop.py`: ProcessPoolExecutor `initializer=_init_worker` pins each
  worker to `torch.set_num_threads(1)` (forwards are batch=1, threads only
  thrash). Main-process training stays multi-threaded.
- `pool.map(..., chunksize=1)` removes the straggler/tail effect from
  variable-length games.
- All quality-neutral: games are seeded; worker/thread/chunk counts cannot
  change results.

## Immediate Action Required on VPS
```bash
# git pull to get the speed fixes, then in tmux Ctrl+C the stuck run:
python -u v17_loop.py --fresh --iterations 18 --games 100 --n-sims 100 --workers 7 --vs-v15-frac 0.0
```
Use `--workers 7` (8 vCPUs minus 1 for the parent/training), NOT 14.

Monitor: `watch -n 60 "ls -la analysis/v17_loop.pt"`
Timestamp change = iter 1 done. Expected: ~5-7h per iter.

## Expected Training Behavior
- Iter 1 log line: `policy_ce≈3.38` (uniform random), `value_mse≈0.44`, `win_avg=0.000`
- Iter 3+ (after ~15h): run bench gate: `python -u bench_v17.py --ckpt analysis/v17_iter3.pt --games 24 --n-sims 50`
  - Must show WR > 5% vs V15 (any improvement from random is a good sign)
- Convergence expected around iter 10-15 for 50-70% WR vs V15

## Training Schedule
- 18 iterations × ~5-7h each = ~4-5 days total
- After P1 (2p only), run `--mode 4` for 4p games (P2)
- After that: ONNX int8 quantization + Kaggle packaging (P3)

## Smoke Test Results (from PC, reference)
Two runs with --fresh --games 6-10 --n-sims 20 both completed in ~17-21 min:
- policy_ce=3.38-3.39 (expected uniform)
- value_mse=0.43-0.44
- win_avg=0.000 (2p symmetric, expected)

## Architecture Notes
- Player-relative encoding: same net handles 2p and 4p
- MCTS candidate moves: k=10 planet targets per owned planet + pass
- Temperature=1.0 for first 30 moves, then greedy
- Buffer: 160k samples max, FIFO
- No warmstart needed — random init converges fine via self-play
