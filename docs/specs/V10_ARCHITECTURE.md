# V10 Architecture (4p-First)

## Goal
V10 is a full 4p-first branch designed to break the V9 ceiling.

Core rule:
- optimize for stable 4-player wins first
- keep 2p as a secondary signal

## Inputs Used
V10 design is driven by:
- recent V9 logs in `logs/` and `logs/logsfromVPS/`
- top leaderboard replay summaries from `analysis/top1_0505_metrics.json` and `analysis/top1_0505_metrics_dedup.json`
- replay samples from `replays/top1-05-05/`

Observed top1 pattern used in V10:
- fast early conversion
- high t60/t100 planet conversion
- lower unstable target switching in 4p

## Main Changes vs V9
1. New config: `V10Config` with strict 4p defaults.
2. New policy surface: `war_orbit/agents/v10/policy.py`.
3. New trainer: `war_orbit/training/v10_trainer.py`.
4. New runner: `run_v10.py`.
5. New wrapper entrypoint: `bot_v10.py`.

## Strategy Axis
- stricter front budget in 4p (`target_active_fronts=1.8`)
- stronger front overlap penalties
- longer front lock horizon (`front_lock_turns=42`)
- strict single-target mode enabled by default
- reduced opportunistic snipe prior
- stronger consolidation/backbone bias

## Training Objective (V10)
Selection and training scores are 4p-dominant:
- train score prioritizes `wr_4p`, `conversion_t60/t100`, and front pressure
- explicit penalty for unstable focus switches
- promotion uses 4p-heavy weighted score with hard blockers

## New Logging
Each generation now logs:
- `utc_ts` and `local_ts`
- `train/eval/benchmark_focus_switches`
- full 4p diagnostics and pressure adjustment

This is written in `evaluations/v10_4p_train.jsonl` by default.

## Fast Engine Run
Default run is tuned for speed on `official_fast` and 4p signal density:

```bash
.venv/bin/python run_v10.py --game-engine official_fast --minutes 30 --workers 8 --train-only
```

## Success Gate
V10 target gate:
- benchmark 4p >= 0.50
- backbone >= target
- fronts <= target
- focus switching stable

