# SESSION 2026-04-30 - BOT_V7 / KAGGLE TRAINING STATE

Context:
- Project: Orbit Wars / warorbit
- Date: 2026-04-30
- Goal: keep a compact handoff for the next conversation and for the overnight run

## Current State

The active local bot logic lives in `bot_v7.py`.
The Kaggle ES training loop lives in `train_kaggle.py`.
The Kaggle smoke / long-run launchers live in:
- `run_kaggle_smoke.ps1`
- `run_kaggle_night.ps1`

Important point:
- training outputs are not automatically used by `submission.py` yet
- `submission.py` is still a separate bot path
- the learned ES checkpoint is stored locally in `evaluations/scorer_v7_kaggle*.npz`

## What The Bot Does

`bot_v7.py` is the current production-quality heuristic bot:
- it builds a projected world model from the current observation
- it ranks candidate missions with a scorer plus fixed heuristic constants
- it uses `set_scorer()` and `set_heuristic_params()` at training time
- it keeps the opening / midgame / lategame logic inside the bot, not in the trainer

Current notable tuned change:
- `AOW_OPENING_MIN_PROD = 1`
- this allows opening captures on prod=1 neutrals instead of skipping them

## Training Setup

The Kaggle trainer is ES with:
- antithetic sampling
- rank-shaped fitness
- momentum on the ES update
- fixed eval set on the local notebook zoo
- auto-resume from checkpoint
- best checkpoint separated from latest checkpoint
- rollback on clearly bad evals

Current checkpoint logic:
- `evaluations/scorer_v7_kaggle.npz` = best checkpoint
- `evaluations/scorer_v7_kaggle_latest.npz` = latest state
- the trainer resumes from `best` by default

## Recent Results

Observed runs:
- smoke Kaggle run with `pairs=2`, `episode_steps=150`, `eval_games_per_opp=1`
- 25 min run on Kaggle with 7 workers
- 7h-style conservative run plan with `resume_source=best`

Key numbers from the latest longer runs:
- best fixed eval reached: `43%`
- later latest state drifted down to around `37%`
- a subsequent best-preserving run still recovered `40%+` on fixed eval

Interpretation:
- the optimizer is learning something real
- the latest state can drift away from the best
- the best checkpoint is the only reliable thing to carry forward

## What This Means

The current system is useful, but the signal is still noisy:
- the ES loop is not the main bug
- the main risk is over-exploration and checkpoint drift
- the next useful gains come from more conservative exploitation around the best checkpoint

The most important practical rule is:
- for long runs, resume from `best`, not `latest`
- keep rollback on
- monitor fixed evals often enough to catch regressions

## Vision

Short term:
- squeeze more WR out of the current heuristic bot by exploiting the best checkpoint
- keep the opening policy aligned with the top notebook behavior
- avoid drifting away from the validated best state

Medium term:
- use the learned checkpoint to warm-start a stronger bot
- move from pure heuristic tuning toward a richer learned policy
- branch the training signal into a model that can be used directly in submission

Long term:
- the current ES layer should be treated as a bridge, not the final architecture
- the real end state is a stronger agent that captures the top-bot behaviors more directly

## Night Run Command

Use this launcher for the current long run:

```powershell
.\run_kaggle_night.ps1
```

Current night-run settings:
- `--minutes 420`
- `--workers 7`
- `--pairs 4`
- `--games-per-eval 2`
- `--eval-games-per-opp 3`
- `--episode-steps 220`
- `--resume-source best`
- `--rollback-on-bad-eval`
- `--eval-every 10`

## Resume Notes For Tomorrow

When reopening a new session, the first things to check are:
- `evaluations/scorer_v7_kaggle.npz`
- `evaluations/scorer_v7_kaggle_latest.npz`
- `training_kaggle_7h.log`

The best checkpoint is the one that should drive the next run.
