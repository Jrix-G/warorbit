# V10 War Orbit Audit Prompt

You are auditing a War Orbit bot training system. The current V10 code has received many changes, but benchmark performance is still poor or unstable compared with the older bot_v7 checkpoint that reportedly reached about 53% and roughly 700 Elo.

Your task is to find what is actually wrong with V10 and propose concrete code-level fixes. Do not give generic advice. Read the code and logs first, then explain the failure modes with evidence.

## Context

- Project root contains `run_v10.py`, `war_orbit/`, `opponents/`, and `notebooks/`.
- V10 is built mostly on V9 planner/policy code.
- Recent changes added:
  - main-front mass metrics: `main_front_ship_share`, `main_front_ready_frac`, `main_front_core_ship_share`
  - conversion metrics at t80/t100/t120
  - Thompson controller and SPRT/progress monitor
  - direct training against benchmark notebooks
  - CLI flags to reduce holdout/benchmark cost in train-only mode
- The user wants practical performance, not clean academic separation. If training on benchmark notebooks helps, assume that is acceptable.

## Current Code/Run State After Latest Fixes

Latest fixes already applied before this audit bundle:

- CLI now defaults to `workers=6`.
- `--hard-timeout-minutes` defaults to `--minutes` if omitted.
- `--reset-best-on-resume` was added to ignore stale best scores from older scoring.
- Train now includes the benchmark notebook pool directly.
- `train_state_perturbation` now actually jitters training seed offsets.
- 4p defaults were tightened:
  - `target_active_fronts=2.2`
  - `four_p_front_budget=2.1`
  - `front_budget_midgame_4p=2.0`
  - `target_backbone_turn_frac=0.22`
  - stronger front/backbone penalties.
- Planner/policy now more strongly favors `staging_transfer` and `defensive_consolidation`.

Latest volume run command:

```bash
.venv/bin/python run_v10.py \
  --train-only \
  --minutes 60 \
  --pairs 12 \
  --games-per-eval 8 \
  --eval-games 0 \
  --benchmark-games 8 \
  --holdout-eval-games-train-only 0 \
  --train-only-benchmark-every 0 \
  --train-only-benchmark-games 0 \
  --workers 6 \
  --reset-best-on-resume 1
```

Important: because holdout was disabled in this command, `eval` equals `train` in the logs for gen 14/15. Do not interpret gen 14/15 eval as real holdout.

## Observed Run Symptoms

Representative V10 logs:

```text
gen=0011 train=0.467 eval=0.438 sel=0.468 promo=0
conv=8.4/10.6/10.1/10.6
4pdiag=WARN xfer=0.59 bb=0.15 lock=0.99 fronts=3.2 mf=0.75 ready=0.87

gen=0012 train=0.484 eval=0.374 sel=0.792 promo=0
block=train_only,holdout_t120_low,skill_lcb_down,sprt=accept(+4.20)
conv=7.8/10.8/11.0/11.8
4pdiag=WARN xfer=0.60 bb=0.18 lock=1.00 fronts=2.8 mf=0.76 ready=0.88

gen=0013 train=0.498 eval=0.128 sel=0.447 promo=0
block=train_only,score_not_improved,holdout4p_low,train_holdout_gap,holdout_t120_low
conv=9.1/11.6/10.6/11.0
4pdiag=WARN xfer=0.53 bb=0.13 lock=1.00 fronts=3.1 mf=0.77 ready=0.88

gen=0014 train=0.375 eval=0.375 sel=0.348 promo=0
workers=6, pairs=12, games_per_eval=8, volume improved: 192 train games in ~18 min
conv=8.1/10.1/9.2/9.1
4pdiag=WARN xfer=0.52 bb=0.15 lock=1.00 fronts=2.9 mf=0.78 ready=0.91

gen=0015 train=0.405 eval=0.405 sel=0.313 promo=1
conv=8.1/10.4/9.8/9.8
4pdiag=WARN xfer=0.55 bb=0.147 lock=1.00 fronts=2.71 mf=0.765 ready=0.88
```

Earlier final benchmark stayed near `0.08` and sometimes started `0/9` wins. This is unacceptable versus the older v7 baseline.

## Current Interpretation

The worker/throughput problem appears fixed: gen 14 completed about 192 train games in about 18 minutes with `workers=6`.

The remaining issue is strategic:

- `mf` and `ready` are high, so the bot creates a large main front.
- `bb` is still too low: about `0.147-0.153`, below the target `0.18+` and new intended target `0.22`.
- `fronts` improved from about `3.1` to `2.71`, but still above target (`2.2-2.4`).
- conversion at t100/t120 is poor for a publishable bot: t120 around `9-10`, not `12+`.
- `reserve_hold` remains high (`~0.21-0.24`), while `staging_transfer`/`defensive_consolidation` may still not be enough to close secondary fronts.
- Since holdout was disabled for volume, the next audit must not rely on gen 14/15 eval as generalization evidence.

## Questions To Answer

1. Is the V10 objective optimizing the wrong proxy?
   - Check `_regularized_train_score`, `_front_pressure_adjustment`, `_main_front_progress_score`, train-only promotion logic.
   - Determine whether `mf` and `ready` are over-rewarded while actual win conversion is weak.

2. Is the planner creating a large front that is too passive?
   - Inspect `war_orbit/agents/v9/planner.py` and `war_orbit/agents/v9/policy.py`.
   - Look for excessive reserve/transfer/consolidation behavior.
   - Explain why `mf=0.75` can coexist with poor benchmark winrate.

3. Is V10 worse than v7 because it lost a tactical capability?
   - Compare V10/V9 planning behavior against `bot_v7.py` and notebooks if useful.
   - Identify concrete missing heuristics: attack timing, finishing, neutral expansion, focus switching, sniper defense, or multi-front closure.

4. Are train/eval/benchmark schedules misleading?
   - Inspect `war_orbit/training/curriculum.py`, `self_play.py`, and `v10_trainer.py`.
   - Check whether train-only still spends too much time evaluating or promotes from weak evidence.
   - Check whether benchmark opponents are actually available through `opponents.ZOO`.

5. What should be changed next?
   - Provide a ranked list of code changes.
   - For each change, name the exact files/functions.
   - Prefer high-impact fixes over more logging.
   - Be explicit about what to remove or de-weight if needed.

6. After the latest fixes, why is backbone still below target?
   - Inspect `war_orbit/agents/v9/planner.py` and `war_orbit/agents/v9/policy.py`.
   - Determine whether `backbone_turn_frac` is undercounted, or whether the bot is still not selecting backbone plans enough.
   - Check if `reserve_hold` is stealing turns from useful transfers.
   - Check whether `front_budget`/`active_fronts` metrics match actual map behavior.

7. Should `main_front` be de-weighted now?
   - The bot seems to over-satisfy `mf=0.76+` while failing t120/benchmark.
   - Decide whether `_main_front_progress_score` is still too generous.
   - Propose a better objective that rewards conversion/closing instead of static mass.

## Desired Output

Return a concise engineering report:

- Root cause summary
- Evidence from logs/code
- Top 5 concrete fixes
- Which changes are safe to implement first
- What metrics should improve after each fix

Do not assume V10 is close to working just because train/eval sometimes rise. For gen 14/15, eval is not holdout because holdout was disabled. The primary facts to explain are: high main-front mass, low backbone, too many fronts, weak t100/t120 conversion, and poor benchmark history.
