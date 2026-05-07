# V11 Implementation and Goal

## Scope

V11 is a new branch built from the V7 tactical core, not from the V10 proxy-training line.
The intent was to recover the parts of V7 that already won games, then add a top1-inspired
tempo layer: faster opening, heavier supply chain, stronger focus on the weak enemy, and
multi-source pressure.

## What Was Added

### Tactical core

- Reuse of the V7 world model and target scoring machinery.
- Arrival ledger, timeline simulation, comet prediction, and interception logic.
- Garrison floors tied to production and front distance.

### V11 behavior layer

- Persistent focus on the weak enemy in 4p, with a lightweight memory.
- Opening capture floors to stop tiny probes from dominating early play.
- A top1-like transfer/supply chain that pushes ships from rear planets toward the front.
- Multi-source swarm generation, including 3-source and 4-source pressure.
- A search bias layer that favors `expand`, `transfer`, `strike`, and `finish` depending on board state.

### Training support

- `train_v11_fast.py` trains the V11 constants directly with antithetic ES.
- `benchmark_v11.py` compares V11 to V7 on the same local 2p/4p pools.
- `analysis/v11_profile_report.py` measures the behavioral fingerprint:
  - `actions_per_active_turn`
  - `transfer_ratio`
  - `planets_t60_mean`
  - `planets_t100_mean`
  - `garrison_p10_median`

## What The Runs Show

The key point is that V11 is no longer behind V7 locally.

- Local 4p benchmark: V7 and V11 both reached `10/24 = 0.417` on the same pool.
- ES training on V11 showed a baseline `0.083` and a peak `0.583` on 12 fixed eval games.
- The training curve is still noisy, so the best checkpoint matters more than the last one.

Interpretation:

- V11 has a real improvement region.
- The optimizer can find it.
- The search is not stable enough yet to guarantee that the final generation is the best one.

## Current Bottleneck

The remaining issue is not that V11 has no signal.
The issue is that it still needs more stable convergence toward the top1-like fingerprint:

- higher tempo in 4p,
- enough transfer flow to create a strong front,
- enough capture mass to convert that front into ownership,
- but not so much early transfer that opening expansion collapses.

## Target

The immediate objective is not a vague "better bot".
The concrete target is:

- keep V11 at least at parity with V7 locally,
- then drive the fixed `eval` score toward `0.6`,
- then verify that the 4p benchmark follows, not just the training set.

## Practical Reading Of `eval = 0.6`

For this project, `eval >= 0.6` means:

- the current parameter set wins a clear majority on the fixed evaluation pool,
- the improvement is not a one-generation spike,
- and the behavior remains coherent on the profile metrics.

## Useful Commands

```bash
.venv/bin/python train_v11_fast.py \
  --minutes 30 \
  --workers 6 \
  --pairs 4 \
  --games-per-eval 1 \
  --eval-games 12 \
  --eval-every 1 \
  --match-4p-ratio 0.95 \
  --eval-4p-ratio 1.0 \
  --max-steps 220 \
  --out evaluations/scorer_v11_kaggle
```

```bash
.venv/bin/python benchmark_v11.py \
  --games 24 \
  --workers 6 \
  --max-steps 220 \
  --modes 4p
```

```bash
.venv/bin/python analysis/v11_profile_report.py \
  --games 8 \
  --mode 4p \
  --max-steps 220
```

