# Session 2026-05-01 - V8.2 Stabilization

## Scope

Strict scope for this pass:

- `bot_v8_2.py`
- `train_v8_2.py`
- `benchmark_v8_2.py`
- V8 documentation

No changes were made in `neural_network/`.

## Starting Point

Recent 4p-only training runs were stable but stuck:

```text
best_score=0.3250
eval_mean=0.250 for multiple generations
eval_median=0.0
train_mean varied from ~0.062 to ~0.562
promo=0
```

The trainer was producing gradients and the checkpoint pipeline worked, but
training gains did not transfer to fixed-seed eval.

## Diagnosis

### 1. The plateau was not only a trainer problem

The trainer was correctly refusing noisy promotions. The repeated
`eval_mean=0.250` with `eval_median=0.0` indicated a weak and unstable 4p
policy, not a broken ES loop.

### 2. `4p_late_blitz` was not reachable in the short training regime

The docs originally described `4p_late_blitz` as available after `step > 280`.
The active local run used:

```text
--max-steps 120
--eval-max-steps 260
```

So the finisher was either completely absent from training or too rare to
affect short eval.

### 3. Early 4p collapse still mattered

Benchmarks showed many games ending before a finisher could matter. This
matches the documented `early_4p_crush` pattern from `V8_DEFEAT_ANALYSIS.md`:
the bot often fails to reach enough planets before the 4p economy closes.

### 4. Benchmark instrumentation was incomplete

The old benchmark only showed chosen plans. It could not distinguish:

- a plan not being generated,
- a plan being generated but never selected,
- a plan dominating the ranker.

This made it hard to identify candidate-set failures.

### 5. Promotion logic was wrong for 4p-only eval

In 4p-only runs, `eval_wr_2p` is `0.0` because no 2p games are evaluated. The
old promotion gate could still compare this absent 2p score to a historical
best 2p score. That could block a valid 4p-only improvement.

### 6. Runtime checkpoint export could overwrite the deployed bot with a non-promoted state

At the end of training, `train_v8_2.py` exported the current latest params to
`evaluations/v8_2_ranker.npz`, even if that latest state had not been promoted.
This meant a bad exploratory generation could become the runtime bot while
`best_score` remained unchanged.

## Changes Made

### `bot_v8_2.py`

#### Controlled earlier `4p_late_blitz`

Added guarded midgame availability:

```text
LATE_BLITZ_EARLY_TURN = 90
LATE_BLITZ_MIN_GARRISON_RATIO = 0.36
LATE_BLITZ_MAX_ACTIVE_FRONTS = 1
LATE_BLITZ_MIN_SHIP_LEAD = 1.05
LATE_BLITZ_MIN_PROD_LEAD = 1.05
```

`4p_late_blitz` remains unconditional after turn 280, but can appear earlier
only when:

- the bot has healthy garrison,
- there is at most one active front,
- the bot leads the strongest enemy in ships or production.

Reason: expose a finisher inside short 120-turn ES games without reintroducing
early overextension.

#### Early 4p expansion override

Added a bounded override before ranker argmax:

```text
if 4p
and step < 70
and my_planets < 6
and no active fronts
and garrison_ratio >= 0.28
then prefer expand_focus if it has moves
```

Reason: directly target the documented early_4p_crush failure while preserving
defensive throttles and final garrison caps.

#### `4p_conservation` differentiation

`4p_conservation` now allows rear-to-front redistribution while still blocking
new offense. This makes it more distinct from `reserve_hold`.

#### Opportunistic mode investigation

Instrumentation showed that `4p_opportunistic` could dominate plan choice.
An overly strict gate was tested, but it degraded short eval. The final version
keeps the plan available and maps the no-scavenge case into guarded tempo
behavior rather than removing the plan.

### `benchmark_v8_2.py`

Added candidate availability instrumentation.

The benchmark now reports:

- `Plan-choice histogram`
- `Candidate availability histogram`

This separates "plan exists" from "plan selected".

### `train_v8_2.py`

#### Mode-aware promotion

Promotion now checks only modes that are actually present in the eval set.

In 4p-only eval:

- compare against `best_wr_4p` when available,
- ignore absent 2p floor checks,
- keep non-regression checks for active modes.

This prevents `eval_wr_2p=0.0/0 games` from blocking valid 4p improvements.

#### Safer runtime export

At the end of a run, `evaluations/v8_2_ranker.npz` now exports the best
checkpoint when available, not the latest unpromoted state.

Reason: avoid deploying a worse exploratory generation.

### `docs/specs/V8_ARCHITECTURE.md`

Updated V8.2 docs to reflect:

- earlier gated `4p_late_blitz`,
- early 4p expand override,
- guarded opportunistic fallback behavior.

## Test Results

### Syntax

```bash
python3 -m py_compile bot_v8_2.py train_v8_2.py benchmark_v8_2.py
```

Result: passed.

### Benchmark before candidate instrumentation fix

Short 4p benchmark showed mode collapse:

```text
4p_opportunistic: 201 choices, 93.9%
expand_focus:      12 choices,  5.6%
```

This showed that the ranker could over-prefer one 4p plan.

### Benchmark after candidate instrumentation and adjustments

Short 4p benchmark:

```text
Plan-choice histogram:
  v7_baseline       129  78.7%
  expand_focus       25  15.2%
  4p_conservation    10   6.1%

Candidate availability histogram:
  v7_baseline       164  14.4%
  expand_focus      164  14.4%
  attack_focus      164  14.4%
  defense_focus     164  14.4%
  reserve_hold      164  14.4%
  4p_opportunistic  147  12.9%
  4p_conservation   147  12.9%
  4p_eliminate_weakest 25 2.2%
```

The plan distribution is more diverse, but the benchmark still produced:

```text
4p W/L/D = 0/2/0
```

This is not enough to claim the plateau is broken.

### Short training probes

Two short runs were executed after candidate changes:

```text
gen=0034 eval=0.167
gen=0035 eval=0.083
gen=0036 eval=0.167
gen=0037 eval=0.167
```

Conclusion: the attempted behavior change did not immediately improve eval.
Because those runs moved `latest` into a worse exploratory region, `latest`
and runtime export were restored from the best checkpoint parameters while
keeping the current generation number.

## Current State

The plateau is not proven solved yet.

What is improved:

- benchmark now exposes candidate availability,
- 4p-only promotion is no longer blocked by absent 2p scores,
- runtime bot export is protected from unpromoted latest states,
- `4p_late_blitz` and early 4p expansion are now reachable under controlled
  conditions.

What remains unresolved:

- short probes still did not exceed `eval_mean=0.250`,
- `eval_median` remains `0.0`,
- many 4p losses happen before finisher plans matter,
- the best known checkpoint remains `best_score=0.3250`.

## Commands to Run Next

Recommended 20-minute run:

```bash
python3 train_v8_2.py \
  --minutes 20 \
  --four-player-ratio 1.0 \
  --eval-four-player-ratio 1.0 \
  --pairs 4 \
  --games-per-eval 2 \
  --eval-games 12 \
  --max-steps 120 \
  --eval-max-steps 260 \
  --eval-every 1 \
  --workers 8 \
  --pool-limit 4
```

Benchmark with candidate visibility:

```bash
python3 benchmark_v8_2.py \
  --mode 4p \
  --games-per-opp 4 \
  --workers 8 \
  --pool-limit 4
```

Longer, more reliable run:

```bash
python3 train_v8_2.py \
  --minutes 60 \
  --four-player-ratio 0.85 \
  --eval-four-player-ratio 1.0 \
  --pairs 6 \
  --games-per-eval 3 \
  --eval-games 24 \
  --max-steps 160 \
  --eval-max-steps 300 \
  --eval-every 1 \
  --workers 8 \
  --pool-limit 4
```

## Success Criteria

Watch for:

- `eval_mean > 0.250` repeatedly,
- `eval_wr_4p > 0.281` so the mode-aware promotion can beat the current best
  4p reference,
- `best_score > 0.3250`,
- `eval_median > 0.0`,
- benchmark histogram showing `expand_focus`, `4p_conservation`, and
  `4p_late_blitz` appearing in the relevant phases.

## Next Step if Plateau Persists

If `eval_mean` stays under `0.250`, the next pass should focus on 4p early
survival and expansion quality, not on late blitz:

- inspect actual lost 4p trajectories around turns 30-90,
- measure planet count at t60/t70,
- measure garrison ratio at collapse,
- add benchmark logging for per-game final steps and score margins,
- consider a dedicated early 4p candidate that expands to nearest safe neutrals
  with hard garrison preservation.

