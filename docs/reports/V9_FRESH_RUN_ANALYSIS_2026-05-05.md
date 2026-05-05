# V9 fresh run analysis - 2026-05-05

## Observed run

Run: `v9_fresh_4p_guardian`

The useful peak was generation 4:

```text
gen=4 bench=0.466  bench_2p=1.000  bench_4p=0.411
eval=0.438 train=0.621 sel=0.372 gap=-0.028
benchmark_fronts=3.652 benchmark_bb=0.195
blockers=bench4p_low,fronts_high
```

The later run state drifted down:

```text
gen=8 bench=0.282  bench_2p=0.000/0  bench_4p=0.282/32
eval=0.375 train=0.375 sel=0.217 gap=0.093
benchmark_fronts=3.200 benchmark_bb=0.161
blockers=benchmark_low,bench4p_low,fronts_high
```

## Why 2p was strong

In 2p, V9's strategic assumptions line up well with the game:

- There is exactly one enemy, so `focus_enemy_id` is always correct. No coalition
  or leader/third-party ambiguity exists.
- Expansion and conversion are directly rewarded. The default policy weights
  favor `target_prod_gain`, `weak_enemy_focus`, `high_prod_focus`, `attack_move_frac`
  and neutral capture enough to snowball.
- The front problem is much simpler. In 2p, opening a second front is usually just
  pressure against the same opponent, not exposure to two other players.
- The benchmark logs show near-perfect 2p before the guardian pushed the schedule
  to 4p-only:

```text
gen=1 benchmark_2p=0.835/6
gen=2 benchmark_2p=1.000/5
gen=3 partial benchmark_2p=1.000/4
gen=4 benchmark_2p=1.000
```

Mathematically, the 2p signal is also lower variance. A 2p reward is close to a
binary strength test: if our expansion/attack policy is better than the opponent,
the result is stable. A 4p reward mixes our policy with two opponents' interactions,
third-party captures, leader punishment, and map timing.

## Why 4p degraded

The run did not fail because V9 ignored 4p fundamentals. It had good raw 4p
structure:

```text
xfer usually >= 0.48
bb usually >= 0.15 after gen4
lock = 1.0 in benchmark
```

The failure was spatial dispersion:

```text
gen=1 benchmark_fronts=3.655 global_fronts=6.998
gen=2 benchmark_fronts=3.657 global_fronts=7.089
gen=4 benchmark_fronts=3.652 global_fronts=6.906
gen=8 benchmark_fronts=3.200 global_fronts=5.934
```

This means the bot kept a logical target lock, but its real board footprint still
created too many contact zones. In 4p, that is expensive: every extra active front
creates more enemy owners able to punish overcommit, steal weakened planets, or
force defensive transfers.

The critical mismatch is:

```text
front_lock_turn_frac = 1.0
benchmark_fronts ~= 3.2 to 3.7
```

So `lock` is not enough. V9 can be mentally focused while physically spread out.

## Guardian feedback loop

The guardian tried to correct the weak 4p benchmark:

```text
if benchmark_4p < guardian_min_benchmark_4p:
    four_player_ratio += step
    benchmark_four_player_ratio += step
    candidate_diversity += 2 * step
```

This explains why the run eventually showed:

```text
benchmark_2p=0.000/0
benchmark_4p=32/32
diversity=2.00
explore=0.20
```

The run started as a mixed evaluator, then became effectively 4p-only because the
guardian kept responding to `bench4p_low`. That is useful for pressure-testing 4p,
but it hides the strong 2p signal and increases policy exploration at the same
time. More exploration can help escape a local optimum, but here it also moved the
weights away from the clean early generations.

Strict focus did not save this run because the current condition is late and hard:

```text
low_4p_streak >= 4 and benchmark_4p < 0.22
```

This run mostly stayed around `0.28-0.41`, so it was bad enough to fail guardian,
but not bad enough to trigger the strongest anti-dispersion mode.

## Practical conclusion

Generation 4 was the best mathematical compromise: huge 2p, near-threshold 4p,
low generalization gap, and strong backbone. It failed only because `fronts` was
far too high. The later generations did not discover a better 4p strategy; they
mainly shifted into 4p-only pressure with more diversity and lower benchmark mean.

For future runs, save every generation, keep the mixed 2p/4p signal visible, and
use the snapshots to choose a checkpoint by objective:

```text
Kaggle experiment: maximize benchmark_mean with nontrivial benchmark_4p
4p research: maximize benchmark_4p, then minimize fronts
2p duel test: maximize benchmark_2p, with sanity-check eval_mean
```
