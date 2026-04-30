# Orbit Wars V8 Architecture

Goal: avoid the V7 ceiling by moving learning from a late local multiplier to a policy-level ranker over candidate plans, with a value head and DAgger-style dataset aggregation.

## V8 Prerequisites — V7 Bug Fixes

Before any training, three hard bugs in V7's candidate generation must be fixed.
Training on a broken base policy embeds broken behaviors into the dataset irreversibly.

### Fix #0 — Already-in-flight check (CRITICAL)

Measured: 88 wasted missions in a single 4p game (ep75680588).
Root cause: `planned_commitments` prevents double-counting *within* a turn but not *across* turns.
The bot sees a partially-captured target, computes `ships_needed > 0`, and sends more fleets — ignoring
the 20-30 own fleets already en route from previous turns.

```python
def my_ships_en_route_to(target_id, world):
    return sum(
        entry[2]
        for entry in world.arrival_ledger.get(target_id, [])
        if entry[1] == world.player
    )

# In plan_moves target loop:
already = my_ships_en_route_to(target.id, world)
needed = world.ships_needed_to_capture(target.id, turns, planned_commitments)
if already >= needed:
    continue  # enough in flight, do not escalate
if target.owner != -1 and already > needed * 0.5:
    continue  # enemy target, partial commitment — wait for resolution
```

### Fix #1 — Garrison floor proportional to production

`min_garrison` stays at 1-5 ships across all loss games. Any enemy fleet of 6 ships
takes any planet unconditionally.

```python
def garrison_floor(planet, world):
    dist = world.nearest_enemy_dist(planet)
    if dist < 20:   return max(5, planet.production * 8)
    if dist < 35:   return max(5, planet.production * 5)
    return max(5, planet.production * 3)

# keep_needed returns max(timeline_result, garrison_floor(planet, world))
```

### Fix #4 — Early 4p expansion thresholds

In 2p games the bot reaches 10+ planets by t60. In 4p it averages 3-4 planets at t70.
The neutral capture margin is calibrated for 2p; in 4p the contested window is 2-3× shorter.

```python
IS_4P = len(obs.initial_planets) > 8
if IS_4P and world.step < 60:
    NEUTRAL_MARGIN = 1.05          # was TWO_PLAYER_NEUTRAL_MARGIN_BASE = 1.2
    REAR_STAGING_MIN_TURN = 60     # no rear staging before mass is established
```

---

## Why V7 Plateaus

Current V7 learns only in a narrow slot:

```text
π_theta(s) = argmax_{a in A(s)} b(s,a) * exp(theta^T f(s,a))
```

Where:
- `A(s)` is already heavily filtered by hard rules.
- `b(s,a)` is the handcrafted base value.
- `f(s,a)` is a 15-D mission feature vector.
- `theta` is trained by ES.

This creates a piecewise-constant policy in `theta`. Small parameter changes often do not change the selected action at all, so gradient-free search sees high variance and weak signal.

The second bottleneck is distribution shift. Training on a fixed rollout distribution is not enough for a sequential policy that changes the states it visits.

## V8 Principle

Learn at the level of complete candidate plans, not only mission multipliers.

```text
score_phi(s, c) -> R
value_psi(s) -> [0,1]
pi(s) = argmax_c score_phi(s, c) + lambda * value_psi(s')
```

Where:
- `c` is a candidate plan.
- `s'` is a short-rollout successor state.
- `lambda` balances immediate candidate quality and longer-horizon value.

In the implemented V8 shape, the candidate set is intentionally small and
structured:

```text
C(s) = { V7 baseline, attack, expand, defense, comet, reserve,
          4p_opportunistic*, 4p_eliminate_weakest*, 4p_conservation* }
```

The three starred candidates are only generated when `IS_4P = True`.
This avoids the plateau of a late scalar reranker because the model can now
switch between qualitatively different plan families.

### 4p-Specific Candidate Plans

#### `4p_opportunistic`
Target only planets that are contested or recently fought over.
A planet is a candidate if: `target.ships < target.production * 3` (just changed hands),
or an enemy fleet is en route to it (they will fight, leaving it depleted).
Ships sent = minimum needed to take the post-battle residual.
Effect: the bot scavenges from other players' wars instead of initiating its own.

#### `4p_eliminate_weakest`
Concentrate all offensive resources against the single weakest enemy
(`min over enemies of: total_ships + production * 20`).
Stop attacking others entirely until that enemy is eliminated or drops to 0 planets.
Effect: each elimination removes 1/3 of total pressure and redistributes resources.

#### `4p_conservation`
Triggered when `active_fronts >= 2` (2+ enemies have fleets en route to my planets).
Zero new offensive missions. All ships go to reinforcement of threatened planets.
Releases when `active_fronts < 2` for 10 consecutive turns.
Effect: prevents the self-destruction loop where the bot attacks A while B and C
take undefended planets from behind.

## State and Candidate Features

Use two encoders:

```text
z_s = E_s(s)
z_c = E_c(s, c)
```

`E_s` consumes global state statistics:
- ship counts and production totals by owner,
- planet ownership ratios,
- turn phase,
- comet pressure,
- threat pressure,
- reserve ratios,
- orbit / center proximity statistics,
- **[4p]** `n_active_fronts`: number of enemies currently attacking my planets,
- **[4p]** `weakest_enemy_fraction`: weakest enemy total ships / my total ships,
- **[4p]** `contested_planet_count`: planets with 2+ enemy fleets inbound,
- **[4p]** `inter_enemy_fight_intensity`: sum of enemy fleets targeting other-enemy planets (they are busy),
- **[4p]** `garrison_ratio`: my ships on planets / my total ships (key collapse predictor),
- **[4p]** `min_garrison_normalized`: min garrison across my planets / avg production.

`E_c` consumes:
- variant identity,
- action count,
- ship fraction sent,
- source / target mix,
- average ETA,
- source production / reserve stats,
- coverage and concentration statistics,
- **[4p]** `targets_weakest_enemy`: fraction of actions targeting the weakest enemy,
- **[4p]** `targets_contested`: fraction of actions targeting contested/depleted planets,
- **[4p]** `already_committed_ratio`: my ships already en route to targets / ships sent this plan.

## Training Loss

### 1. Pairwise ranking loss

For each state `s`, let `c+` be the oracle candidate and `c-` a rejected candidate.

```text
L_rank = mean log(1 + exp(-(score_phi(s,c+) - score_phi(s,c-))))
```

This is the main objective. It directly trains candidate ordering, which is what the policy needs.

### 2. Value loss

Let `y in [0,1]` be the terminal outcome or normalized margin.

```text
L_value = mean (value_psi(s) - y)^2
```

### 3. Behavior cloning regularizer

For states from expert or near-expert rollouts:

```text
L_bc = -log p_phi(c_expert | s)
```

In the implementation this is approximated by cross-entropy over the softmax of candidate scores.

### 4. Weight decay

```text
L = L_rank + alpha * L_value + beta * L_bc + gamma * ||params||^2
```

In code, the current implementation uses a linear ranker over:
- 33 state features,
- 32 candidate-plan features,
- 8 state-plan interaction features.

The score head is a softmax classifier over the 6 plan candidates.
The value head is a separate linear regressor on state features.

## DAgger Loop

DAgger fixes the train-test mismatch by aggregating states visited by the current policy.

Pseudocode:

```text
D = empty
pi = initial policy
for iter in 1..N:
    trajectories = rollouts(pi against training opponents)
    for each visited state s:
        candidates = generate_candidates(s)
        for each candidate c in candidates:
            r_c = short_rollout_score(s, c)
        c+ = argmax_c r_c
        add (s, c+, {c != c+}, final_outcome) to D
update score_phi and value_psi on D
evaluate on held-out opponents
keep best checkpoint
```

## 4p Training Considerations

4p games have structurally different dynamics from 2p and must not be pooled naively.

### Separate oracle labeling per mode

In 2p, `c+ = argmax_c margin_after_H_steps` works well.
In 4p, margin is noisy because 3 opponents interact. Use:

```text
c+_4p = argmax_c (ship_delta * 0.3 + planet_delta * 0.5 + eliminated_enemies * 0.2)
```

The `eliminated_enemies` term captures the asymmetric value of 4p kills
(removing one player collapses the threat landscape discontinuously).

### Stratified sampling

Training batch composition:
- 50% 2p games (stable signal)
- 50% 4p games (harder, but where the ELO gap is largest)

Within 4p, oversample states where `active_fronts >= 2` and `garrison_ratio < 0.30`
— these are the states where V7 fails most and where the 4p candidates add value.

### Late-game 4p blitz

At `step > 280` in 4p, most resources have been spent by other players.
Add a `4p_late_blitz` candidate: all offense, ignore garrison floor.
This is only valid late because early use triggers the overextend pattern.
The model learns to activate it via `turn_phase` in `E_s`.

```python
if IS_4P and world.step > 280:
    candidates.append(generate_4p_late_blitz(world))
```

---

## V8.1 Training Improvements

The implementation now keeps the same architecture, but makes the signal less
fragile:

- hard-negative mining: if the model disagrees with the oracle, the sample is
  duplicated in replay so mistakes matter more than easy states;
- multi-horizon oracle: each labeled state is scored with a short and a longer
  rollout, then blended;
- benchmark stability score: checkpoint selection is not based only on the
  global win rate, but on a weighted combination of global, minimum per-opponent
  and mean per-opponent win rates.

These changes do not increase model size. They just make the learning target
and the validation criterion closer to the real objective.

## Short-Rollout Oracle

The oracle is not a human expert; it is a local search label:

```text
r_c(s) = margin_after_H_steps(s with candidate c, opponent_policy)
```

This is much less sparse than terminal win/loss and is closer to the actual induced utility of the candidate.

## Why This Should Plateau Less

1. The learned object is a ranking over full candidate plans, not a scalar multiplier after filtering.
2. The loss is pairwise and dense at the state level, so each state yields many comparisons.
3. DAgger keeps the training distribution near the policy's own visited states.
4. The value head adds a longer-horizon signal without forcing the ranker to solve the entire game alone.
5. The simulator is used as an oracle generator and validator, not as the single source of truth for final policy quality.

## Implementation Sequence

Order matters. Each step must benchmark before the next starts.

```text
Step 0 (pre-ML): Apply V7 bug fixes
  - Fix #0: already_in_flight check           → benchmark 4p, target +14pts
  - Fix #1: garrison_floor                    → benchmark 4p, target +10pts
  - Fix #4: early 4p expansion thresholds     → benchmark 4p, target +5pts
  → Gate: 4p winrate > 40% before proceeding to Step 1

Step 1: Extend candidate set
  - Add 4p_opportunistic, 4p_eliminate_weakest, 4p_conservation
  - Add 4p_late_blitz (t>280 only)
  - Zero weights → fallback to V7 baseline (safe warm-start preserved)

Step 2: Extend feature vectors
  - Add 6 new E_s features (4p dynamics)
  - Add 3 new E_c features
  - Retrain with existing DAgger loop, stratified 50/50 2p/4p

Step 3: 4p oracle calibration
  - Separate labeling function for 4p (ship_delta + planet_delta + eliminated_enemies)
  - Oversample active_fronts>=2 states

Step 4: Benchmark and iterate
  - Target: 4p > 55%, 2p > 70%, overall > 62%
```

## File Layout

- `bot_v8.py`: inference-time policy, candidate generation, model, feature extraction.
- `train_v8.py`: DAgger + pairwise ranking + value training.
- `test_v8.py`: smoke tests, shape checks, benchmark checks.
- `docs/specs/V8_ARCHITECTURE.md`: this spec.
- `docs/reports/V8_DEFEAT_ANALYSIS.md`: empirical loss analysis, 40 Kaggle games.

## Implemented Baseline

Zero weights mean:
- candidate scores tie,
- tie-break chooses the first candidate,
- first candidate is the existing V7 baseline plan.

So the warm-start behaviour remains conservative and safe.

## External References

- Evolution Strategies as a Scalable Alternative to Reinforcement Learning: https://arxiv.org/abs/1703.03864
- A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning: https://arxiv.org/abs/1011.0686
- Search-based Structured Prediction: https://proceedings.mlr.press/v15/daume11a.html
- Variance Reduction for Policy Gradient with Action-Dependent Factorized Baselines: https://openai.com/index/variance-reduction-for-policy-gradient-with-action-dependent-factorized-baselines
