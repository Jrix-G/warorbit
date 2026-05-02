# Orbit Wars V8 Architecture

Goal: avoid the V7 ceiling by moving learning from a late local multiplier to a policy-level ranker over candidate plans, with a value head and DAgger-style dataset aggregation.

---

## Changelog — 2026-04-30 audit V8.2

Audit complet `bot_v8_2.py` / `train_v8_2.py` / `benchmark_v8_2.py` + corrections.

### Bugs identifiés

**`bot_v8_2.py`**
- `latent_threat` sommait *tous* les ennemis joignables × 0.30 → en 4p le plancher
  garrison gonflait ~3× et bloquait l'expansion.
- `garrison_floor` identique en 4p early et 4p mid → `prod * 8` early avec 8
  planètes ⇒ ~24 ships locked/planète, raids early impossibles.
- `build_candidate_features` : tolérance angulaire `0.25 rad` trop serrée
  (orbits orientent les angles d'interception bien au-delà). La majorité des
  moves restaient non-classifiés ⇒ features `targets_attack/expand/defense`
  s'effondraient à 0 ⇒ ranker sans signal catégoriel.
- `agent(...) except Exception: return []` masquait toute erreur en forfait
  silencieux du tour, sans fallback V7.
- `reserve_hold` et `4p_conservation` quasi-redondants
  (`block_offense + defense_boost`) ; différenciés uniquement par
  `suppress_rear_staging` et magnitude.

**`train_v8_2.py`**
- `ProcessPoolExecutor` recréé à chaque `evaluate_params` → 8 spawns / gen +
  ré-imports notebooks (3000+ lignes) ⇒ racine du 14 min/gen observé.
- Eval seeds variaient avec `seed + 50000 + generation` ⇒ eval gen-N et
  gen-(N+5) sur tirages d'adversaires différents ⇒ promotion "best" sur
  seeds chanceux.
- Pas de séparation 2p / 4p dans les métriques.
- Promotion best : tout gain numérique, pas de plancher de non-régression
  par mode.
- `pairs=4` → 8 échantillons rank-shaped : SE ≈ 18% sur le gradient.
- `max_steps=500` pendant training : ES n'a pas besoin de jouer la fin de
  partie pour apprendre l'ordering des plans.
- `improved=1` au gen 1 même quand eval=0 (best initial -1) ⇒ log trompeur.
- `best_wr_*` lu de `best_path` même avec `--no-resume`.

**`benchmark_v8_2.py`**
- Pas de WR 2p / 4p séparé.
- Pas de temps par game ni p95.
- Pas d'histogramme plan-choice (impossible de vérifier mode-collapse).
- Pas de pool persistante.

### Corrections appliquées

**`bot_v8_2.py`**
- `latent_threat` → top-K (K=1) plus fort ennemi joignable, pas la somme.
  Constantes : `THREAT_LATENT_TOP_K = 1`, `THREAT_LATENT_DISCOUNT = 0.30`.
- `garrison_floor` → multiplier `GARRISON_FLOOR_4P_EARLY_RATIO = 0.55` quand
  `world.is_four_player and world.step < FOUR_P_EARLY_TURN_LIMIT`.
- `build_candidate_features` → projection forward (`proj > 0`) +
  `perp ≤ p.radius + 6`, abandon de la tolérance angulaire fixe ; couvre les
  angles d'interception orbital correctement.
- `agent` → fallback `bot_v7.agent(obs, config)` au lieu de `[]` sur exception.

**`train_v8_2.py`** (réécriture complète)
- `multiprocessing.Pool` **persistante** (`spawn`) avec `initializer=_worker_init`
  ⇒ notebooks importés une seule fois par worker.
- Un seul `pool.map` par génération : `2 × pairs × games_per_eval` tasks
  (+ `eval_games` quand `gen % --eval-every == 0`).
- Eval seeds **fixes** : `seed + 50000 + i`. Comparable gen-à-gen.
- Métriques 2p / 4p séparées : reward, WR par mode, n_games par mode.
- Promotion best : `gain ≥ --min-improvement` ET pas de régression par mode
  > `--min-mode-floor`.
- `--max-steps` séparé de `--eval-max-steps`.
- Nouveaux flags : `--eval-every`, `--eval-four-player-ratio`,
  `--min-improvement`, `--min-mode-floor`, `--log-jsonl`.
- Log JSONL `evaluations/v8_2_train.jsonl` (gen, train_*, eval_*, best_*,
  grad_norm, promoted, elapsed_min).
- `--no-resume` ignore aussi `best_path`.

**`benchmark_v8_2.py`** (réécriture complète)
- Pool persistante spawn + worker initializer.
- WR 2p, 4p, global ; median + p95 temps par game.
- Histogramme plan-choice via candidate-log callback.
- 4p picks rotatifs (anchor + 2 buddies) au lieu de répétition.

**Nouveaux fichiers**
- `run_v8_2_train_vps.sh` : wrapper 10h calibré 2 vCPU avec timestamp logs.

### Tests passés

```
py_compile bot_v8_2.py benchmark_v8_2.py train_v8_2.py            OK
benchmark mixed passive greedy --max-steps 80 --workers 1         2/0/0 (2p)
benchmark mixed bot_v7 --max-steps 80 --workers 1                 0/1/0 (poids zéro, attendu)
train smoke 0.1 min --no-resume                                   gen=1 promo OK
train resume 0.05 min                                             gen=2 reprise OK
train 3 min pairs=4 ggames=2 eval=16 workers=4 max=220 4p=0.5     gen=1 en 4:35 wall
```

### Estimations 10h après corrections

| Hardware  | workers | gens/10h | evals | gen wall |
|-----------|---------|----------|-------|----------|
| 8-core    | 8       | 170-200  | 40-50 | ~3 min   |
| 2 vCPU    | 2       | 100-120  | 20-24 | ~5-6 min |

Gain plausible local (60-game eval pool 8 notebooks) :
**WR 4p +5 à +15 pts**, **WR 2p -3 à +5 pts**, **global +3 à +10 pts**.
Kaggle ELO conversion : **+50 à +150**, distribution longue traîne, ~20%
chance zéro mouvement.

Le gros lift vient déjà des fixes hard-coded `#0/#1/#4/#5` dans `bot_v8_2`
(42% WR Kaggle vs ~21% V7 en 4p mesuré). ES sur ranker = incrémental.

---

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
C(s) = { V7 baseline, attack, expand, defense, reserve, transfer_push,
          4p_opportunistic*, 4p_eliminate_weakest*, 4p_conservation* }
```

`transfer_push` is always available and biases friendly staging transfers
from safe rear planets into forward launch platforms. The three starred
candidates are only generated when `IS_4P = True`.
This avoids the plateau of a late scalar reranker because the model can now
switch between qualitatively different plan families.

### 4p-Specific Candidate Plans

#### `4p_opportunistic`
Target only planets that are contested or recently fought over.
A planet is a candidate if: `target.ships < target.production * 3` (just changed hands),
or an enemy fleet is en route to it (they will fight, leaving it depleted).
Ships sent = minimum needed to take the post-battle residual.
Effect: the bot scavenges from other players' wars instead of initiating its own.
When no such target exists, V8.2 keeps the plan available as a guarded 4p tempo
plan. Existing rankers often score this row highly; mapping the no-opportunity
case to controlled expansion avoids turning that preference into passivity.

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
- friendly transfer fraction,
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
For shorter validation/training horizons, it can be enabled from the midgame
only when the position is already clearly favorable, garrison is healthy, and
multi-front pressure is low.
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

- `bot_v8_2.py`: inference-time V8.2 policy, V7-derived planner with safety fixes, candidate generation, ranker, value head, checkpoint I/O.
- `train_v8_2.py`: executable long-run ES trainer for the V8.2 candidate-plan ranker.
- `benchmark_v8_2.py`: reproducible SimGame benchmark for 2p, 4p, and mixed local validation.
- `bot_v8.py`: planned generic V8 inference-time policy name; not required by the current V8.2 implementation.
- `train_v8.py`: planned DAgger + pairwise ranking + value training name; superseded locally by `train_v8_2.py` until the richer oracle is implemented.
- `test_v8.py`: smoke tests, shape checks, benchmark checks.
- `docs/specs/V8_ARCHITECTURE.md`: this spec.
- `docs/reports/V8_DEFEAT_ANALYSIS.md`: empirical loss analysis, 40 Kaggle games.

## Implemented Baseline

Zero weights mean:
- candidate scores tie,
- tie-break chooses the first candidate;
- first candidate is the V7 action grammar with V8.2 safety fixes applied.

So the warm-start behaviour is conservative, but not byte-for-byte V7. This is
intentional: the empirical V7 loss analysis identified the inter-turn
overcommitment and garrison failures as hard bugs that should not be preserved
as a fallback.

---

## V8.2 Implemented Shape

V8.2 is now implemented in `bot_v8_2.py`.

### Runtime planner

The runtime path is:

```text
agent(obs)
  -> _build_world(obs) from V7
  -> _generate_candidates(world)
  -> build_state_features(world)
  -> score_candidates(state_features, candidates, world)
  -> argmax score, tie-break to candidate 0
```

Candidate 0 is `v7_baseline`, meaning V7's planner grammar with the V8.2
corrections enabled. Additional candidates bias the same grammar toward
expansion, attack, defense, reserve, friendly staging, and 4p-specific plans.

### Corrections shipped

- Fix #0: `already_committed_enough` checks own fleets already in transit to a
  target before sending more ships.
- Fix #1: `garrison_floor` forces a production-proportional reserve based on
  nearest enemy distance.
- Fix #2: `GARRISON_FLOOR_RATIO` trims emitted moves if too many ships would
  leave planets.
- Fix #3: `latent_threat` raises reserves for planets reachable by enemy
  capacity inside a short horizon.
- Fix #4: early 4p neutral captures use a lower margin and rear staging is
  delayed until turn 60.
- Fix #5: 4p pressure throttle blocks offense when multiple active fronts or
  top-enemy pressure make overextension likely.
- Early 4p override: when the bot has fewer than 6 planets before turn 70,
  has no active fronts, and still has an acceptable garrison ratio,
  `expand_focus` can override the learned ranker. This addresses the documented
  early_4p_crush pattern without disabling defensive throttles or the final
  garrison cap.

### Candidate plans

V8.2 currently emits up to nine candidate plans:

```text
0 v7_baseline
1 expand_focus
2 attack_focus
3 defense_focus
4 reserve_hold
5 4p_opportunistic
6 4p_eliminate_weakest
7 4p_conservation
8 4p_late_blitz
```

The 4p variants are only available in four-player states. `4p_late_blitz`
appears after turn 280 unconditionally, and can appear earlier inside short
training horizons only behind a favorable-position gate.

### Ranker and value head

The ranker is intentionally linear for safe long training:

```text
score(plan) = state_w[plan_id] dot state_features
              + candidate_w dot candidate_features
```

Dimensions:

```text
state_features:     24
candidate_features: 12
plan rows:           9
trainable weights: 253 including value head and bias
```

The value head is:

```text
value = value_w dot state_features + value_b
```

The value estimate is logged for training analysis; the current runtime
selection is ranker-only.

### Training and benchmarks

Short smoke:

```bash
python3 -m py_compile bot_v8_2.py benchmark_v8_2.py train_v8_2.py
python3 benchmark_v8_2.py --games-per-opp 1 --mode 2p --opponents bot_v7 --max-steps 80 --workers 1
python3 benchmark_v8_2.py --games-per-opp 1 --mode 4p --opponents bot_v7 --max-steps 80 --workers 1
```

Medium local benchmark:

```bash
python3 benchmark_v8_2.py --games-per-opp 8 --mode mixed --pool-limit 4 --max-steps 220 --workers 4
```

Long ranker run (10h calibrated):

```bash
python3 train_v8_2.py \
  --minutes 600 \
  --pairs 6 --games-per-eval 4 \
  --eval-games 60 --eval-every 4 \
  --max-steps 260 --eval-max-steps 500 \
  --four-player-ratio 0.65 --eval-four-player-ratio 0.70 \
  --workers 8 --pool-limit 8 \
  --min-improvement 0.015 --min-mode-floor 0.05
```

Why these numbers (rationale, not magic):

- `--workers 8` with the **persistent** spawn pool: notebook agents are imported
  once per worker (~3-5s each); the original draft re-created the pool 8 times
  per generation, which alone explained the 14-min/gen wall clock.
- `--max-steps 260` for training. ES does not need 500-turn games to estimate
  plan ordering — most of the policy signal sits in the first 200 turns of a
  match. Games average ~ish 4-5s for 2p / ~25-30s for 4p in this regime.
- `--eval-max-steps 500` for the periodic fixed-seed eval. This keeps the
  promotion criterion honest without paying the 500-turn cost on every ES
  sample.
- `--pairs 6 --games-per-eval 4` → 24 antithetic samples per generation. Below
  16 the rank-shaped gradient is mostly noise; above 32 the cost dominates.
- `--four-player-ratio 0.65` for training, 0.70 for eval: 4p is where the ELO
  gap is largest, but 2p signal is more stable so we keep some of it for ES.
- `--eval-every 4`: evaluation budget ≈ 25-30% of total run; promotion gated
  by `--min-improvement 0.015` AND no per-mode regression > 0.05.
- Eval seeds are **fixed across generations** (`seed + 50000 + i`), so
  generation-to-generation eval scores are directly comparable. The previous
  draft used `seed + 50000 + generation`, which made every eval an apples-to-
  oranges draw and caused noisy "best" promotions.

Resume is on by default:

```bash
python3 train_v8_2.py --minutes 600 --workers 8 --pool-limit 8
```

Sanity smoke (≈1 minute):

```bash
python3 train_v8_2.py --minutes 1 --pairs 2 --games-per-eval 1 --eval-games 4 \
  --max-steps 120 --workers 4 --pool-limit 2 --no-resume \
  --checkpoint /tmp/v8_2_latest.npz --best-checkpoint /tmp/v8_2_best.npz \
  --export-bot-checkpoint /tmp/v8_2_ranker.npz --log-jsonl /tmp/v8_2_train.jsonl
```

Important outputs:

```text
evaluations/v8_2_ranker_train_latest.npz
evaluations/v8_2_ranker_train_best.npz
evaluations/v8_2_ranker.npz
```

`bot_v8_2.py` auto-loads `evaluations/v8_2_ranker.npz` unless
`BOT_V8_2_NO_AUTOLOAD=1` is set.

### Metrics to watch (live in `evaluations/v8_2_train.jsonl`)

Each generation emits a JSON line with:

- `train_mean`, `train_wr_2p`, `train_wr_4p` — ES sample mean reward and
  per-mode WR. Noisy by design.
- `eval_mean`, `eval_wr_2p`, `eval_wr_4p` — fixed-seed eval (only every
  `--eval-every` generations). These are the honest signal.
- `best`, `best_wr_2p`, `best_wr_4p` — best score and per-mode WR currently
  promoted.
- `grad_norm` — ES update norm. Pre-clip values > 50 mean the loss landscape
  is locally flat (mostly draws) and the rank-shaping is amplifying noise. If
  it stays high for many generations without `eval` movement, raise `--pairs`
  or shrink `--sigma`.

Stop the run early if:

- `eval_mean` does not improve over 8 consecutive eval points.
- `eval_wr_4p` collapses below 0.20 (probably a regression bug, not learning).
- `grad_norm > 8` for 20+ generations with stagnant `eval_mean`.

Benchmark gate: run `benchmark_v8_2.py --mode mixed --workers 8
--games-per-opp 6` against the full pool **before** trusting any ranker
checkpoint. The training eval set is a small subsample.

### Remaining gaps

- `train_v8_2.py` is ES over the ranker, not the full DAgger short-rollout
  oracle described above.
- The value head is trained only indirectly by ES unless a future supervised
  dataset is added.
- SimGame is a fast local approximation; Kaggle/replay validation remains
  mandatory before treating any win rate as real.
- No explicit opponent-memory model is persisted across games yet.

## External References

- Evolution Strategies as a Scalable Alternative to Reinforcement Learning: https://arxiv.org/abs/1703.03864
- A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning: https://arxiv.org/abs/1011.0686
- Search-based Structured Prediction: https://proceedings.mlr.press/v15/daume11a.html
- Variance Reduction for Policy Gradient with Action-Dependent Factorized Baselines: https://openai.com/index/variance-reduction-for-policy-gradient-with-action-dependent-factorized-baselines
