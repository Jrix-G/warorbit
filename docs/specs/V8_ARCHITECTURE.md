# Orbit Wars V8 Architecture

Goal: avoid the V7 ceiling by moving learning from a late local multiplier to a policy-level ranker over candidate plans, with a value head and DAgger-style dataset aggregation.

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
C(s) = { V7 baseline, attack, expand, defense, comet, reserve }
```

This avoids the plateau of a late scalar reranker because the model can now
switch between qualitatively different plan families.

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
- orbit / center proximity statistics.

`E_c` consumes:
- variant identity,
- action count,
- ship fraction sent,
- source / target mix,
- average ETA,
- source production / reserve stats,
- coverage and concentration statistics.

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

## File Layout

- `bot_v8.py`: inference-time policy, candidate generation, model, feature extraction.
- `train_v8.py`: DAgger + pairwise ranking + value training.
- `test_v8.py`: smoke tests, shape checks, benchmark checks.
- `docs/specs/V8_ARCHITECTURE.md`: this spec.

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
