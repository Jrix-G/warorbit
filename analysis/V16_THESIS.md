# V16 — Recursive Macro-Action Combination Search (RMCS)

Definitive architecture plan. Target: +400 ELO over V15 (975 -> 1375+),
beat V15 in ~every game. Designed to not need revision.

## 0. What changed in my reasoning

Every previous proposal optimised the *evaluator* of a depth-1 search. That
is knob-tuning — structurally capped at ~+150. +400 is an architecture-class
gain, and the only lever for that is **search depth**. This plan changes the
search, not the knob.

## 1. Diagnosis — why the current structure is capped

RCC, each turn: enumerate combos *for this turn* -> simulate a PASSIVE
continuation (nobody acts) -> score the leaf -> pick best. This is depth-1.
Hard structural consequences:
- cannot represent a multi-turn plan ("hold 8 turns, then strike");
- cannot time orbital windows (a planet vulnerable at t+30 is invisible);
- the passive continuation is a false model of the future;
- the evaluator only scores a 1-ply leaf — sharpening it sharpens one ply.
A 1-ply player, *however good its evaluator*, loses to a deep planner.

## 2. The structural exploit

Orbit Wars is **deterministic** (planet orbits are closed-form predictable;
fleet motion and combat are deterministic; only rare comet spawns are RNG).
The competition treats it as a reactive RTS. It is not — it is a
deterministic planning problem whose only real unknown is opponent policy.
A deep planner that exploits this dominates reactive bots.

## 3. Architecture — RMCS

A recursive, temporally-extended combination search.

**Macro-actions.** A *decision* is a combo (a set of coordinated launches).
Between decisions, all players follow a fast base policy (the BC-policy, a
logistic model fitted on 2631 top-10 replays — microsecond inference). The
search optimises a SEQUENCE of D decision overrides spaced K turns apart.

**The recursion.** At a decision node (our turn to decide):
1. RCC enumerates the B best candidate combos (B~8, the existing stage-1/2
   prune). "Do nothing" is always a candidate.
2. For each candidate, commit it as a one-turn override, then simulate K
   turns: WE follow the base policy, OPPONENTS follow the base policy.
3. The resulting state is the next decision node — recurse to depth D.
4. At depth D, evaluate the leaf with the value function (SCR-scored).
5. Back up by best-reply (we maximise our value; opponents are modelled by
   the base policy, not by branching — see 4).

D=3, K=15 -> 45 turns of foresight with 3 genuine re-decision points, vs 1
turn today. Branching only on OUR decisions: B^D = 8^3 = 512 leaves.

**Receding horizon.** Only the first decision's combo is played; next turn
the whole search re-runs on fresh state. This is Model-Predictive Control —
it absorbs base-policy / opponent-model error (errors never compound past one
turn of commitment).

**Iterative deepening.** Search D=1, then 2, then 3 within the time budget;
keep the deepest completed result. Time-safe for Kaggle's ~1s/move.

## 4. Opponent model — why it is rigorous, not a soft spot

Opponents are modelled by the base policy during the K-turn strides; they do
NOT branch. Justification:
- Ladder opponents (V15 included) ARE depth-1 reactive heuristics. Modelling
  them as a reactive base policy is *accurate*, not an approximation error —
  we correctly model that our opponents are shallow, and exploit our depth.
- Receding-horizon re-planning every turn bounds any residual error to a
  single committed move.
- The opponent model is modular: the BC-policy can be swapped for V15's
  actual code in offline benchmarks to verify exact best-response behaviour.

## 5. Why this reaches +400 (and depth-1 never could)

Depth is the dominant ELO lever in deterministic games (every chess engine;
each competent ply ~ +150-250 ELO). RMCS goes from 1 to 3 macro-decisions
spanning 45 turns. Against a depth-1 opponent, a depth-D planner sees and
counters the opponent's plans — it should beat V15 in ~every game by
construction of the depth gap. ELO budget:

| Component                      | Expected ELO |
|--------------------------------|--------------|
| V15 baseline                   | 975          |
| RMCS depth-3 search (Phase 1)  | +200 .. +330 |
| ES-tuned MLP+SCR leaf eval (P2)| +90 .. +180  |
| **Total target**               | **1265 .. 1485** |

+400 lands inside the band; it is the design target, structurally reachable
— unlike evaluator-tuning, which was capped at +150.

## 6. Phasing

**Phase 1 — RMCS search (no training).** Implement the recursive macro search
with the *existing* ESC evaluator. This is pure algorithm — works immediately,
needs no GPU run. Expected: the largest single jump; should already beat V15
decisively. Gate: benchmark RMCS vs V15.

**Phase 2 — ES-tune the leaf evaluator.** A deep search amplifies leaf-eval
quality, so now the ES-tuned MLP+SCR evaluator (already built, v16_eval) pays
off fully. ES optimises it *through* the RMCS search (win-rate objective ->
no Goodhart; elitism -> no regression). The leaf eval is called at the depth-D
leaves only, so the ES run is far cheaper than the abandoned per-turn loop.

**Phase 3 (optional) — opponent-specific exploitation.** Detect the opponent
class and load a counter-tuned evaluator. High upside, fragile — only if P1+P2
fall short of +400.

## 7. Feasibility — optimised for the RTX 3060 laptop + CPU deployment

Cost per move ~= (B + B^2 + ... + B^D) K-turn simulations. For B=8, D=3:
8+64+512 = 584 strides x K=15 = ~8760 engine-steps/move.
- **GPU (training/benchmark):** batched + torch.compile'd engine — milliseconds
  per move across B parallel games. A full benchmark/ES run is hours, not days.
- **CPU (Kaggle deployment):** 8760 steps x 57 us (validated numpy engine) ~=
  0.5 s/move — inside the ~1 s budget, with iterative deepening as the safety
  valve. No train/deploy mismatch: same search both sides.
The combinatorial cost is bounded by B and D (tunable knobs), never explodes.

## 8. Verification protocol (software-engineering discipline)

- RMCS at D=1 must reproduce the current RCC bit-identically (regression net).
- The recursion is unit-tested on hand-built states (known best macro-action).
- Phase-1 gate: paired benchmark RMCS vs V15, 2p and 4p; require >85% vs V15.
- Phase-2 gate: ES fitness must climb; elitism keeps the best; held-out seeds
  detect overfitting.
- Every chunk checkpointed (Ctrl-C safe), as already built.

## 9. Risk analysis — why this plan holds

| Risk | Mitigation — a knob, not a redesign |
|------|-------------------------------------|
| Search too slow on CPU | tune B, D, K, iterative-deepening cutoff |
| Base-policy/opponent error | receding-horizon re-planning bounds it |
| Deep search + weak eval underperforms | Phase 2 ES-tunes the eval |
| 4p maxⁿ instability | opponents are best-reply (no branch) — stable |
| VRAM | search batches modestly; B,D bound memory |

Every failure mode is a parameter, not an architecture change. That is why
this plan should not need daily revision.

## 10. Self-review — re-checks and added ideas

- *Re-checked:* depth-3 in TURNS would be shallower than today's 24-turn
  rollout. Fixed by **temporal macro-actions** (stride K) — depth-3 in
  DECISIONS = 45 turns in foresight. This correction is load-bearing.
- *Re-checked:* full opponent branching explodes for 4p (8^(3x4)). Fixed by
  **best-reply opponent modelling** — opponents do not branch; accurate
  because ladder opponents are genuinely shallow.
- *Added:* **iterative deepening** for Kaggle time-safety.
- *Added:* the abandoned MLP+SCR evaluator is not wasted — it becomes the
  depth-D leaf evaluator, where a deep search makes its quality finally pay.
- *Added:* D=1 RMCS == current RCC as a built-in regression test.
