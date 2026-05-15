# V15 NEW-GEN BOT — ENGINEERING BRIEF FOR CLAUDE OPUS 4.7

You are a senior engineer with deep expertise in game AI, heuristic design, search algorithms (minimax/expectimax/MCTS), replay mining, and pragmatic ML. You are dropped into the `warorbit` codebase and given **6 weeks** to take the existing `bot_v14.py` from ~400 ELO (rank ~2000) to **top 10** on the Kaggle Orbit Wars leaderboard (~1400 ELO target). Most competitors above us are heuristic, not RL. Your output is a single deliverable: a working **`bot_v15.py`** that beats V14 on every benchmarked opponent and demonstrates measurable gains on Kaggle within 7 days, with a roadmap for weeks 2–6.

This is engineering, not research. Prefer rapid validated steps over theoretical purity. Always validate against V14 before committing changes.

---

## 1. CURRENT STATE — READ FIRST

### Repository layout
- `C:\Users\Capitaine\Documents\GitHub\warorbit\` — main repo, `cwd` here.
- `bot_v14.py` (462 lines) — current production bot. Hybrid: heuristic candidate generator + neural ranker (`v14_core.V14Scorer`) + V12 fallback. Has 4-player profiles (`base`, `anti_pool`, `closer`, `eco`, `deep_eco`).
- `v14_core.py` — `build_v14_features`, `get_candidates`, `V14Scorer` class, candidate selection.
- `bot_v12.py`, `bot_v13.py` — older versions kept as fallbacks / benchmark opponents.
- `V14_RUNBOOK.md` — V14 training & promotion workflow (BC → fine-tune → gate).
- `benchmark_v14.py`, `gate_v14.py`, `benchmark_v14_candidates_4p.ps1` — benchmark + promotion gates.
- `analyze_replays.py`, `analyze_opponent_strategies.py`, `analyze_competition.py`, `analyze_kovi_loss.py` — existing replay analyzers. **Read and reuse before rewriting.**
- `SimGame.py` — local simulator. Slow. Use `c_engine` (in `neural_network/c_engine/`) or `kaggle_environments` for speed.
- `artOfWar.txt` — strategic notes from prior analysis.

### Replay corpus (your gold)
- **Raw 4-player top-10 replays**: `D:\warorbit_kaggle_raw\orbit-wars-top10-episodes-2026-05-04\episodes\episodes\*.json` — **2631 episodes**, manifest at `manifest.csv` (one row per episode, with `scores`, `submission_ids`, `create_time`).
- `replay_corpus/imitation_4p_top10_v1/` — pre-processed BC dataset (50k samples, 13 shards) from these replays. Mission distribution: support 19738, expand 7905, attack 22357. Skipped: 4978 unmatched actions, 340 too-many-turns. **Use this as a sanity check, do not retrain from it.**
- `replay_corpus/kaggle_top10_2p_filtered_v1/` — 2-player variant.

Each raw replay JSON contains the full game state per turn for all 4 players. Parse them yourself — `analyze_replays.py` already has loaders, reuse.

### What we already tried and burned 3 months on (do not repeat)
- Pure PPO/self-play from scratch → policy collapse, can't beat `starter` heuristic.
- BC from these same replays → top1 = 2.5%, top3 = 5%. **Too weak to be a policy on its own**, useful only as a feature ranker.
- PPO from BC checkpoint with aggressive shaping → supervisor saturated penalties chasing a metric that included forced no-ops. Recently patched (see `neural_network_gpu/src/gpu_trainer.py` legal/forced noop separation) but still not the path to top 10.

The RL branch continues in the background. **Your job is the heuristic+search path, which is what the top of the leaderboard actually uses.** V15 must not depend on a GPU at submission time. Inference must fit in Kaggle's per-turn latency budget (~2s).

### Competition target
- Leaderboard: rank ~2000 → top 10 in 6 weeks.
- ELO gap: ~1000 points.
- Top 10 is mostly heuristic + lookahead. Few RL bots in the top 50.

---

## 2. STRATEGIC FRAME

The competition rewards three things:

1. **Domain knowledge** — micro-tactics: when to send, how many ships, which target, when to defend.
2. **Lookahead** — 2-3 ply search of "if I send X to Y, what does the opponent do?" — beats reactive play.
3. **Anti-meta** — counter the specific top-50 bots, not a Platonic optimum.

V14 has good domain knowledge but no real lookahead and no anti-meta. V15 fixes both.

### V15 architecture (three layers)

```
turn t →
  ┌─ 1. heuristic_candidate_gen()    ← inherits V14 candidate generator
  ├─ 2. expectimax_search(depth=2-3) ← NEW: lookahead over top candidates
  └─ 3. nn_tiebreaker (optional)     ← V14 scorer used only on near-ties
```

The search dominates. The NN intervenes only when the search's top-2 candidates are within ε. This guarantees:
- Robustness: heuristic worst case ≥ V14.
- Strength: search adds 200-400 ELO on top of pure heuristic in games this combinatorial.
- Latency: prune aggressively (top-k=5 candidates × depth=2 = manageable).

---

## 3. DELIVERABLES (in order)

### D1 (day 1-2) — Replay-driven strategic audit
Build `analysis/v15_replay_audit.py` that, on the 2631 raw replays, computes:

- **Per-rank statistics** (split bots by their final ELO score from `manifest.csv`):
  - Average turn of first attack.
  - Average ratio expand/attack/support actions over the game.
  - Average % production allocated to attack vs defense vs expansion.
  - Average distance of targets at each game phase (early/mid/late).
  - Ship-send ratio distribution (when they attack, do they send 50%, 70%, 95%?).
  - Multi-source attack frequency (concentrating fleets from multiple planets).
- **Win-correlated patterns**: which behaviors correlate with winning, controlling for ELO.
- **Loss patterns for V14-likes**: simulate V14 against top-10 traces (or use existing `analyze_kovi_loss.py` pattern) — where do we systematically lose?

Output: `analysis/v15_replay_audit.json` + a 1-page markdown summary `analysis/V15_FINDINGS.md` listing the top 5 actionable patterns. **Do not write code beyond this until the audit is done.**

### D2 (day 3-4) — Heuristic refit
Patch the V14 heuristic constants (`v14_core.py`, `bot_v14.py` profiles) using D1 findings. Concretely:

- Calibrate `FOUR_PLAYER_ROTATING_SEND_RATIO`, `FOUR_PLAYER_ROTATING_TURN_LIMIT`, `MULTI_SOURCE_TOP_K`, `WEAK_ENEMY_THRESHOLD`, `ELIMINATION_BONUS`, etc. to match top-10 distributions.
- Add a new profile `top10_mimic` that bakes in the most-winning observed configuration.

Validate: `benchmark_v14.py` style — V14 vs V14_refit, 64 games × 4 modes (2p random, 2p greedy, 4p mixed, 4p top10_mimic vs top10_mimic). Refit must win every matchup by ≥+3% winrate. Otherwise revert.

### D3 (day 5-7) — Lookahead search
Implement `v15_core.expectimax_search(obs, depth=2)`:

- Generate top-k=5 candidates via `v14_core.get_candidates`.
- For each, roll the game forward 1 ply using a **lightweight forward model** (not the full `SimGame` — too slow). Read `SimGame.py` to extract just the ship-movement and combat-resolution math; write a stripped `v15_fast_sim.py` (~150 lines). Validate output against `SimGame` on 100 random states.
- At depth 2, model opponent as: 50% greedy V12, 30% V14, 20% top10_mimic. Mix the resulting evaluations weighted.
- Eval function = `v14_core.V14Scorer` applied to the resulting state's candidates, summed/discounted.
- Hard latency budget: 1.5 s per turn measured on the slowest replay state. If exceeded, reduce depth or top-k. Log and abort if can't fit.

Wire into `bot_v15.py`:
```
1. candidates = get_candidates(obs)
2. top_k = candidates[:5]
3. best = expectimax_search(obs, top_k, depth=2)
4. if margin(best, second_best) < epsilon:
       best = nn_tiebreaker(top_k, scorer_scores)
5. return select_actions(best, ...)
```

Validate via `gate_v15.py` (copy of `gate_v14.py` updated): V15 must beat V14 by ≥+5% winrate over 128 games per matchup before submission.

### D4 (week 2) — First Kaggle submission + iterate
Submit V15. Observe ELO over 48h. Pull the games where V15 lost (Kaggle exposes them). Build `analyze_v15_losses.py` (copy `analyze_kovi_loss.py`). Identify top 3 loss patterns. Patch.

### D5 (weeks 3-4) — Anti-meta + search depth
Once V15 is at rank ~200, identify the top 30 bots that beat V15 by replay analysis. Build counter-patches (specific profile overrides triggered when the opening 10 turns of an opponent match a known signature).

Tune search to depth 3 if latency permits. Add MCTS option as a swappable strategy.

### D6 (week 5) — NN tiebreaker
At this point, V14's NN scorer is probably underused. Train a *small* tiebreaker net (~100k params, MLP, ~30 features) only on the question "given these 5 candidates and the current state, which one wins?" — using the replay corpus filtered to winning trajectories. The output augments the search's eval function. Only ship if it adds ≥+2% winrate.

### D7 (week 6) — Polish
- 3-5 V15 variants with different tunings, A/B on Kaggle.
- Fix all crashes / timeouts seen in production.
- Final submission.

---

## 4. CONSTRAINTS & GUARDRAILS

- **Never break V14**: every commit must pass `gate_v14.py` against the previous best as a safety net. If V15 underperforms V14 on any single benchmarked matchup by >2%, do not submit — fix first.
- **Latency**: `python benchmark_v14.py --modes 4p 2p --max-steps 220` on the V15 candidate must complete without per-turn timeout warnings (Kaggle ≈ 2s/turn cap).
- **No new dependencies** in the submission bot. NumPy + Python stdlib only. The training/analysis pipeline can use anything.
- **Determinism**: every search/eval must be seedable. Reproduction is non-negotiable for debugging.
- **Repo hygiene**: project rules in `CLAUDE.md` — minimize tokens, no markdown spam, only modified files in responses.
- **Branch**: work on `main` (this is the user's pattern) but tag `v15-d1`, `v15-d2`, etc. before each major step so rollback is trivial.

---

## 5. RISKS YOU MUST FLAG (before week 3 commits)

- **Search too slow**: if depth-2 lookahead exceeds 1.5s/turn on any replay state, you must downgrade. Report it.
- **V14 already near plateau**: if D2 refit gives <+2% winrate, the heuristic ceiling is closer than expected; search becomes the only lever.
- **Top-10 bots are RL too**: if D1 audit reveals top-10 patterns inconsistent with any reasonable heuristic (e.g., highly non-greedy moves justified only by deep simulation), heuristic+search caps lower than 1400. Report this honestly — the user accepts a "top 30 ceiling" outcome if argued from data.
- **Kaggle ELO drift**: top 10 moves too. Add +50-100 ELO to your internal target.

---

## 6. WHAT TO DO IF STUCK

If after D1 the replay audit yields no clear actionable patterns: **stop and ask the user**. Do not invent strategy. The whole plan is built on the assumption that top-10 behavior is mineable. If it's not, we need a different approach (e.g., direct opening-book replay of top-10 first 30 turns).

If at D3 the lookahead doesn't beat the refit by ≥+5%, **stop and ask**. Either the eval function is too weak (improve `V14Scorer`) or the opponent model is wrong (look at the actual replay distribution).

---

## 7. SESSION KICK-OFF CHECKLIST

Before you write any code:

1. Read `CLAUDE.md` (root + nested).
2. Read `bot_v14.py`, `v14_core.py`, `V14_RUNBOOK.md`, `artOfWar.txt`.
3. Run `python analyze_replays.py --help` and `python analyze_opponent_strategies.py --help` to see what already exists.
4. Run `benchmark_v14.py --bots v14 v13 v12 v7 --modes 4p --games 8` to baseline V14 numbers on your machine. Record them.
5. Open 5 random replay JSONs and confirm you can parse them. Verify the manifest score → submission_id mapping.
6. **Then** start D1.

Report back at end of D1 with `analysis/V15_FINDINGS.md` and your 5 actionable patterns. Wait for user feedback before D2.

---

## 8. SUCCESS DEFINITION

End of week 6:
- V15 ranked ≤ 30 on Kaggle (stretch: ≤ 10).
- Reproducible benchmark vs V14 showing ≥+10% winrate aggregate.
- Documented post-mortem of which interventions paid off and which didn't.
- Codebase clean enough that a V16 iteration is straightforward.

If you hit top 10 by week 4, freeze and ship variants. If you're stuck at rank 100 by week 4, pivot — talk to the user about MCTS, opening books, or a focused anti-meta-only sprint.

Begin with the kick-off checklist. Do not skip steps. Do not write `bot_v15.py` on day 1.
