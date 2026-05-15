# V15 architecture — peer review

Reviewer: senior game-AI / RL / statistics referee
Documents reviewed: `analysis/V15_ARCHITECTURE.md`, `analysis/V15_FINDINGS.md`, `V15_PROMPT.md`, `artOfWar.txt`.

---

## 1. Executive verdict

**Accept with major revisions.**

The strategic pivot (rebase on V7 because it is the only empirically winning ancestor, and confine new code to opt-in layers behind flags) is *correct* and is the single most important decision in the plan. The "if everything is off, V15 ≡ V7 bit-for-bit" invariant is well-engineered defense-in-depth. However, the plan as written conflates three different epistemic regimes — measured constants (Layer 2), behavioral cloning on an under-validated dataset (Layer 3), and PPO-on-a-tiny-reranker with sparse terminal rewards (Stage B) — and treats them as a linear pipeline with go/no-go gates that, on close reading, are statistically too lax in some places and operationally infeasible in others. The Kaggle ELO milestones (M4: 900, M6: 1200/1400) are aspirational and not derived from any quantitative model of layer contribution. The plan needs (i) honest uncertainty quantification on D1 patterns, (ii) a properly specified PPO objective (it currently does not exist as a learnable problem), (iii) acceptance of the fact that BC top-1 = 2.5% historically and the new ≥25% target needs justification beyond "random is 12%", and (iv) a fallback path that does not depend on Layer 4 being feasible at all. Address §3 and the resubmission should be near-final.

## 2. Strengths

- **Architectural invariant** (`V15_ARCHITECTURE.md` §1, line 36: "Si toutes les couches sont off → bot_v15 == bot_v7 bit-pour-bit"). This is the single best engineering decision and unique among the V8–V14 iterations that all silently regressed.
- **Constants-as-dataclass refactor** (§2 Layer 1): replacing module-level globals with `V15Config` kills the monkey-patching class of bugs that almost certainly contributed to V14's regressions.
- **Bisectable flags per patch** (§2 Layer 2, end): each D1-derived constant change is independently togglable. This is precisely how to handle a multi-patch heuristic refit and is rare discipline.
- **Top-K ceiling on the ranker** (§2 Layer 3, "ranker scoré uniquement parmi le top-K (K=8) candidats de V7"): hard upper bound on damage the ML can do. Correctly applied insurance.
- **Honest enumeration of risks** (§5, §8 H1–H4): the author flags H3 (search latency) and H4 (curriculum B4 too optimistic) themselves. Good intellectual honesty.

## 3. Critical issues (must-fix)

1. **PPO objective is not specified well enough to implement** (§3 Stage B, line 100: "PPO sur les seuls weights du re-ranker, Layer 1+2 reste fixe. Reward = +1 win, −1 loss, +0.001·planet_share_delta").
   - (a) The re-ranker selects 1 candidate out of K each turn. The "action" PPO would optimize is a softmax over K logits, but the candidates themselves are state-dependent and change every turn. Standard PPO assumes a stationary action space; this is a contextual-bandit-per-turn embedded in a sparse-reward MDP. You cannot just plug `gpu_trainer.py` at it.
   - (b) Credit assignment: with ~150 turns/game and terminal reward ±1, the per-candidate advantage estimate has variance dominated by trajectory noise. A 5k-param MLP has no chance of extracting signal at the per-candidate level without either (i) reward shaping at every turn (you have `+0.001·planet_share_delta` but that is too small to dominate the terminal noise) or (ii) GAE with a learned value head — which the plan does not mention.
   - (c) Fix: specify Stage B as **policy-gradient with REINFORCE-style per-turn baseline** (subtract mean candidate score) rather than PPO, *or* keep PPO but add an explicit value head and document GAE λ, clip ε, KL target. Better still: replace Stage B with **CMA-ES / population-based search over the dataclass constants** — 30-50 scalars, dense fitness from 200-game evals, no credit-assignment problem. This is what the top of competitive heuristic ladders historically uses.

2. **BC success criterion is mis-calibrated** (§3 Stage A, line 87: "top-1 accuracy ≥ 25% (random baseline ~12% pour K=8 candidats)").
   - (a) Random over K=8 is 12.5%, fine. But `V15_PROMPT.md` line 31 explicitly states the prior BC attempt got top-1 = 2.5% and top-3 = 5% on essentially the same corpus. Going from 2.5% → 25% is a 10× jump, not a tuning improvement.
   - (b) The 2.5% figure was over the *full* action space; the new 25% is over the K=8 V7-prefiltered candidates. These are different denominators. The plan does not state whether 25% is over `top-K from V7's candidate generator` (apples-to-apples not possible because V7 didn't generate the prior corpus) or over `all legal actions`. **The numbers are not comparable as written.**
   - (c) Fix: define the BC top-1 metric explicitly as "fraction of replay turns where the top-player's chosen action appears in V7's top-K and the ranker scores it #1". Provide the **ceiling** estimate: fraction of replay turns where the chosen action even appears in V7's top-K. If that ceiling is, say, 40%, then a 25% target is plausible; if it's 15%, the milestone is unreachable by construction.

3. **D1 patterns are correlations, not interventions; the plan treats them as causal constants** (§2 Layer 2 table).
   - (a) Pattern 1 ("multi-source coordination") and Pattern 4 ("commit hard ≥90%") in `V15_FINDINGS.md` are both **post-hoc behaviors of winning players in the games they won**. A leading player has more sources available, hence more multi-source moves; a player about to win can commit 90% because they have already secured production. Both behaviors are partially endogenous to "I am winning".
   - (b) `V15_FINDINGS.md` line 18 attempts to control for this by noting "ELOs of paired won/lost in each bucket are within 30 points". This is a **between-player** control, not a **within-game** control. Two equally-rated players can still split into the won-this-game and lost-this-game group precisely because one got lucky on the opening and the resulting state enabled multi-source play. ELO equality does not eliminate the leading-position confound.
   - (c) Fix: before baking constants, run an **A/B simulation experiment** — V7 with `MULTI_SOURCE_TOP_K=14, SEND_RATIO=0.90` vs V7 baseline, 200 games × 4 modes. If +3% holds vs the existing zoo, the constants survive M1. If not, the patterns are confounds and should be downgraded to soft priors only. The plan does this at M1 but only on the **aggregate** patch set; bisecting per-patch (which the flags allow but the milestone does not require) should be mandatory at M1.

4. **Latency budget for Layer 4 is unsubstantiated** (§2 Layer 4, line 78: "Budget: 1.0s/turn max").
   - (a) Depth-1 expectimax over K=5–8 candidates with WorldModel rollouts of 8 turns × 3 opponent models = 5 × 8 × 3 = 120 simulated turns per real turn, each requiring full combat resolution. The `V15_PROMPT.md` (line 98) already conceded that the V7/SimGame forward model is "too slow" and demanded a stripped `v15_fast_sim.py` — which the architecture document never mentions. There is no plan to build the fast sim.
   - (b) Without a profile of WorldModel cost per simulated turn, the 1.0s budget is wishful. Kaggle's hard cap is ~2s but submission timeouts are a leading cause of leaderboard regressions.
   - (c) Fix: add a D3-prelude milestone "M3a: WorldModel-based 8-turn rollout from a representative state benchmarked at ≤80ms" *before* M3 itself, and gate Layer 4 work on it. If M3a fails, either build the fast sim or downgrade Layer 4 to a pure heuristic post-strike eval (no rollout).

5. **Curriculum gating asymmetry**: §3 Stage B requires ≥80% / ≥60% / ≥50% / ≥40% per stage to advance, but never specifies sample size for the gating eval, nor a confidence interval. With 200 games, a 50% true winrate has a 95% CI of roughly [43%, 57%]. **A stage that "passes" at 51% may actually be 44% true.** This is the same statistical fragility that let V8–V14 regress.
   - Fix: every gate must be "lower bound of 95% Wilson CI ≥ threshold", and sample sizes must be specified (200 minimum, 500 preferred for the ≥50% gates).

## 4. Statistical & methodological concerns

- **D1 sample sizes**: `top/won` has n=50, `top/lost` has n=83 in `V15_FINDINGS.md`. The headline multi-source frac difference (0.40 vs 0.27) has 95% CIs roughly [0.27, 0.53] and [0.18, 0.36] — they overlap. The pattern is **directionally consistent across buckets** (mid/won: 0.45 vs mid/lost: 0.25 with n=866/2009 is rock-solid), so the conclusion survives, but the doc should report CIs explicitly and avoid quoting top-bucket numbers as if they were independent confirmation.
- **The ELO≤30 control** (line 18 of FINDINGS) is the right idea but applied incorrectly. ELO controls for *player skill ex ante*, not for *position in this game ex post*. Multi-source frequency is a behavior conditioned on state, and state correlates with winning by definition. The honest framing is "behavior X co-occurs with winning, may be partially causal", not "X is a robust strategic prior".
- **Pattern 3 (first-attack timing not the lever) is the most reliable finding** in the audit because it is a *negative result* with consistent direction across buckets, and it directly contradicts `artOfWar.txt` Rule 1 (which was derived from 95 2p games and explicitly does not generalize to 4p, as Pattern 3 correctly notes). This single finding deserves more weight than the four "do more X" findings.
- **Safety net via flag bisect**: the plan claims that with all flags off, `bot_v15 == bot_v7` bit-for-bit (line 36). This is testable only if the V15Config dataclass perfectly preserves V7's global-state order-of-application. The M0 test ("1000 obs aléatoires, même action") is necessary but not sufficient — it samples states, not transitions. Recommend: also a **full-game determinism test**: same seed, same opponent, V15 (flags=off) and V7 must produce identical action streams for 10 full games. If they diverge mid-game, there is hidden state and the safety net is illusory.

## 5. Algorithmic concerns

- **PPO on the 5k-MLP re-ranker is ill-posed** (see §3.1 above). The credit-assignment problem is severe. Even if it learns, the policy gradient through a re-ranker that selects among V7-generated candidates can only learn to **re-order** V7's preferences; it cannot learn to **prefer states V7 never visits**. This bounds the strategic improvement to the convex hull of V7's candidate distribution, which is exactly the V7 ceiling the plan claims it will surpass. **The architecture cannot, in principle, exceed V7's strategic horizon.** It can only execute V7's strategy with better tie-breaking. The H1 hypothesis (line 156: V7 plafonne parce que constantes sous-optimales) is the only honest path to >900 ELO; CMA-ES on constants gets you that directly, without RL.
- **Depth-1 expectimax with V7-as-opponent-model is mostly self-play hallucination**: you are scoring "what if opponents play exactly like me" rather than "what if opponents play like the actual Kaggle field". `V15_PROMPT.md` line 99 was wiser: "50% greedy V12, 30% V14, 20% top10_mimic". The architecture doc collapsed this to plain V7 (§2 Layer 4, line 76) and loses the anti-meta value. **This is a regression from the brief.** Fix: keep the mixed opponent model, weighted to match the actual Kaggle field distribution if possible.
- **Curriculum advance criteria gates**: §3 Stage B's thresholds are *static*, but a 5k-param reranker with PPO will plausibly oscillate near the boundary. Without a moving-average gate and CI floor (see §4 fix above), a stage will alternately pass and fail and waste training compute. Stage B5 ("ELO interne ≥ +50 par cycle") is the only one specified relative to history and is the best-designed gate.

## 6. Engineering risks

- **Latency**: see §3.4. No measurement, no fast sim, only an aspiration.
- **Training compute**: 5k × 25k games × ~150 turns ≈ 3.75e9 forward passes. On CPU this is hours; on a 1080 it is minutes if batched. Tractable. No concern.
- **Replay data quality**: `V15_PROMPT.md` line 24 reveals **4978 unmatched actions and 340 too-many-turns** out of ~50k samples — that is **~10% silent data loss**. The plan (line 84) hand-waves "Re-extraction si besoin avec le matcher amélioré" but does not budget time for it or specify what "amélioré" means. The BC ceiling estimate (§3.2 fix) depends critically on this. **Mandatory action**: profile the 4978 unmatched cases before Stage A starts. If they are systematically the multi-source actions (because the matcher does single-action assumption), the entire D1 finding about multi-source frequency is consistent but the BC dataset is biased toward single-source moves, and the ranker will learn the wrong thing.
- **Reproducibility**: the plan mentions seedable search (V15_PROMPT.md §4) but the architecture doc omits it. With PPO and self-play, exact reproducibility is hard; ask for "same-seed, same-game-trace" determinism at least for the inference path (Layer 1+2+3 frozen weights → identical output).

## 7. Alternatives the user should seriously consider

1. **CMA-ES (or simple genetic search) over V15Config constants, skipping Layers 3 and 4 entirely.** V7 has ~30–50 scalar knobs. With 200-game evals as fitness, CMA-ES can climb the local heuristic-ceiling in 100–500 generations, dominated by simulation cost. No credit assignment, no BC, no RL pathology. This is, empirically, what the top of heuristic competition ladders use, and it is what `H1` (the only well-supported hypothesis) directly motivates. **Strongly recommended as the primary track**, with Layers 3–4 as research side-projects.
2. **Opening book + V7 only.** `V15_FINDINGS.md` notes top-10 first-attack window is t29–35 in 4p with low variance, and `artOfWar.txt` Rule 8 highlights the t40–80 decisive window. Encode the top-10's first 25 turns as a state→action lookup (k-NN over a small feature set) and let V7 take over after. This captures most of Patterns 1, 2, 4 without any ML, in a few hundred lines. Cheap, robust, and additively combinable with option 1.
3. **MCTS over V7's candidates with V7 as rollout policy**, instead of depth-1 expectimax with mixed opponent. Same simulator cost, but MCTS handles the credit-assignment-in-search problem better at the latency the plan budgets (1s allows ~500–2000 playouts depending on rollout length). Stronger in long-horizon games like Orbit Wars than depth-1.

## 8. Specific recommendations (priority order)

1. **Build M0 first, then stop**: implement Layer 1 (`bot_v15_core.py`, `V15Config` dataclass, regression test of full-game determinism vs V7). Do not write any other code until M0 passes on 10 full-game traces. This catches the hidden-state class of bugs that killed V8–V14.
2. **Replace Stage B (PPO on reranker) with CMA-ES on V15Config constants** as the primary improvement track. Keep PPO as an explicit research branch with a 1-week timebox; if it does not beat CMA-ES by week 4, archive it.
3. **Audit the 4978 unmatched replay actions** before any BC training. Report the failure mode. If multi-source actions are over-represented in unmatched, fix the matcher; if not, document the bias and proceed.
4. **Per-patch bisect at M1**: the plan allows it via flags, but make it a *required* part of the milestone, not optional. Each of the 9 Layer-2 patches must independently demonstrate ≥+0% winrate (with CI lower bound ≥ −1%) before being included in the aggregate.
5. **Specify all gates as Wilson 95% CI lower bound ≥ threshold**, with sample sizes ≥200 for ≥50% thresholds and ≥500 for ≥40% thresholds. Update the milestone table accordingly.
6. **Add M3a (latency profile of WorldModel)** before committing to Layer 4. If 8-turn rollout > 80ms, kill Layer 4 and route the engineering effort into MCTS-with-fast-sim or option 7-alt-3 above.
7. **Restore the mixed opponent model** for Layer 4 (50% V12 / 30% V14 / 20% top10_mimic, as `V15_PROMPT.md` originally specified). Self-as-opponent loses anti-meta value.
8. **Define the BC top-1 metric numerator/denominator explicitly** in `V15_ARCHITECTURE.md` §3 Stage A, and compute the V7-candidate-coverage ceiling on a 5k held-out sample before declaring the 25% target.
9. **Honor `artOfWar.txt` only for 2p**: Pattern 3 in D1 already contradicts Rule 1 for 4p. The architecture doc should explicitly note which `artOfWar.txt` rules apply to which mode in the V15Config, or you will re-import a 2p prior into a 4p path silently.
10. **Tag every D-milestone in git** as the original brief requested (V15_PROMPT.md §4, line 140). The architecture doc dropped this guardrail.

---

End of review. The plan is recoverable and the core decision (rebase on V7, layered with kill-switches) is right. The work needed before implementation is concentrated in §3 issues 1–3 and §6 (replay data audit) — perhaps three days of analysis. After that, recommend executing option-7-alt-1 (CMA-ES) as the primary track and keeping Layers 3–4 as opt-in research.
