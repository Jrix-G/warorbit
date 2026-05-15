# V15 D1 — replay audit findings

Source: `analysis/v15_replay_audit.json` (2630 episodes, 10520 player-records, manifest ELO p95=1373 p99=1536 max=1833).

## Bucketed numbers (mean per record)

| bucket | outcome | n    | elo  | first_action | first_attack | mission ex/at/sup | ship_alloc ex/at/sup | send 90-100% | multi-src frac | peak / final planets |
|--------|---------|------|------|--------------|--------------|-------------------|----------------------|--------------|----------------|----------------------|
| top    | won     | 50   | 1576 | 3.6          | 31.1         | .13/.47/.39       | .10/.49/.40          | 82%          | **0.40**       | 25.7 / 23.6          |
| top    | lost    | 83   | 1558 | 4.3          | 35.4         | .17/.51/.31       | .14/.57/.28          | 68%          | 0.27           | 9.2 / 1.0            |
| high   | won     | 241  | 1383 | 4.2          | 29.2         | .14/.42/.44       | .11/.47/.41          | 72%          | **0.47**       | 26.4 / 25.7          |
| high   | lost    | 588  | 1380 | 3.8          | 29.1         | .23/.43/.32       | .21/.49/.28          | 77%          | 0.24           | 7.8 / 0.5            |
| mid    | won     | 866  | 1146 | 4.3          | 31.1         | .13/.48/.38       | .11/.53/.35          | 67%          | **0.45**       | 27.8 / 27.0          |
| mid    | lost    | 2009 | 1150 | 5.1          | 32.0         | .24/.49/.25       | .22/.56/.20          | 70%          | 0.25           | 8.3 / 0.5            |
| low    | won     | 1473 | 964  | 4.5          | 30.9         | .11/.46/.42       | .10/.52/.37          | 57%          | **0.52**       | 27.9 / 27.0          |
| low    | lost    | 5210 | 936  | 4.7          | 32.3         | .25/.49/.26       | .22/.56/.20          | 62%          | 0.27           | 8.3 / 0.5            |

ELOs of paired won/lost in each bucket are within 30 points — outcomes are not a side-effect of skill gap, so the behavioral deltas are signal.

## Top 5 actionable patterns

1. **Multi-source coordination is the single strongest winning signal.** Across every bucket, winners use ≥2 distinct source planets in **0.40–0.52** of their action turns; losers in **0.24–0.27**. Same gap at 1576 ELO and 936 ELO, so this is a robust strategic prior, not a side-effect of leading. → V14's `focus_finish` and three/four-source plans only fire once `allow_finish=True`; we should fire 2-source coordinated strikes much earlier and weight `MULTI_SOURCE_TOP_K` candidates higher in the ranker.

2. **Losers over-fork into neutrals; winners consolidate ships to the front.** Losers send **22–25 %** of their ships to neutrals vs winners' **10–14 %**, and only **20–28 %** to own-planet support vs winners' **35–44 %**. Mission counts mirror this: losers' moves are 23–25 % `expand`, winners' 11–14 %. → Tighten 4p expansion: `_OPENING_4P_TURNS=50` is too generous; add a kill-switch (e.g. stop `opportunistic_expand` once `my_planets>=6` or `step>=40`) and raise the floor of "support / staging" candidate scores.

3. **First-attack timing is not the lever.** Winners and losers all open attack between turns **29 and 35** in 4p; the gap is at most 4 turns and not consistent in direction (top winners attack 4 turns earlier than top losers; high bucket has the opposite sign). `artOfWar.txt`'s "open at turn 2-8" rule is from 2p analysis and does **not** generalize to 4p — V14's `pressure` candidates from step 0 are probably premature. → For 4p, suppress single-source `pressure` before turn ~25 and use 0–25 for expansion + multi-source build-up.

4. **When you commit, commit hard (≥90 % of source ships).** Top winners send 90-100 % of available ships in **82 %** of moves; top losers in **68 %**. The pattern is muddier in lower buckets (because both sides ramble), but among top players it's the cleanest behavior split. → Raise `FOUR_PLAYER_ROTATING_SEND_RATIO` from 0.72-0.78 to **~0.90** for the `closer` / `top10_mimic` profile, and bake `_FOCUS_SEND_RATIO` up from 0.70 to ~0.92.

5. **Top players are not doing something secret.** Winning behavior at ELO 1576 ≈ winning behavior at ELO 964 (multi-source, low-expand, high-support, 90-100 send). The top bucket just executes the same patterns more consistently and avoids losing positions. → This argues **for** the heuristic+search plan: there is no hidden RL-only knowledge between us and rank 10, so beating the heuristic ceiling is mostly a matter of (a) wiring the four patterns above in, and (b) using lookahead to catch the consistency mistakes the V14 scorer makes.

## Open risks for V15

- Top bucket sample is small (133 records). Patterns 1–4 hold in the much larger high/mid buckets, so they are unlikely to be artifacts.
- The audit's "expand/attack/support" classifier uses `_infer_target` with the same projection geometry as V14; it can be wrong for grazing trajectories. The headline gaps are large enough to absorb the noise.
- All numbers are 4p. Don't port these constants into the 2p path without re-running the audit on a 2p corpus.

## What this means for D2 (refit) and D3 (search)

- D2 should: (i) raise `closer` send ratios to 0.90, (ii) add a `top10_mimic` profile that drops `_OPP_SEND_RATIO`, gates `opportunistic_expand` after step 40, and increases `MULTI_SOURCE_TOP_K`, (iii) add a multi-source bonus to `_four_player_heuristic_scores` for `focus_finish` candidates with ≥2 sources, even before step 40.
- D3 should: rank candidates partly by post-strike planet-share, so the search naturally prefers multi-source moves that finish a target in one go.
