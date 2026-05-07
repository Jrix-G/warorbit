# V11 Audit Prompt

You are analyzing Orbit Wars V11 in a local repository. The goal is to decide whether the
current V11 branch is moving toward a genuinely stronger bot, or whether it is still only
mixing heuristics without converting them into wins.

## Context

V11 was built from the V7 tactical core and then extended with:

- a top1-like supply chain,
- persistent weak-enemy focus,
- stronger opening capture pressure,
- 3-source and 4-source swarms,
- a lightweight task-search bias layer,
- and a direct ES trainer (`train_v11_fast.py`) for V11-specific constants.

The relevant files are:

- `bot_v11.py`
- `train_v11_fast.py`
- `benchmark_v11.py`
- `analysis/v11_profile_report.py`
- `docs/reports/TOP1_REPLAY_STRATEGY_2026-05-05.md`
- `analysis/v7_fast_replay_summary.md`

## What We Already Learned

Local benchmark:

- V7 4p local benchmark: `10/24 = 0.417`
- V11 4p local benchmark: `10/24 = 0.417`

Training signal:

- baseline fixed eval: `0.083`
- best observed fixed eval: `0.583`
- last observed value in the run dropped back down, which means the search is noisy and the best checkpoint must be preserved explicitly

Behavioral fingerprint:

- V11 improved transfer flow, but not always in the right phase
- early over-transfer can starve the opening
- too much garrison protection can prevent expansion
- too much focus can block opportunistic captures

## What You Must Analyze

1. Read the V11 code and identify which behaviors are now close to top1 and which are still missing.
2. Explain why V11 can reach a strong fixed eval on one generation, then lose stability later.
3. Compare the V11 fingerprint against the top1 replay fingerprint:
   - opening timing,
   - actions per active turn,
   - transfer ratio,
   - ship mass per move,
   - planets at t60 and t100.
4. Determine whether the current bottleneck is:
   - search noise,
   - over-strong opening transfer,
   - over-strong garrison,
   - weak conversion after front buildup,
   - or poor target selection.

## Success Criteria

The target is not a cosmetic improvement.
The target is to push the fixed eval toward `0.6` and keep the behavior coherent.

Concrete signs of progress:

- fixed eval stabilizes above `0.6`,
- 4p benchmark stops falling behind V7,
- profile metrics move toward the top1-like fingerprint instead of only inflating one metric,
- the best checkpoint remains best after several generations, not just for one spike.

## Deliverable

Return:

- the main reason V11 is still unstable,
- the one or two changes most likely to move fixed eval toward `0.6`,
- and the risk of those changes on the 4p benchmark.

