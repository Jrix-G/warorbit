# Session Report - Neural Network Orbit Wars

Date: 2026-04-30

## 1. Purpose of This Report

This document captures the current state of the `neural_network/` package so the next session can resume without re-discovering the same issues.

It summarizes:

- what was implemented;
- what was validated;
- what failed in real gameplay;
- what the auto-correction loop changed;
- what remains to be fixed before the network becomes strategically useful.

---

## 2. Current Package State

The package `neural_network/` now exists as an isolated research workspace with:

- NumPy-based encoder, model, policy, reward, storage, benchmark, trainer;
- Orbit Wars Kaggle adapter;
- real-environment analysis runner;
- diagnostics and auto-correction;
- documentation and tests;
- checkpoint and log directories.

Main files:

- `neural_network/src/encoder.py`
- `neural_network/src/model.py`
- `neural_network/src/policy.py`
- `neural_network/src/reward.py`
- `neural_network/src/self_play.py`
- `neural_network/src/benchmark.py`
- `neural_network/src/storage.py`
- `neural_network/src/trainer.py`
- `neural_network/src/orbit_wars_adapter.py`
- `neural_network/src/diagnostics.py`
- `neural_network/scripts/run_30min_analysis.py`
- `neural_network/scripts/auto_correct.py`

---

## 3. Documentation Basis

The implementation was built from the existing neural proposal document:

- source reference: `docs/analysis/neural_proposal.txt`

Useful ideas kept from that document:

- separate encoder / policy / value structure;
- action scoring instead of hard-coded plan selection;
- compact model suitable for CPU inference;
- self-play and checkpoint management;
- checkpoint export for evaluation-time use.

Ideas intentionally not copied blindly:

- dependence on the old bot grammar;
- assumptions about old heuristics being correct;
- overly complex architecture before proving the pipeline;
- direct coupling to the previous V7/V8 decision logic.

---

## 4. What Was Validated Successfully

### Technical plumbing

- Python imports work.
- Configuration loading works.
- Encoder produces a consistent vector.
- Model forward pass works.
- Policy returns a valid candidate.
- Reward returns a numeric value.
- Checkpoint save/load works.
- Mini self-play works.
- Minimal benchmark works.
- The real Kaggle `orbit_wars` environment is available locally.

### Real-environment integration

- The runner was updated to use the real `orbit_wars` environment, not only the synthetic game.
- The adapter now converts Kaggle observations to the package game dictionary.
- The runner can produce logs, checkpoints, and markdown summaries while playing real matches.

---

## 5. What Failed In Real Gameplay

The important failure mode is stable and explicit:

- the true `orbit_wars` run stays in terminal collapse;
- `avg_reward` remains at `-1.0`;
- `best_score` remains at `-1.0`;
- the run does not recover after auto-correction;
- the run continues to generate episodes, but the policy does not improve the outcome.

Observed symptoms:

- real gameplay is running;
- heartbeats are printed correctly;
- the pipeline is not stuck;
- but the policy is not competitive.

Interpretation:

- this is not a runner bug anymore;
- this is a decision / policy / reward problem;
- the current policy is unable to escape a losing regime in the real environment.

---

## 6. Auto-Correction System

An auto-correction path was added because the real run repeatedly collapsed.

Behavior:

- diagnostics classify the run state;
- when collapse is detected, the runner can switch to safer parameters;
- the system emits `status: autocorrect` events;
- a corrected config can also be written externally.

Current diagnosis observed during the 60-minute run:

- mode: `terminal_collapse`
- severity: `high`
- suggested fix:
  - `policy_mode = baseline_first`
  - `temperature = 1.4`
  - `min_ratio = 0.1`
  - `explore = True`

Important result:

- the auto-correction is working mechanically;
- it does not solve the collapse by itself.

So the auto-correction is a safety layer, not a strategic fix.

---

## 7. Real Run Summary

The last real run lasted about 60 minutes and produced the following pattern:

- episodes kept increasing;
- heartbeats were printed every minute;
- `avg_reward` stayed at `-1.0`;
- `best_score` stayed at `-1.0`;
- auto-correction triggered multiple times;
- no recovery was observed.

This is the key conclusion:

- the entire infrastructure is now functioning;
- the current policy is still not good enough for the real game.

---

## 8. Likely Root Causes

The current failure is probably caused by a combination of:

1. policy quality is too weak for the real environment;
2. the reward signal is too coarse and saturates at defeat;
3. the action conversion is still too simplistic for real Orbit Wars;
4. the fallback/baseline logic is not yet strong enough to serve as a rescue policy;
5. the model has not been trained on real strategic trajectories.

Most likely, the issue is not a single bug but a structural gap between the current prototype and the actual game complexity.

---

## 9. Current Diagnostics Available

The package now contains diagnostics that can be used immediately:

- `neural_network/src/diagnostics.py`
- `neural_network/scripts/auto_correct.py`

These tools can classify:

- `terminal_collapse`
- `invalid_action_collapse`
- `underactive_policy`
- `stable`

They also generate:

- a corrected config JSON;
- a markdown report;
- a JSONL event trail.

---

## 10. Important Files to Inspect Next

If you reopen the session, start with:

- `neural_network/logs/analysis_30min/report.md`
- `neural_network/logs/analysis_30min/autocorrect_report.md`
- `neural_network/logs/analysis_30min/metrics.jsonl`
- `neural_network/logs/analysis_30min/summary.csv`
- `neural_network/docs/neural_network_v2_pseudocode.md`

And the code paths:

- `neural_network/scripts/run_30min_analysis.py`
- `neural_network/src/policy.py`
- `neural_network/src/reward.py`
- `neural_network/src/orbit_wars_adapter.py`
- `neural_network/src/diagnostics.py`

---

## 11. Current Best Commands

### Real 60-minute run with auto-correction

```bash
python3 neural_network/scripts/run_30min_analysis.py --duration-minutes 60 --auto-correct
```

### More frequent auto-correction

```bash
python3 neural_network/scripts/run_30min_analysis.py --duration-minutes 60 --auto-correct --auto-correct-every-minutes 2
```

### External auto-correction step

```bash
python3 neural_network/scripts/auto_correct.py --metrics neural_network/logs/analysis_30min/metrics.jsonl --config neural_network/configs/default_config.json --output-config neural_network/configs/autocorrected_config.json --output-report neural_network/logs/analysis_30min/autocorrect_report.md
```

### Relaunch from corrected config

```bash
python3 neural_network/scripts/run_30min_analysis.py --duration-minutes 60 --config neural_network/configs/autocorrected_config.json --auto-correct
```

---

## 12. What This Means Scientifically

The engineering pipeline is now credible:

- real environment integrated;
- diagnostic loop present;
- auto-correction present;
- logging and replayability present.

But the network itself is not yet strategically adequate:

- the real game still defeats it consistently;
- no meaningful positive reward trend was observed;
- the current decision stack needs a stronger baseline or a much better training loop.

This is a useful result because it cleanly separates:

- infrastructure success;
- strategic failure.

---

## 13. Recommended Next Steps

1. Inspect `metrics.jsonl` to identify whether failures are due to:
   - invalid actions;
   - empty actions;
   - early terminal losses;
   - overconservative play.
2. Add explicit terminal reason tracking.
3. Add stronger baseline policies.
4. Replace the current policy with a more informed real-game policy.
5. Train on real Orbit Wars trajectories instead of synthetic behavior.
6. Add a richer reward decomposition so the signal is not only `-1.0` / win-loss.

---

## 14. Minimal Status Summary

- Package exists: yes
- Real environment hook: yes
- Auto-correction: yes
- Stable logs/checkpoints: yes
- Positive real-game performance: no
- Next major task: improve the policy and reward signal on the real game

