# Agent Prompt for Neural Network Improvement

You are given the `neural_network/` package of an Orbit Wars training project.

## Goal

Improve the current neural network training stack so it learns more effectively and produces a stronger policy in the real environment.

## Current observed behavior

- The training pipeline is functional.
- The model improves early, then plateaus.
- Best observed checkpoint in the recent run reached around `best_score = 29.1595`.
- The run does not collapse, but it does not continue improving after the early phase.
- 4-player curriculum appears active, but the gain is limited.

## My opinion on what likely matters most

1. Increase model capacity if compute allows.
2. Increase training budget significantly.
3. Keep the best-checkpoint promotion logic, but make sure evaluation is stable enough to avoid noise-driven promotions.
4. Improve the curriculum so the model does not spend too long in a regime that is too easy or too noisy.
5. Review reward shaping and action selection if the policy still plateaus after more capacity and training.

## Files worth inspecting first

- `src/trainer.py`
- `src/notebook_4p_training.py`
- `src/policy.py`
- `src/reward.py`
- `src/diagnostics.py`
- `scripts/run_notebook_4p_training.py`
- `scripts/run_30min_analysis.py`
- `logs/analysis_30min/report.md`
- `logs/analysis_30min/metrics.jsonl`
- `logs/training.jsonl`

## What to deliver

- A concrete improved version of the training pipeline.
- A short explanation of the changes.
- If possible, include changes that make the improvement measurable rather than cosmetic.

## Constraints

- Keep changes inside the `neural_network/` package.
- Do not touch unrelated parts of the repository.
- Prefer simple, testable changes over large rewrites.
