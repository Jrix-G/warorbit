# Session Summary - 2026-04-29

## Objective
Build an autonomous overnight training pipeline for V8 that can run for about 8 hours without manual supervision.

## Core findings
- The analysis report indicates the winning pattern is not raw opening speed.
- The important signal is conversion into a durable production backbone.
- The recurring failure mode is plateauing below the conversion threshold.
- The main training risk was unstable self-labeling and misleading local metrics.

## Decisions made
- Replace the unstable online self-labeling loop with a fixed offline teacher pipeline.
- Use the analysis report to shape the teacher score around conversion and production lock.
- Add an autopilot wrapper that:
  - runs short probes first,
  - checks benchmark score and minimum opponent score,
  - only launches the long run if the probe passes,
  - aborts cleanly if all probes fail.
- Add checkpoint/resume support so overnight execution can recover from interruption.
- Prevent validation passes from mutating model weights.

## Files changed
- `train_v8_offline.py`
- `run_v8_autopilot.py`
- `run_v8_autopilot.ps1`
- `run_v8_offline_probe.ps1`
- `v8_core.py`

## Important behavior
- `train_v8_offline.py` now mines a frozen dataset before training.
- It uses a structural teacher score derived from board conversion and production backbone.
- It can decay learning rates automatically if validation stagnates.
- `run_v8_autopilot.py` accepts only bounded probe configurations.
- If no probe reaches the acceptance gate, the overnight run is aborted instead of wasting time.

## Current launch command
```powershell
.\run_v8_autopilot.ps1
```

## Short probe command
```powershell
.\run_v8_offline_probe.ps1
```

## Caveat
This is a robust control pipeline, not a guarantee of a 70% win rate. It is designed to avoid burning a night on a visibly bad training signal.
