# Orbit Wars Local Optimization

This document describes the local-only performance work around Kaggle's
official `orbit_wars` environment.

## Non-Negotiable Rule

The game rules must not change.  The official Python engine remains the
reference implementation:

```text
github/kaggle-environments/kaggle_environments/envs/orbit_wars/orbit_wars.py
```

The optimized code is allowed only when it produces the same game states as
Kaggle's official `make("orbit_wars").run(...)` path.

## Layers

### 1. Official Fast Runner

File:

```text
local_simulator/official_fast.py
```

This runner still calls the official `orbit_wars.interpreter()`, but it avoids
the generic Kaggle wrapper overhead during local training:

- no `jsonschema` validation on every action;
- no recursive `structify`/`deepcopy` of the full state on every wrapper step;
- no stdout/stderr capture around every agent call.

It also preserves a Kaggle wrapper quirk exactly: only player 0's
`observation.step` is updated by `core.py` after each interpreter call.

### 2. Optional C Accelerator

Files:

```text
local_simulator/c_accel/orbit_wars_c.c
local_simulator/c_accel/orbit_wars_accel.py
local_simulator/c_accel/build_orbit_wars_c.py
```

The C extension accelerates only:

```python
generate_comet_paths(...)
```

This was selected because profiling showed comet-path generation dominates the
pure engine cost in no-op/local simulation runs.  The C implementation still
uses the caller's Python RNG object for `uniform(...)`, preserving the same RNG
sequence and keeping the Python implementation as the behavior reference.

Build command:

```bash
.venv/bin/python local_simulator/c_accel/build_orbit_wars_c.py
```

Disable explicitly for conservative runs:

```python
from local_simulator.official_fast import OfficialFastGame

game = OfficialFastGame(2, seed=0, use_c_accel=False)
```

### 3. Training Mode

File:

```text
war_orbit/training/self_play.py
```

New engine mode:

```python
config.game_engine = "kaggle_fast"
```

Existing modes remain available:

```python
config.game_engine = "kaggle"   # official Kaggle wrapper
config.game_engine = "simgame"  # existing project simulator
```

## Equivalence Tests

Run:

```bash
.venv/bin/python -m pytest \
  local_simulator/test_official_fast.py \
  local_simulator/test_orbit_wars_c_accel.py \
  -q
```

The tests compare complete states, not just winners:

- planets;
- fleets;
- comet paths and comet IDs;
- rewards;
- statuses;
- step behavior.

If these tests fail, do not use `kaggle_fast` or the C accelerator.

## Benchmarks

Run:

```bash
.venv/bin/python local_simulator/benchmark_official_fast.py \
  --games 20 \
  --episode-steps 500 \
  --agent-set noop
```

Measured on this machine after the C accelerator was built:

```text
No-op agents, 20 games:
Kaggle make/run       2.4278 s/game
Official fast Python  0.4980 s/game
Official fast C       0.1931 s/game
Speedup vs Kaggle     12.57x
Saved per game        2.2347 s
```

With deterministic agents that create real fleets:

```text
Deterministic expand agents, 10 games:
Kaggle make/run       2.4267 s/game
Official fast Python  1.1826 s/game
Official fast C       0.7630 s/game
Speedup vs Kaggle     3.18x
Saved per game        1.6637 s
```

With the current `submission.agent`, the agent itself dominates more of the
runtime, so the relative engine speedup is smaller.  The local runner still
removes wrapper overhead and remains useful for high-volume training.

## Why Not Rewrite Everything?

A full C/Rust rewrite would create a large correctness surface:

- floating-point branch boundaries in collisions;
- RNG sequence preservation;
- exact comet path validation;
- mutable shared observations and Kaggle wrapper quirks;
- action sanitization and invalid action behavior.

The current implementation targets the measured bottleneck while keeping the
official Python interpreter as the system boundary.  That is the highest-gain
change with a manageable equivalence proof.

