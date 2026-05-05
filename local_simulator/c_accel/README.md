# Orbit Wars C Accelerator

This directory contains an optional CPython C extension for local training.
It does not replace the official Python engine.  It only accelerates one
hotspot inside the official engine:

```text
orbit_wars.generate_comet_paths(...)
```

That function is expensive because it samples 5,000 points per comet attempt,
uses trigonometry heavily, and validates candidate paths against all planets.

## Safety Rule

The Python implementation remains the source of truth.  The C module may be
used only after exact equivalence tests pass against:

- the original Python `generate_comet_paths`;
- the official Kaggle `make("orbit_wars").run(...)` wrapper;
- the fast local runner.

If any exact comparison fails, do not use the accelerator.

## Build

From the repository root:

```bash
python3 local_simulator/c_accel/build_orbit_wars_c.py
```

or, if using the project virtual environment:

```bash
.venv/bin/python local_simulator/c_accel/build_orbit_wars_c.py
```

The build creates a local `.so` file next to `orbit_wars_c.c`.

## Runtime

`local_simulator.official_fast.OfficialFastGame` tries to enable the C
accelerator by default. If the `.so` file is missing, it silently keeps the
official Python function.

For conservative runs, disable it explicitly:

```python
game = OfficialFastGame(2, seed=0, use_c_accel=False)
```

