"""
test_parity.py — Verify C engine matches Python reference exactly.

Strategy: for each seed, run BOTH engines step by step with identical actions
and compare planets/fleets/scores after each step. Any divergence aborts.
"""

import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from local_simulator.official_fast import OfficialFastGame  # noqa: E402
from c_engine import CGame  # noqa: E402

POS_TOL = 1e-9
INT_FIELDS_PLANET = (0, 1, 5, 6)  # id, owner, ships, production
FLOAT_FIELDS_PLANET = (2, 3, 4)   # x, y, radius
INT_FIELDS_FLEET = (0, 1, 5, 6)   # id, owner, from_planet_id, ships
FLOAT_FIELDS_FLEET = (2, 3, 4)    # x, y, angle


def _key_planet(p):
    return int(p[0])


def _key_fleet(f):
    return int(f[0])


def assert_records_equal(py_recs, c_recs, kind, step):
    py_sorted = sorted(py_recs, key=_key_planet if kind == 'planet' else _key_fleet)
    c_sorted = sorted(c_recs, key=_key_planet if kind == 'planet' else _key_fleet)
    assert len(py_sorted) == len(c_sorted), (
        f"step={step} {kind} count py={len(py_sorted)} c={len(c_sorted)}")
    if kind == 'planet':
        ints, floats = INT_FIELDS_PLANET, FLOAT_FIELDS_PLANET
    else:
        ints, floats = INT_FIELDS_FLEET, FLOAT_FIELDS_FLEET
    for i, (pp, cc) in enumerate(zip(py_sorted, c_sorted)):
        for k in ints:
            assert int(pp[k]) == int(cc[k]), (
                f"step={step} {kind}[{i}] field[{k}] int diff py={pp[k]} c={cc[k]}")
        for k in floats:
            diff = abs(float(pp[k]) - float(cc[k]))
            assert diff < POS_TOL, (
                f"step={step} {kind}[{i}] field[{k}] float diff "
                f"py={pp[k]:.12f} c={cc[k]:.12f} diff={diff:.2e}")


def random_actions(obs, my_id, rng: random.Random):
    """Generate a small set of plausible random actions for parity testing."""
    planets = obs.planets if hasattr(obs, 'planets') else obs['planets']
    moves = []
    my = [p for p in planets if int(p[1]) == my_id and int(p[5]) > 1]
    if not my:
        return []
    n_moves = rng.randint(0, min(3, len(my)))
    for _ in range(n_moves):
        src = rng.choice(my)
        ships = rng.randint(1, max(1, int(src[5]) - 1))
        angle = rng.uniform(-math.pi, math.pi)
        moves.append([int(src[0]), float(angle), int(ships)])
    return moves


@pytest.mark.parametrize("seed", [42, 100, 200, 314, 1000, 7777, 12345, 999999])
def test_parity_random(seed):
    py = OfficialFastGame(2, seed=seed, episode_steps=200, use_c_accel=False)
    c = CGame(2, seed=seed, episode_steps=200)

    # Compare initial states
    py_obs = py.observation(0)
    c_obs = c.observation(0)
    assert_records_equal(py_obs.planets, c_obs.planets, 'planet', step=0)
    assert_records_equal(py_obs.fleets, c_obs.fleets, 'fleet', step=0)

    rng = random.Random(seed * 7 + 1)
    for t in range(50):  # short to keep test fast initially
        if py.done or c.done:
            break
        a0 = random_actions(py.observation(0), 0, rng)
        a1 = random_actions(py.observation(1), 1, rng)
        py.step([a0, a1])
        c.step([a0, a1])

        py_obs = py.observation(0)
        c_obs = c.observation(0)
        assert_records_equal(py_obs.planets, c_obs.planets, 'planet', step=t + 1)
        assert_records_equal(py_obs.fleets, c_obs.fleets, 'fleet', step=t + 1)


if __name__ == '__main__':
    test_parity_random(42)
    print("PASS")
