"""Unit tests for v15_eval — the Composite Static Evaluator (ESC).

Verifies the evaluator is monotone in the player's favour: gaining ships,
production or planets must never lower the score, and a dominant position
must score higher than a losing one.

Run:
    python tests/test_v15_eval.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import v15_eval
import v15_fast_sim as fsim

ID, OWNER, X, Y, R, SHIPS, PROD = range(7)


def _mk_state(planet_rows, fleet_rows=None, n_players=2):
    """Build a minimal FastState from [id,owner,x,y,r,ships,prod] rows."""
    planets = np.array(planet_rows, dtype=np.float64)
    fleets = (np.array(fleet_rows, dtype=np.float64)
              if fleet_rows else np.zeros((0, 7), dtype=np.float64))
    return fsim.FastState(
        planets=planets,
        p_init=planets[:, X:Y + 1].copy(),
        p_comet=np.zeros(len(planets), dtype=np.bool_),
        fleets=fleets,
        comets=[],
        step=100,
        angular_velocity=0.0,
        next_fleet_id=0,
        episode_steps=500,
        ship_speed=6.0,
        n_players=n_players,
    )


def test_symmetric_is_half():
    """A perfectly symmetric 2p board scores ~0.5 for both players."""
    st = _mk_state([
        [0, 0, 20, 20, 2, 50, 3],
        [1, 1, 80, 80, 2, 50, 3],
    ])
    s0 = v15_eval.evaluate(st, 0)
    s1 = v15_eval.evaluate(st, 1)
    assert abs(s0 - 0.5) < 1e-6, f"symmetric eval p0={s0}"
    assert abs(s1 - 0.5) < 1e-6, f"symmetric eval p1={s1}"
    print(f"  symmetric: p0={s0:.4f} p1={s1:.4f}  OK")


def test_more_ships_scores_higher():
    """Adding ships to our planet must raise our score."""
    base = _mk_state([
        [0, 0, 20, 20, 2, 50, 3],
        [1, 1, 80, 80, 2, 50, 3],
    ])
    strong = _mk_state([
        [0, 0, 20, 20, 2, 200, 3],
        [1, 1, 80, 80, 2, 50, 3],
    ])
    lo = v15_eval.evaluate(base, 0)
    hi = v15_eval.evaluate(strong, 0)
    assert hi > lo, f"more ships did not raise score: {lo} -> {hi}"
    print(f"  more ships: {lo:.4f} -> {hi:.4f}  OK")


def test_more_planets_scores_higher():
    """Owning an extra planet must raise our score."""
    base = _mk_state([
        [0, 0, 20, 20, 2, 50, 3],
        [1, 1, 80, 80, 2, 50, 3],
        [2, -1, 50, 50, 2, 10, 2],
    ])
    captured = _mk_state([
        [0, 0, 20, 20, 2, 50, 3],
        [1, 1, 80, 80, 2, 50, 3],
        [2, 0, 50, 50, 2, 10, 2],
    ])
    lo = v15_eval.evaluate(base, 0)
    hi = v15_eval.evaluate(captured, 0)
    assert hi > lo, f"capturing a planet did not raise score: {lo} -> {hi}"
    print(f"  more planets: {lo:.4f} -> {hi:.4f}  OK")


def test_more_production_scores_higher():
    """Higher production must raise our score (snowball signal)."""
    base = _mk_state([
        [0, 0, 20, 20, 2, 50, 3],
        [1, 1, 80, 80, 2, 50, 3],
    ])
    rich = _mk_state([
        [0, 0, 20, 20, 2, 50, 9],
        [1, 1, 80, 80, 2, 50, 3],
    ])
    lo = v15_eval.evaluate(base, 0)
    hi = v15_eval.evaluate(rich, 0)
    assert hi > lo, f"more production did not raise score: {lo} -> {hi}"
    print(f"  more production: {lo:.4f} -> {hi:.4f}  OK")


def test_dominant_beats_losing():
    """A crushing lead scores far above a losing position."""
    winning = _mk_state([
        [0, 0, 20, 20, 2, 500, 12],
        [1, 1, 80, 80, 2, 10, 1],
    ])
    losing = _mk_state([
        [0, 0, 20, 20, 2, 10, 1],
        [1, 1, 80, 80, 2, 500, 12],
    ])
    win = v15_eval.evaluate(winning, 0)
    lose = v15_eval.evaluate(losing, 0)
    assert win > 0.8, f"dominant position scored only {win}"
    assert lose < 0.2, f"losing position scored {lose}"
    print(f"  dominant={win:.4f}  losing={lose:.4f}  OK")


def test_eliminated_scores_near_zero():
    """An eliminated player (no ships, no planets) scores near 0."""
    st = _mk_state([
        [0, 0, 20, 20, 2, 100, 3],
        [1, 1, 80, 80, 2, 100, 3],
    ])
    # player 2 owns nothing
    s2 = v15_eval.evaluate(st, 1)
    # build a 4p state where player 3 has nothing
    st4 = _mk_state([
        [0, 0, 10, 10, 2, 100, 3],
        [1, 1, 90, 10, 2, 100, 3],
        [2, 2, 10, 90, 2, 100, 3],
        [3, 3, 90, 90, 2, 0, 0],
    ], n_players=4)
    # planet 3 owned but empty -> still some planet_share; drop it instead
    st4b = _mk_state([
        [0, 0, 10, 10, 2, 100, 3],
        [1, 1, 90, 10, 2, 100, 3],
        [2, 2, 10, 90, 2, 100, 3],
    ], n_players=4)
    elim = v15_eval.evaluate(st4b, 3)
    assert elim < 0.15, f"eliminated player scored {elim}"
    print(f"  eliminated (4p): {elim:.4f}  OK")


def test_4p_weights_favour_production():
    """In 4p, a production lead matters more than the same lead in 2p."""
    f = np.array([0.25, 0.25, 0.25, 0.5, 0.5])  # neutral-ish features
    f_prod = f.copy()
    f_prod[1] += 0.3  # +0.3 prod_share
    gain_2p = float((f_prod - f) @ v15_eval._W_2P)
    gain_4p = float((f_prod - f) @ v15_eval._W_4P)
    assert gain_4p > gain_2p, f"4p prod weight not higher: {gain_4p} vs {gain_2p}"
    print(f"  prod_share gain: 2p={gain_2p:.4f} 4p={gain_4p:.4f}  OK")


if __name__ == "__main__":
    tests = [
        test_symmetric_is_half,
        test_more_ships_scores_higher,
        test_more_planets_scores_higher,
        test_more_production_scores_higher,
        test_dominant_beats_losing,
        test_eliminated_scores_near_zero,
        test_4p_weights_favour_production,
    ]
    print(f"v15_eval — running {len(tests)} tests")
    for t in tests:
        t()
    print(f"all {len(tests)} tests passed")
