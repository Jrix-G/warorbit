"""
test_parity_massive.py — Massive parity validation between Python and C engines.

Categories:
  1. Random actions, short games (50 steps), many seeds
  2. Random actions, long games (220 steps with comet spawns)
  3. Real bots (greedy, V12) playing full games
  4. 4-player games
  5. High-volume actions (full-send stress test)
  6. Score equality at game end
"""

import math
import random
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from local_simulator.official_fast import OfficialFastGame  # noqa: E402
from c_engine import CGame  # noqa: E402

POS_TOL = 1e-9


def _records_match(py_recs, c_recs, kind, step, seed):
    py_sorted = sorted(py_recs, key=lambda r: int(r[0]))
    c_sorted = sorted(c_recs, key=lambda r: int(r[0]))
    if len(py_sorted) != len(c_sorted):
        return f"seed={seed} step={step} {kind} count py={len(py_sorted)} c={len(c_sorted)}"
    if kind == 'planet':
        ints, floats = (0, 1, 5, 6), (2, 3, 4)
    else:
        ints, floats = (0, 1, 5, 6), (2, 3, 4)
    for i, (pp, cc) in enumerate(zip(py_sorted, c_sorted)):
        for k in ints:
            if int(pp[k]) != int(cc[k]):
                return f"seed={seed} step={step} {kind}[{i}] field[{k}] int py={pp[k]} c={cc[k]}"
        for k in floats:
            d = abs(float(pp[k]) - float(cc[k]))
            if d >= POS_TOL:
                return f"seed={seed} step={step} {kind}[{i}] field[{k}] float py={pp[k]:.12f} c={cc[k]:.12f} diff={d:.2e}"
    return None


def _random_actions(obs, my_id, rng, max_moves=4):
    planets = obs.planets if hasattr(obs, 'planets') else obs['planets']
    moves = []
    my = [p for p in planets if int(p[1]) == my_id and int(p[5]) > 1]
    if not my:
        return []
    n = rng.randint(0, min(max_moves, len(my)))
    for _ in range(n):
        src = rng.choice(my)
        ships = rng.randint(1, max(1, int(src[5]) - 1))
        angle = rng.uniform(-math.pi, math.pi)
        moves.append([int(src[0]), float(angle), int(ships)])
    return moves


def _aggressive_actions(obs, my_id, rng):
    """Stress test: send many fleets, full-send mostly."""
    planets = obs.planets if hasattr(obs, 'planets') else obs['planets']
    moves = []
    my = [p for p in planets if int(p[1]) == my_id and int(p[5]) > 0]
    if not my:
        return []
    others = [p for p in planets if int(p[1]) != my_id]
    for src in my:
        if not others:
            break
        if rng.random() < 0.7:
            tgt = rng.choice(others)
            ships = max(1, int(int(src[5]) * (0.5 + 0.5 * rng.random())))
            angle = math.atan2(tgt[3] - src[3], tgt[2] - src[2])
            moves.append([int(src[0]), float(angle), ships])
    return moves


def _run_parity(seed, n_players, n_steps, action_fn, rng):
    """Run both engines step by step. Return None on success, error string on first divergence."""
    py = OfficialFastGame(n_players, seed=seed, episode_steps=max(n_steps + 5, 200), use_c_accel=False)
    c = CGame(n_players, seed=seed, episode_steps=max(n_steps + 5, 200))

    err = _records_match(py.observation(0).planets, c.observation(0).planets, 'planet', 0, seed)
    if err: return err

    for t in range(n_steps):
        if py.done or c.done:
            break
        actions = []
        for pid in range(n_players):
            actions.append(action_fn(py.observation(pid), pid, rng))
        py.step(actions)
        c.step(actions)
        err = _records_match(py.observation(0).planets, c.observation(0).planets, 'planet', t + 1, seed)
        if err: return err
        err = _records_match(py.observation(0).fleets, c.observation(0).fleets, 'fleet', t + 1, seed)
        if err: return err
    return None


@pytest.mark.parametrize("seed", list(range(50)))
def test_parity_50_seeds_short(seed):
    """50 distinct seeds × 50 steps × random actions."""
    rng = random.Random(seed * 31 + 7)
    err = _run_parity(seed, 2, 50, _random_actions, rng)
    assert err is None, err


@pytest.mark.parametrize("seed", [11, 22, 33, 44, 55, 66, 77, 88, 99, 111])
def test_parity_long_games_with_comets(seed):
    """10 seeds × 200 steps. Includes comet spawn at step 50 and 150."""
    rng = random.Random(seed * 17 + 3)
    err = _run_parity(seed, 2, 200, _random_actions, rng)
    assert err is None, err


@pytest.mark.parametrize("seed", [10, 20, 30, 40, 50])
def test_parity_aggressive(seed):
    """5 seeds × 100 steps × heavy fleet launches (stress combat)."""
    rng = random.Random(seed * 13)
    err = _run_parity(seed, 2, 100, _aggressive_actions, rng)
    assert err is None, err


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
def test_parity_4players(seed):
    """8 seeds × 60 steps × 4 players."""
    rng = random.Random(seed * 23)
    err = _run_parity(seed, 4, 60, _random_actions, rng)
    assert err is None, err


def _correct_scores(obs, n_players):
    """Score matching the Kaggle interpreter (orbit_wars_official.py lines 680-685).

    OfficialFastGame.scores() has a bug using fleet[4] (angle) instead of
    fleet[6] (ships). We compute the true Kaggle score here.
    """
    scores = [0] * n_players
    for p in obs.planets:
        o = int(p[1])
        if 0 <= o < n_players:
            scores[o] += int(p[5])
    for f in obs.fleets:
        o = int(f[1])
        if 0 <= o < n_players:
            scores[o] += int(f[6])  # ships, NOT fleet[4] (angle)
    return scores


def test_parity_real_bots():
    """V12 vs V12 — full game including all engine features."""
    sys.path.insert(0, str(ROOT))
    import bot_v12

    for seed in [101, 202, 303, 404, 505]:
        py = OfficialFastGame(2, seed=seed, episode_steps=200, use_c_accel=False)
        c = CGame(2, seed=seed, episode_steps=200)

        for t in range(200):
            if py.done or c.done:
                break
            obs0_py = py.observation(0)
            obs1_py = py.observation(1)
            a0 = bot_v12.agent(obs0_py)
            a1 = bot_v12.agent(obs1_py)
            py.step([a0, a1])
            c.step([a0, a1])
            err = _records_match(py.observation(0).planets, c.observation(0).planets, 'planet', t + 1, seed)
            assert err is None, err
            err = _records_match(py.observation(0).fleets, c.observation(0).fleets, 'fleet', t + 1, seed)
            assert err is None, err

        # Final scores via Kaggle's correct formula
        py_scores = _correct_scores(py.observation(0), 2)
        c_scores = _correct_scores(c.observation(0), 2)
        assert py_scores == c_scores, f"seed={seed} scores py={py_scores} c={c_scores}"


def test_parity_full_termination():
    """Run several games to natural termination, verify scores+winner identical."""
    for seed in [50, 60, 70, 80, 90]:
        rng = random.Random(seed * 7)
        py = OfficialFastGame(2, seed=seed, episode_steps=200, use_c_accel=False)
        c = CGame(2, seed=seed, episode_steps=200)

        steps = 0
        while not py.done and not c.done:
            actions = [
                _aggressive_actions(py.observation(0), 0, rng),
                _aggressive_actions(py.observation(1), 1, rng),
            ]
            py.step(actions)
            c.step(actions)
            steps += 1
            if steps > 250:
                break

        # done flags should match
        assert py.done == c.done, f"seed={seed} done py={py.done} c={c.done} after {steps} steps"
        # scores match (using Kaggle's correct formula, bypassing OfficialFastGame helper bug)
        py_s = _correct_scores(py.observation(0), 2)
        c_s = _correct_scores(c.observation(0), 2)
        assert py_s == c_s, f"seed={seed} scores py={py_s} c={c_s}"


if __name__ == '__main__':
    print("Running massive parity tests…")
    n_pass = 0
    n_fail = 0
    failures = []

    # Short, many seeds
    for seed in range(50):
        rng = random.Random(seed * 31 + 7)
        err = _run_parity(seed, 2, 50, _random_actions, rng)
        if err:
            n_fail += 1
            failures.append(err)
        else:
            n_pass += 1
    print(f"50 seeds short: {n_pass} pass / {n_fail} fail")

    # Long with comets
    for seed in [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]:
        rng = random.Random(seed * 17 + 3)
        err = _run_parity(seed, 2, 200, _random_actions, rng)
        if err:
            failures.append(err)
            print(f"  long seed={seed} FAIL: {err}")

    if failures:
        print("\nFAILURES:")
        for f in failures[:5]:
            print(" ", f)
    else:
        print("ALL PASS")
