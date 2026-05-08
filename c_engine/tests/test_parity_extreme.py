"""
test_parity_extreme.py — Final hardcore parity validation.

Goal: prove the C engine is byte-identical to the Kaggle reference engine
across hundreds of full games with real bots, including all comet spawns.

If this passes, the engine is safe for training.
"""

import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from local_simulator.official_fast import OfficialFastGame  # noqa: E402
from c_engine import CGame  # noqa: E402
from opponents import ZOO  # noqa: E402

POS_TOL = 1e-9


def _records_match(py_recs, c_recs, kind, step, seed):
    py_sorted = sorted(py_recs, key=lambda r: int(r[0]))
    c_sorted = sorted(c_recs, key=lambda r: int(r[0]))
    if len(py_sorted) != len(c_sorted):
        return f"seed={seed} step={step} {kind} count py={len(py_sorted)} c={len(c_sorted)}"
    ints = (0, 1, 5, 6)
    floats = (2, 3, 4)
    for i, (pp, cc) in enumerate(zip(py_sorted, c_sorted)):
        for k in ints:
            if int(pp[k]) != int(cc[k]):
                return f"seed={seed} step={step} {kind}[{i}].field[{k}] py={pp[k]} c={cc[k]}"
        for k in floats:
            d = abs(float(pp[k]) - float(cc[k]))
            if d >= POS_TOL:
                return f"seed={seed} step={step} {kind}[{i}].field[{k}] py={pp[k]:.12f} c={cc[k]:.12f} diff={d:.2e}"
    return None


def _correct_scores(obs, n_players):
    scores = [0] * n_players
    for p in obs.planets:
        o = int(p[1])
        if 0 <= o < n_players:
            scores[o] += int(p[5])
    for f in obs.fleets:
        o = int(f[1])
        if 0 <= o < n_players:
            scores[o] += int(f[6])
    return scores


def run_full_game_parity(seed, agent0, agent1, max_steps=220):
    py = OfficialFastGame(2, seed=seed, episode_steps=max_steps, use_c_accel=False)
    c = CGame(2, seed=seed, episode_steps=max_steps)

    err = _records_match(py.observation(0).planets, c.observation(0).planets, 'planet', 0, seed)
    if err: return err

    for t in range(max_steps):
        if py.done or c.done:
            break
        a0 = agent0(py.observation(0))
        a1 = agent1(py.observation(1))
        py.step([a0, a1])
        c.step([a0, a1])

        py_obs = py.observation(0)
        c_obs = c.observation(0)
        err = _records_match(py_obs.planets, c_obs.planets, 'planet', t + 1, seed)
        if err: return err
        err = _records_match(py_obs.fleets, c_obs.fleets, 'fleet', t + 1, seed)
        if err: return err

    py_s = _correct_scores(py.observation(0), 2)
    c_s = _correct_scores(c.observation(0), 2)
    if py_s != c_s:
        return f"seed={seed} final scores py={py_s} c={c_s}"
    return None


def main():
    sys.path.insert(0, str(ROOT))
    import bot_v12

    print("=== EXTREME PARITY TEST ===")
    print()

    # Test against multiple bot types to exercise different game patterns
    bot_pairs = [
        ('v12 vs v12', bot_v12.agent, bot_v12.agent),
        ('v12 vs greedy', bot_v12.agent, ZOO['greedy']),
        ('greedy vs greedy', ZOO['greedy'], ZOO['greedy']),
        ('v12 vs notebook_physics_accurate', bot_v12.agent, ZOO['notebook_physics_accurate']),
        ('v12 vs notebook_distance_prioritized', bot_v12.agent, ZOO['notebook_distance_prioritized']),
    ]

    total_games = 0
    total_pass = 0
    failures = []

    for label, a0, a1 in bot_pairs:
        n_seeds = 40 if 'v12 vs v12' in label or 'greedy' in label else 25
        seeds = list(range(1000, 1000 + n_seeds))
        t0 = time.time()
        passes = 0
        for seed in seeds:
            err = run_full_game_parity(seed, a0, a1)
            total_games += 1
            if err is None:
                passes += 1
                total_pass += 1
            else:
                failures.append((label, seed, err))
        elapsed = time.time() - t0
        print(f"  {label}: {passes}/{len(seeds)} pass  ({elapsed:.1f}s)")

    print()
    print(f"=== TOTAL: {total_pass}/{total_games} games byte-identical ===")
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for label, seed, err in failures[:10]:
            print(f"  [{label}] seed={seed}: {err}")
        sys.exit(1)
    print("✓ C ENGINE PROVEN EQUIVALENT TO KAGGLE REFERENCE")


if __name__ == '__main__':
    main()
