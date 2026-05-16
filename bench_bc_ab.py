"""A/B benchmark — RCC with BC continuation vs RCC with passive continuation.

Head-to-head: one slot runs RCC(bc_cont=True), the rest run RCC(bc_cont=False).
Same seeds for 2p and 4p. A win rate above 0.50 means the behavioral-cloning
continuation improves combo evaluation; below 0.50 means passive quiescence
is better and BC should be dropped.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE python -u bench_bc_ab.py --games 80 --workers 8
"""

from __future__ import annotations

import argparse
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import v14_core
import v15_search
from local_simulator.official_fast import OfficialFastGame


def _wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (p, c - m, c + m)


def _bc(obs, config):
    obs = v14_core.obs_as_dict(obs)
    m = v15_search.search(obs, config, time_budget=0.7, bc_cont=True)
    return m if isinstance(m, list) else []


def _passive(obs, config):
    obs = v14_core.obs_as_dict(obs)
    m = v15_search.search(obs, config, time_budget=0.7, bc_cont=False)
    return m if isinstance(m, list) else []


def _play(task):
    n_players, seed = task
    random.seed(seed)
    np.random.seed(seed)
    our_idx = seed % n_players
    agents = [_passive] * n_players
    agents[our_idx] = _bc
    game = OfficialFastGame(n_players, seed=seed, episode_steps=200,
                            use_c_accel=False)
    while not game.done:
        actions = [agents[p](game.observation(p), game.configuration)
                   for p in range(n_players)]
        game.step(actions)
    sc = game.scores()
    best_other = max(s for i, s in enumerate(sc) if i != our_idx)
    win = 1 if (sc[our_idx] > best_other and sc[our_idx] > 0) else 0
    return n_players, win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=80)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed-offset", type=int, default=400000)
    args = ap.parse_args()

    tasks = []
    for n_players in (2, 4):
        for i in range(args.games):
            tasks.append((n_players, args.seed_offset + i))

    t0 = time.time()
    if args.workers <= 1:
        res = [_play(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            res = list(pool.map(_play, tasks))

    print("RCC BC-continuation vs RCC passive-continuation")
    for mode, n_players in (("2p", 2), ("4p", 4)):
        sub = [r for r in res if r[0] == n_players]
        w = sum(r[1] for r in sub)
        n = len(sub)
        p, lo, hi = _wilson(w, n)
        verdict = ("BC helps" if lo > 0.5 else
                   "BC hurts" if hi < 0.5 else "inconclusive")
        print(f"  {mode}: BC W={w}/{n} WR={p:.3f} "
              f"95%CI=[{lo:.3f},{hi:.3f}]  -> {verdict}")
    print(f"elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
