"""V15.1 gate — new search vs the frozen V15 baseline, head-to-head.

player 0 = v15_search (current = V15.1 candidate)
opponents = v15_search_baseline (frozen V15)
Seats swapped per game. Gate: WR Wilson 95% LB above the seat baseline (1/n).

Run:
    KMP_DUPLICATE_LIB_OK=TRUE python -u v15_1_gate.py --games 16 --workers 8
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
import v15_search_baseline
from local_simulator.official_fast import OfficialFastGame


def _wilson_lb(w: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (c - m) / d


def _new(obs, config):
    obs = v14_core.obs_as_dict(obs)
    m = v15_search.search(obs, config, time_budget=0.7)
    return m if isinstance(m, list) else []


def _base(obs, config):
    obs = v14_core.obs_as_dict(obs)
    m = v15_search_baseline.search(obs, config, time_budget=0.7)
    return m if isinstance(m, list) else []


def _play(task):
    n_players, seed = task
    random.seed(seed)
    np.random.seed(seed)
    our_idx = seed % n_players
    agents = [_base] * n_players
    agents[our_idx] = _new
    game = OfficialFastGame(n_players, seed=seed, episode_steps=200, use_c_accel=False)
    t0 = time.time()
    while not game.done:
        actions = [agents[p](game.observation(p), game.configuration)
                   for p in range(n_players)]
        game.step(actions)
    sc = game.scores()
    best_other = max(s for i, s in enumerate(sc) if i != our_idx)
    win = 1 if (sc[our_idx] > best_other and sc[our_idx] > 0) else 0
    return win, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed-offset", type=int, default=130000)
    ap.add_argument("--modes", nargs="*", default=["2p", "4p"])
    args = ap.parse_args()

    gw = gn = 0
    for mode in args.modes:
        n_players = 2 if mode == "2p" else 4
        tasks = [(n_players, args.seed_offset + i) for i in range(args.games)]
        if args.workers <= 1:
            res = [_play(t) for t in tasks]
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                res = list(pool.map(_play, tasks))
        w = sum(r[0] for r in res)
        n = len(res)
        secs = sum(r[1] for r in res)
        lb = _wilson_lb(w, n)
        print(f"- {mode}: V15.1 W={w}/{n} WR={w/n:.3f} Wilson95%LB={lb:.3f} "
              f"({secs/n:.0f}s/game)")
        gw += w
        gn += n
    lb = _wilson_lb(gw, gn)
    print(f"AGG V15.1 vs V15: W={gw}/{gn} WR={gw/gn:.3f} Wilson95%LB={lb:.3f}")


if __name__ == "__main__":
    main()
