"""V15.3 RCC benchmark — head-to-head vs bot_v7 at the deployed budget.

One RCC agent vs (n-1) V7 agents, same seeds for 2p and 4p. RCC runs at
time_budget=0.7s — the real Kaggle setting.

Success criteria : 2p WR > 0.65, 4p WR > 0.35  (vs V7)
Regression guard : 2p WR < 0.50

Run:
    KMP_DUPLICATE_LIB_OK=TRUE python -u bench_v153.py --games 100 --workers 8
"""

from __future__ import annotations

import argparse
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import bot_v7
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


def _v7(obs, config):
    obs = v14_core.obs_as_dict(obs)
    try:
        m = bot_v7.agent(obs, config)
    except TypeError:
        m = bot_v7.agent(obs)
    return m if isinstance(m, list) else []


def _rcc(obs, config):
    obs = v14_core.obs_as_dict(obs)
    m = v15_search.search(obs, config, time_budget=0.7)
    return m if isinstance(m, list) else []


def _play(task):
    n_players, seed = task
    random.seed(seed)
    np.random.seed(seed)
    our_idx = seed % n_players
    agents = [_v7] * n_players
    agents[our_idx] = _rcc
    game = OfficialFastGame(n_players, seed=seed, episode_steps=200,
                            use_c_accel=False)
    t0 = time.time()
    while not game.done:
        actions = [agents[p](game.observation(p), game.configuration)
                   for p in range(n_players)]
        game.step(actions)
    sc = game.scores()
    best_other = max(s for i, s in enumerate(sc) if i != our_idx)
    win = 1 if (sc[our_idx] > best_other and sc[our_idx] > 0) else 0
    return n_players, win, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100, help="games per mode")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed-offset", type=int, default=300000)
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

    print("V15.3 RCC vs V7")
    for mode, n_players in (("2p", 2), ("4p", 4)):
        sub = [r for r in res if r[0] == n_players]
        w = sum(r[1] for r in sub)
        n = len(sub)
        p, lo, hi = _wilson(w, n)
        print(f"  {mode}: W={w}/{n} WR={p:.3f} 95%CI=[{lo:.3f},{hi:.3f}]")
    print(f"elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
