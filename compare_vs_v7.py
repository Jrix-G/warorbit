"""Definitive check: does the search actually beat V7 at the DEPLOYED budget?

Benchmarks both search variants head-to-head vs bot_v7, same seeds:
  - v15            = v15_search_baseline (flat MC, the deployed V15)
  - v15.1-A        = v15_search (sequential halving)
Both run at time_budget=0.7s — the real Kaggle setting.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE python -u compare_vs_v7.py --games 20 --workers 8
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
import v15_search_baseline
from local_simulator.official_fast import OfficialFastGame


def _wilson_lb(w, n, z=1.96):
    if n == 0:
        return 0.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (c - m) / d


def _v7(obs, config):
    obs = v14_core.obs_as_dict(obs)
    try:
        m = bot_v7.agent(obs, config)
    except TypeError:
        m = bot_v7.agent(obs)
    return m if isinstance(m, list) else []


def _mk(mod):
    def fn(obs, config):
        obs = v14_core.obs_as_dict(obs)
        m = mod.search(obs, config, time_budget=0.7)
        return m if isinstance(m, list) else []
    return fn


def _play(task):
    bot_name, n_players, seed = task
    random.seed(seed)
    np.random.seed(seed)
    our = _mk(v15_search_baseline) if bot_name == "v15" else _mk(v15_search)
    our_idx = seed % n_players
    agents = [_v7] * n_players
    agents[our_idx] = our
    game = OfficialFastGame(n_players, seed=seed, episode_steps=200, use_c_accel=False)
    t0 = time.time()
    while not game.done:
        actions = [agents[p](game.observation(p), game.configuration)
                   for p in range(n_players)]
        game.step(actions)
    sc = game.scores()
    best_other = max(s for i, s in enumerate(sc) if i != our_idx)
    win = 1 if (sc[our_idx] > best_other and sc[our_idx] > 0) else 0
    return bot_name, n_players, win, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed-offset", type=int, default=140000)
    args = ap.parse_args()

    tasks = []
    for bot in ("v15", "v15.1-A"):
        for n_players in (2, 4):
            for i in range(args.games):
                tasks.append((bot, n_players, args.seed_offset + i))

    if args.workers <= 1:
        res = [_play(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            res = list(pool.map(_play, tasks))

    for bot in ("v15", "v15.1-A"):
        gw = gn = 0
        for mode, n_players in (("2p", 2), ("4p", 4)):
            sub = [r for r in res if r[0] == bot and r[1] == n_players]
            w = sum(r[2] for r in sub)
            n = len(sub)
            print(f"[{bot}] {mode} vs V7: W={w}/{n} WR={w/n:.3f} "
                  f"Wilson95%LB={_wilson_lb(w, n):.3f}")
            gw += w
            gn += n
        print(f"[{bot}] AGG vs V7: W={gw}/{gn} WR={gw/gn:.3f} "
              f"Wilson95%LB={_wilson_lb(gw, gn):.3f}\n")


if __name__ == "__main__":
    main()
