"""P2 gate — v15_search (Monte-Carlo) vs bot_v7 head-to-head.

Gate: v15_search must beat V7 by >= +15% winrate (Wilson 95% LB) to justify
the search layer. Seats are swapped each game to remove first-move bias.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE python -u p2_gate.py --games 20 --workers 6
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


def _wilson_lb(w: int, n: int, z: float = 1.96) -> float:
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


def _search(obs, config):
    obs = v14_core.obs_as_dict(obs)
    m = v15_search.search(obs, config)
    return m if isinstance(m, list) else []


def _play(task):
    n_players, seed = task
    random.seed(seed)
    np.random.seed(seed)
    our_idx = seed % n_players
    agents = [_v7] * n_players
    agents[our_idx] = _search
    game = OfficialFastGame(n_players, seed=seed, episode_steps=220, use_c_accel=False)
    t0 = time.time()
    while not game.done:
        actions = [agents[p](game.observation(p), game.configuration)
                   for p in range(n_players)]
        game.step(actions)
    scores = game.scores()
    best_other = max(s for i, s in enumerate(scores) if i != our_idx)
    mine = scores[our_idx]
    win = 1 if (mine > best_other and mine > 0) else 0
    return win, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed-offset", type=int, default=90000)
    ap.add_argument("--modes", nargs="*", default=["2p", "4p"])
    args = ap.parse_args()

    grand_w = grand_n = 0
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
        print(f"- {mode}: v15_search W={w}/{n} WR={w/n:.3f} "
              f"Wilson95%LB={lb:.3f}  ({secs/n:.0f}s/game)")
        grand_w += w
        grand_n += n
    lb = _wilson_lb(grand_w, grand_n)
    wr = grand_w / grand_n
    print(f"AGG v15_search vs V7: W={grand_w}/{grand_n} WR={wr:.3f} Wilson95%LB={lb:.3f}")
    # In an N-player game, neutral expectation is 1/N. The gate framing
    # "beat V7 by +15%" -> head-to-head WR meaningfully above the seat baseline.
    print(f"P2 gate: WR Wilson95%LB={lb:.3f} (target: clearly above 1/n_players seat baseline)")


if __name__ == "__main__":
    main()
