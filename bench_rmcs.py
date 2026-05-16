"""V16 Phase-1 gate — RMCS vs V15 (RCC+V7), head-to-head.

One RMCS agent vs (n-1) V15 agents, on the validated v15_fast_sim engine.
RMCS is depth-3; V15 is the deployed RCC+V7. Success gate: RMCS must beat
V15 decisively (the depth advantage should be near-total).

Run:
    KMP_DUPLICATE_LIB_OK=TRUE python -u bench_rmcs.py --games 24 --workers 8
"""

from __future__ import annotations

import argparse
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import v14_core
import v15_fast_sim as fsim
import v15_search
import v16_rmcs
from local_simulator.official_fast import OfficialFastGame

EPISODE = 250


def _wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (p, c - m, c + m)


def _play(task):
    n_players, seed, depth = task
    random.seed(seed)
    np.random.seed(seed)
    g = OfficialFastGame(n_players, seed=seed, episode_steps=EPISODE,
                         use_c_accel=False)
    obs0 = v14_core.obs_as_dict(g.observation(0))
    fs = fsim.from_obs(obs0, n_players=n_players, episode_steps=EPISODE)
    fs.n_players = n_players
    our = seed % n_players
    while not fs.done:
        actions = []
        for p in range(n_players):
            o = v15_search.state_to_obs(fs, p)
            if p == our:
                m = v16_rmcs.search(o, None, depth=depth)
            else:
                m = v15_search.search(o, None)
            actions.append(m if isinstance(m, list) else [])
        fs = fsim.step(fs, actions)
    sc = fsim.scores(fs)
    best = max(s for i, s in enumerate(sc) if i != our)
    win = 1 if (sc[our] > best and sc[our] > 0) else 0
    return n_players, win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=24, help="games per mode")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--seed-offset", type=int, default=8_000_000)
    args = ap.parse_args()

    tasks = []
    for n_players in (2, 4):
        for i in range(args.games):
            tasks.append((n_players, args.seed_offset + i, args.depth))

    t0 = time.time()
    if args.workers <= 1:
        res = [_play(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            res = list(pool.map(_play, tasks))

    print(f"V16 RMCS (depth {args.depth}) vs V15 (RCC+V7)")
    agg_w = agg_n = 0
    for mode, n_players in (("2p", 2), ("4p", 4)):
        sub = [r for r in res if r[0] == n_players]
        w = sum(r[1] for r in sub)
        n = len(sub)
        agg_w += w
        agg_n += n
        p, lo, hi = _wilson(w, n)
        print(f"  {mode}: RMCS W={w}/{n} WR={p:.3f} 95%CI=[{lo:.3f},{hi:.3f}]")
    p, lo, hi = _wilson(agg_w, agg_n)
    print(f"  AGG: W={agg_w}/{agg_n} WR={p:.3f} 95%CI=[{lo:.3f},{hi:.3f}]")
    print(f"elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
