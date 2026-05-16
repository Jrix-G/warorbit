"""Benchmark a V17 checkpoint vs V15 (RCC+V7).

Phase-0 gate: the warm-started net should play ~= V15.
Iteration gates: a trained net should beat V15.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE python -u bench_v17.py --ckpt analysis/v17_warmstart.pt --games 24
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import v14_core
import v15_fast_sim as fsim
import v15_search
from local_simulator.official_fast import OfficialFastGame
from v17_agent import make_agent

EPISODE = 250


def _wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (p, c - m, c + m)


_CKPT = None
_NSIMS = None


def _play(task):
    n_players, seed = task
    v17 = make_agent(_CKPT, n_sims=_NSIMS)
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
            m = v17(o) if p == our else v15_search.search(o, None)
            actions.append(m if isinstance(m, list) else [])
        fs = fsim.step(fs, actions)
    sc = fsim.scores(fs)
    best = max(s for i, s in enumerate(sc) if i != our)
    return n_players, 1 if (sc[our] > best and sc[our] > 0) else 0


def _init(ckpt, nsims):
    global _CKPT, _NSIMS
    _CKPT, _NSIMS = ckpt, nsims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-sims", type=int, default=120)
    ap.add_argument("--seed-offset", type=int, default=9_100_000)
    args = ap.parse_args()

    tasks = []
    for n_players in (2, 4):
        for i in range(args.games):
            tasks.append((n_players, args.seed_offset + i))

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.ckpt, args.n_sims)) as pool:
        res = list(pool.map(_play, tasks))

    print(f"V17 ({args.ckpt}) vs V15")
    aw = an = 0
    for mode, npl in (("2p", 2), ("4p", 4)):
        sub = [r for r in res if r[0] == npl]
        w = sum(r[1] for r in sub)
        n = len(sub)
        aw += w
        an += n
        p, lo, hi = _wilson(w, n)
        print(f"  {mode}: V17 W={w}/{n} WR={p:.3f} 95%CI=[{lo:.3f},{hi:.3f}]")
    p, lo, hi = _wilson(aw, an)
    print(f"  AGG: W={aw}/{an} WR={p:.3f} 95%CI=[{lo:.3f},{hi:.3f}]")
    print(f"elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
