"""Overnight benchmark — isolate the 4p value-function's effect.

Head-to-head: search(use_value_fn=True) vs search(use_value_fn=False).
Everything else identical, so any winrate gap is the value function alone.

  2p : regression check — VF only changes 4p, so this should sit near 50%.
  4p : the real test — does the value function fix 4p?

Writes analysis/V15_OVERNIGHT_RESULT.txt so the result survives the night.

Run (VPS, 8 workers):
    KMP_DUPLICATE_LIB_OK=TRUE python -u overnight_bench.py --games 250 --workers 8
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


def _vf_on(obs, config):
    obs = v14_core.obs_as_dict(obs)
    m = v15_search.search(obs, config, time_budget=0.7, use_value_fn=True)
    return m if isinstance(m, list) else []


def _vf_off(obs, config):
    obs = v14_core.obs_as_dict(obs)
    m = v15_search.search(obs, config, time_budget=0.7, use_value_fn=False)
    return m if isinstance(m, list) else []


def _play(task):
    n_players, seed = task
    random.seed(seed)
    np.random.seed(seed)
    our_idx = seed % n_players
    agents = [_vf_off] * n_players
    agents[our_idx] = _vf_on
    game = OfficialFastGame(n_players, seed=seed, episode_steps=220, use_c_accel=False)
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
    ap.add_argument("--games", type=int, default=250, help="games per mode")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed-offset", type=int, default=200000)
    args = ap.parse_args()

    tasks = []
    for n_players in (2, 4):
        for i in range(args.games):
            tasks.append((n_players, args.seed_offset + i))

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        res = list(pool.map(_play, tasks))

    lines = ["V15 overnight benchmark — VF-on vs VF-off (value fn isolates 4p)"]
    for mode, n_players in (("2p", 2), ("4p", 4)):
        sub = [r for r in res if r[0] == n_players]
        w = sum(r[1] for r in sub)
        n = len(sub)
        p, lo, hi = _wilson(w, n)
        verdict = ("VF helps" if lo > 0.5 else
                   "VF hurts" if hi < 0.5 else "inconclusive")
        lines.append(f"{mode}: VF-on W={w}/{n} WR={p:.3f} "
                     f"95%CI=[{lo:.3f},{hi:.3f}]  -> {verdict}")
    lines.append(f"elapsed {(time.time()-t0)/60:.0f} min")
    out = "\n".join(lines)
    print(out)
    with open("analysis/V15_OVERNIGHT_RESULT.txt", "w") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
