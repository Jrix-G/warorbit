"""wo_bench — does V15 + the learned value net beat raw V15?

The Week-1 milestone gate. One seat plays V15 search with the value net
blended into the leaf score (value_lambda > 0); every other seat plays raw
V15 (value_lambda = 0). Identical time budget for both — a fair A/B. 2p and
4p, Wilson 95% CI.

Pass criterion: aggregate Wilson lower bound > 0.5 (V15+net beats raw V15).

Run:
    python -u wo_bench.py --games 20 --lam 0.5 --budget 0.7 --workers 11
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor

import v14_core
import v15_fast_sim as fsim
import v15_search
from local_simulator.official_fast import OfficialFastGame

EPISODE = 250

_LAM = 0.0
_BUDGET = 0.7
_VALUE_FN = None


def _init(lam, budget):
    global _LAM, _BUDGET, _VALUE_FN
    import torch
    torch.set_num_threads(1)
    _LAM, _BUDGET = lam, budget
    from wo_value import load_value_fn
    _VALUE_FN = load_value_fn()


def _wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (p, c - m, c + m)


def _play(task):
    """One game: seat `our` = V15+net, all other seats = raw V15."""
    n_players, seed = task
    g = OfficialFastGame(n_players, seed=seed, episode_steps=EPISODE,
                         use_c_accel=False)
    fs = fsim.from_obs(v14_core.obs_as_dict(g.observation(0)),
                       n_players=n_players, episode_steps=EPISODE)
    fs.n_players = n_players
    our = seed % n_players
    while not fs.done:
        actions = []
        for p in range(n_players):
            o = v15_search.state_to_obs(fs, p)
            if p == our:
                m = v15_search.search(o, None, time_budget=_BUDGET,
                                      value_fn=_VALUE_FN, value_lambda=_LAM)
            else:
                m = v15_search.search(o, None, time_budget=_BUDGET)
            actions.append(m if isinstance(m, list) else [])
        fs = fsim.step(fs, actions)
    sc = fsim.scores(fs)
    best = max(s for i, s in enumerate(sc) if i != our)
    return n_players, 1 if (sc[our] > best and sc[our] > 0) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20,
                    help="games per mode (2p and 4p)")
    ap.add_argument("--lam", type=float, default=0.5,
                    help="value-net blend weight (0 = raw V15)")
    ap.add_argument("--budget", type=float, default=0.7,
                    help="per-move search time budget, seconds")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--seed-offset", type=int, default=4_000_000)
    args = ap.parse_args()

    tasks = [(npl, args.seed_offset + i)
             for npl in (2, 4) for i in range(args.games)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.lam, args.budget)) as pool:
        res = list(pool.map(_play, tasks))

    print(f"V15+net(lambda={args.lam}) vs raw V15  budget={args.budget}s")
    aw = an = 0
    for mode, npl in (("2p", 2), ("4p", 4)):
        sub = [r for r in res if r[0] == npl]
        w = sum(r[1] for r in sub)
        n = len(sub)
        aw += w
        an += n
        p, lo, hi = _wilson(w, n)
        print(f"  {mode}: W={w}/{n} WR={p:.3f} 95%CI=[{lo:.3f},{hi:.3f}]")
    p, lo, hi = _wilson(aw, an)
    verdict = "BEATS raw V15" if lo > 0.5 else "does NOT beat raw V15"
    print(f"  AGG: W={aw}/{an} WR={p:.3f} 95%CI=[{lo:.3f},{hi:.3f}] -> {verdict}")
    print(f"elapsed {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
