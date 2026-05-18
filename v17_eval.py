"""v17_eval — measure a V17 checkpoint's real strength vs V15 (RCC) and V7.

This is the evaluation gate: self-play loss going down proves nothing about
ladder strength. Run this every few iterations on the latest checkpoint to get
a real winrate signal.

V17 plays with greedy MCTS (temperature 0). Opponents fill every other slot.
2p and 4p, with Wilson 95% CI.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE python -u v17_eval.py --ckpt analysis/v17_iter4.pt \
        --games 50 --workers 7 --n-sims 80 --opponent both
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import bot_v7
import v14_core
import v15_fast_sim as fsim
import v15_search
from local_simulator.official_fast import OfficialFastGame
from v17_mcts import mcts_move
from v17_net import V17Net

EPISODE = 250

_NET = None
_NSIMS = None


def _init(ckpt, nsims):
    global _NET, _NSIMS
    import torch
    torch.set_num_threads(1)
    c = torch.load(ckpt, map_location="cpu")
    net = V17Net(d=c["d"])
    net.load_state_dict(c["state_dict"])
    net.eval()
    _NET, _NSIMS = net, nsims


def _wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (p, c - m, c + m)


def _opp_move(opponent, o):
    if opponent == "v15":
        m = v15_search.search(o, None)
    else:                                       # v7
        try:
            m = bot_v7.agent(o, None)
        except TypeError:
            m = bot_v7.agent(o)
    return m if isinstance(m, list) else []


def _play(task):
    opponent, n_players, seed = task
    rng = np.random.default_rng(seed)
    g = OfficialFastGame(n_players, seed=seed, episode_steps=EPISODE,
                         use_c_accel=False)
    obs0 = v14_core.obs_as_dict(g.observation(0))
    fs = fsim.from_obs(obs0, n_players=n_players, episode_steps=EPISODE)
    fs.n_players = n_players
    our = seed % n_players
    while not fs.done:
        actions = []
        for p in range(n_players):
            if p == our:
                action, _ = mcts_move(_NET, fs, p, n_sims=_NSIMS, rng=rng,
                                      temperature=0.0)
                actions.append(action if isinstance(action, list) else [])
            else:
                o = v15_search.state_to_obs(fs, p)
                actions.append(_opp_move(opponent, o))
        fs = fsim.step(fs, actions)
    sc = fsim.scores(fs)
    best = max(s for i, s in enumerate(sc) if i != our)
    return opponent, n_players, 1 if (sc[our] > best and sc[our] > 0) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=50,
                    help="games per (opponent, n_players) cell")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--n-sims", type=int, default=80)
    ap.add_argument("--opponent", choices=("v15", "v7", "both"), default="both")
    ap.add_argument("--seed-offset", type=int, default=9_200_000)
    args = ap.parse_args()

    opps = ("v15", "v7") if args.opponent == "both" else (args.opponent,)
    tasks = []
    for opp in opps:
        for n_players in (2, 4):
            for i in range(args.games):
                tasks.append((opp, n_players, args.seed_offset + i))

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.ckpt, args.n_sims)) as pool:
        res = list(pool.map(_play, tasks))

    print(f"V17 ({args.ckpt}) n_sims={args.n_sims}")
    for opp in opps:
        aw = an = 0
        for mode, npl in (("2p", 2), ("4p", 4)):
            sub = [r for r in res if r[0] == opp and r[1] == npl]
            w = sum(r[2] for r in sub)
            n = len(sub)
            aw += w
            an += n
            p, lo, hi = _wilson(w, n)
            print(f"  vs {opp} {mode}: W={w}/{n} WR={p:.3f} "
                  f"95%CI=[{lo:.3f},{hi:.3f}]")
        p, lo, hi = _wilson(aw, an)
        print(f"  vs {opp} AGG: W={aw}/{an} WR={p:.3f} "
              f"95%CI=[{lo:.3f},{hi:.3f}]")
    print(f"elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
