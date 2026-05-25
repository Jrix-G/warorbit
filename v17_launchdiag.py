"""v17_launchdiag — is a V17 checkpoint passive or aggressive-but-losing?

Plays the net (greedy MCTS) vs V15 and logs launches/game for both sides plus
final scores. The single number that matters: launches/game. Collapsed warmstart
did 5-12; V15 does 60-130. If the net is near V15's count it is aggressive
(learning, just weak); if near 10 the collapse fix did not take.

Run: python -u v17_launchdiag.py --ckpt analysis/v17_iter4.pt --games 4
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import v14_core
import v15_fast_sim as fsim
import v15_search
from local_simulator.official_fast import OfficialFastGame

EPISODE = 250
_NET = None
_NSIMS = 50


def _init(ckpt, nsims):
    global _NET, _NSIMS
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(1)
    from v17_net import V17Net
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = V17Net(d=c["d"])
    net.load_state_dict(c["state_dict"])
    net.eval()
    _NET, _NSIMS = net, nsims


def _play(task):
    n_players, seed = task
    from v17_mcts import mcts_move
    rng = np.random.default_rng(seed)
    g = OfficialFastGame(n_players, seed=seed, episode_steps=EPISODE,
                         use_c_accel=False)
    fs = fsim.from_obs(v14_core.obs_as_dict(g.observation(0)),
                       n_players=n_players, episode_steps=EPISODE)
    fs.n_players = n_players
    our = seed % n_players
    our_l = opp_l = 0
    our_steps_active = 0
    steps = 0
    while not fs.done:
        actions = []
        for p in range(n_players):
            if p == our:
                m, _ = mcts_move(_NET, fs, p, n_sims=_NSIMS, rng=rng,
                                 temperature=0.0)
                m = m if isinstance(m, list) else []
                our_l += len(m)
                if m:
                    our_steps_active += 1
            else:
                o = v15_search.state_to_obs(fs, p)
                m = v15_search.search(o, None)
                m = m if isinstance(m, list) else []
                opp_l += len(m)
            actions.append(m)
        fs = fsim.step(fs, actions)
        steps += 1
    sc = fsim.scores(fs)
    best = max(s for i, s in enumerate(sc) if i != our)
    win = 1 if (sc[our] > best and sc[our] > 0) else 0
    return (n_players, seed, win, our_l, opp_l, our_steps_active, steps,
            float(sc[our]), float(best))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-sims", type=int, default=50)
    ap.add_argument("--seed-offset", type=int, default=30_000_000)
    args = ap.parse_args()

    tasks = [(npl, args.seed_offset + i)
             for npl in (2, 4) for i in range(args.games)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.ckpt, args.n_sims)) as pool:
        res = list(pool.map(_play, tasks))

    print(f"\n=== launch diagnostic {args.ckpt} n_sims={args.n_sims} ===")
    for npl in (2, 4):
        sub = [r for r in res if r[0] == npl]
        for (_, seed, win, ol, opl, act, steps, sc, best) in sub:
            print(f"  {npl}p seed={seed} win={win} launches us={ol} "
                  f"v15={opl} active_steps={act}/{steps} "
                  f"score={sc:.0f} vs {best:.0f}")
        avg_ol = np.mean([r[3] for r in sub])
        avg_opl = np.mean([r[4] for r in sub])
        print(f"  -> {npl}p AVG launches us={avg_ol:.1f} v15={avg_opl:.1f}")


if __name__ == "__main__":
    main()
