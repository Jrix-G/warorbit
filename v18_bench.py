"""v18_bench — benchmark the V18 search agent (MCTS + supervised nets) vs V15.

V18 plays one seat with MCTS; every other seat plays V15 (RCC). Fair: V15 is
the ~975-ELO deployed bot, so beating it clearly is the Week-1 gate.

Rich per-game logging (-> --log file) so each run can be analysed: launches
made by each side, score margin, where games are won/lost. Wilson 95% CI.

Run:
    python -u v18_bench.py --games 24 --modes 2,4 --n-sims 64 --workers 8 \
        --search puct --log analysis/v18_run1.jsonl
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import v14_core
import v15_fast_sim as fsim
import v15_search
from local_simulator.official_fast import OfficialFastGame

EPISODE = 250

_EVAL = None
_NSIMS = 64
_SEARCH = "puct"


def _init(policy_ckpt, value_ckpt, nsims, search, evaluator, rollout):
    global _EVAL, _NSIMS, _SEARCH
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(1)
    import v17_mcts
    if evaluator == "rollout-esc":
        from v18_agent import RolloutESCEvaluator
        _EVAL = RolloutESCEvaluator(policy_ckpt, device="cpu", rollout=rollout)
    else:
        from v18_agent import SupervisedEvaluator
        _EVAL = SupervisedEvaluator(policy_ckpt, value_ckpt, device="cpu")
    v17_mcts.set_evaluator(_EVAL)
    _NSIMS = nsims
    _SEARCH = search


def _wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (p, c - m, c + m)


def _our_move(fs, player, rng):
    if _SEARCH == "gumbel":
        from v18_search import gumbel_move
        return gumbel_move(fs, player, n_sims=_NSIMS, rng=rng)
    from v17_mcts import mcts_move
    action, _ = mcts_move(None, fs, player, n_sims=_NSIMS, rng=rng,
                          temperature=0.0)
    return action


def _play(task):
    n_players, seed = task
    rng = np.random.default_rng(seed)
    g = OfficialFastGame(n_players, seed=seed, episode_steps=EPISODE,
                         use_c_accel=False)
    fs = fsim.from_obs(v14_core.obs_as_dict(g.observation(0)),
                       n_players=n_players, episode_steps=EPISODE)
    fs.n_players = n_players
    our = seed % n_players

    our_launches = 0
    opp_launches = 0
    steps = 0
    while not fs.done:
        actions = []
        for p in range(n_players):
            if p == our:
                m = _our_move(fs, p, rng)
                m = m if isinstance(m, list) else []
                our_launches += len(m)
            else:
                o = v15_search.state_to_obs(fs, p)
                m = v15_search.search(o, None)
                m = m if isinstance(m, list) else []
                opp_launches += len(m)
            actions.append(m)
        fs = fsim.step(fs, actions)
        steps += 1

    sc = fsim.scores(fs)
    best = max(s for i, s in enumerate(sc) if i != our)
    win = 1 if (sc[our] > best and sc[our] > 0) else 0
    return {
        "mode": n_players, "seed": seed, "our_seat": our, "win": win,
        "our_score": float(sc[our]), "best_opp": float(best),
        "all_scores": [float(s) for s in sc],
        "our_launches": our_launches, "opp_launches": opp_launches,
        "steps": steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=24, help="games per mode")
    ap.add_argument("--modes", default="2,4")
    ap.add_argument("--n-sims", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed-offset", type=int, default=20_000_000)
    ap.add_argument("--search", choices=("puct", "gumbel"), default="puct")
    ap.add_argument("--evaluator", choices=("supervised", "rollout-esc"),
                    default="rollout-esc")
    ap.add_argument("--rollout", type=int, default=14)
    ap.add_argument("--policy-ckpt", default="analysis/wo_policy.pt")
    ap.add_argument("--value-ckpt", default="analysis/wo_value.pt")
    ap.add_argument("--log", default="")
    args = ap.parse_args()

    modes = [int(x) for x in args.modes.split(",") if x.strip()]
    tasks = [(npl, args.seed_offset + i)
             for npl in modes for i in range(args.games)]
    acc = {npl: [0, 0] for npl in modes}
    rows = []

    print(f"V18 search={args.search} evaluator={args.evaluator} "
          f"rollout={args.rollout} n_sims={args.n_sims}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.policy_ckpt, args.value_ckpt,
                                       args.n_sims, args.search,
                                       args.evaluator, args.rollout)) as pool:
        futs = {pool.submit(_play, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            acc[r["mode"]][0] += r["win"]
            acc[r["mode"]][1] += 1
            done += 1
            print(f"[{done}/{len(tasks)}] {r['mode']}p seed={r['seed']} "
                  f"win={r['win']} score={r['our_score']:.0f} "
                  f"vs{r['best_opp']:.0f} L={r['our_launches']}/"
                  f"{r['opp_launches']}", flush=True)

    print(f"\n=== FINAL  search={args.search}  n_sims={args.n_sims} ===")
    aw = an = 0
    for npl in modes:
        w, n = acc[npl]
        aw += w
        an += n
        p, lo, hi = _wilson(w, n)
        sub = [r for r in rows if r["mode"] == npl]
        avg_ol = np.mean([r["our_launches"] for r in sub]) if sub else 0
        avg_pl = np.mean([r["opp_launches"] for r in sub]) if sub else 0
        print(f"  {npl}p: W={w}/{n} WR={p:.3f} CI=[{lo:.3f},{hi:.3f}] | "
              f"avg launches us={avg_ol:.0f} v15={avg_pl:.0f}")
    p, lo, hi = _wilson(aw, an)
    print(f"  AGG: W={aw}/{an} WR={p:.3f} CI=[{lo:.3f},{hi:.3f}]")
    print(f"elapsed {(time.time()-t0)/60:.1f} min", flush=True)

    if args.log:
        with open(args.log, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"per-game log -> {args.log}")


if __name__ == "__main__":
    main()
