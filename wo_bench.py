"""wo_bench — does V15 + the learned value net beat raw V15?  (lambda sweep)

The Week-1 milestone gate. One seat plays V15 search with the value net
blended into the leaf score at weight lambda; every other seat plays raw V15.
Identical time budget for both — a fair A/B. Sweeps several lambda values,
2p and 4p, Wilson 95% CI per lambda.

The search compares ~69 near-identical leaves and takes the argmax, which
amplifies the net's local noise — so a small lambda (ESC keeps the clean
ranking, the net only nudges) is expected to be safer than a large one.

Pass: some lambda's aggregate Wilson lower bound > 0.5.

Run:
    python -u wo_bench.py --lams 0.1,0.25 --games 16 --budget 0.7 --workers 11
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

_BUDGET = 0.7
_VALUE_FN = None


_CKPT = "analysis/wo_value.pt"
_ESC_WEIGHTS = None
_POLICY_FN = None


def _init(budget, ckpt, esc_weights_path, policy_ckpt):
    global _BUDGET, _VALUE_FN, _CKPT, _ESC_WEIGHTS, _POLICY_FN
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU only in workers
    import torch
    torch.set_num_threads(1)
    _BUDGET = budget
    _CKPT = ckpt
    if esc_weights_path:
        import v15_eval
        _ESC_WEIGHTS = v15_eval.EvalWeights.load(esc_weights_path)
    if ckpt and not esc_weights_path:
        from wo_value import load_value_fn
        _VALUE_FN = load_value_fn(ckpt)
    if policy_ckpt:
        from wo_policy_fn import load_policy_fn
        _POLICY_FN = load_policy_fn(policy_ckpt)


def _wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (p, c - m, c + m)


def _play(task):
    """One game: seat `our` = V15+net(lambda), every other seat = raw V15."""
    lam, n_players, seed = task
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
                m = v15_search.search(o, _ESC_WEIGHTS, time_budget=_BUDGET,
                                      value_fn=_VALUE_FN, value_lambda=lam,
                                      policy_fn=_POLICY_FN)
            else:
                m = v15_search.search(o, None, time_budget=_BUDGET)
            actions.append(m if isinstance(m, list) else [])
        fs = fsim.step(fs, actions)
    sc = fsim.scores(fs)
    best = max(s for i, s in enumerate(sc) if i != our)
    return lam, n_players, 1 if (sc[our] > best and sc[our] > 0) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lams", default="0.1,0.25",
                    help="comma-separated value-net blend weights to sweep")
    ap.add_argument("--games", type=int, default=16,
                    help="games per (lambda, mode)")
    ap.add_argument("--budget", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--seed-offset", type=int, default=4_000_000)
    ap.add_argument("--ckpt", default="",
                    help="value net checkpoint to benchmark (empty = no net)")
    ap.add_argument("--esc-weights", default="",
                    help="optimised EvalWeights .npz (mutually exclusive with --ckpt)")
    ap.add_argument("--policy-ckpt", default="",
                    help="policy net checkpoint for candidate generation")
    ap.add_argument("--modes", default="2,4",
                    help="player counts to test, comma-separated (e.g. 2 or 2,4)")
    args = ap.parse_args()

    lams = [float(x) for x in args.lams.split(",") if x.strip()]
    modes = [int(x) for x in args.modes.split(",") if x.strip()]
    tasks = [(lam, npl, args.seed_offset + i)
             for lam in lams
             for npl in modes
             for i in range(args.games)]

    # accumulators: {(lam, npl): [wins, total]}
    acc = {(lam, npl): [0, 0] for lam in lams for npl in modes}

    def _print_running():
        print("\n--- running totals ---", flush=True)
        for lam in lams:
            aw = an = 0
            parts = []
            for npl in modes:
                w, n = acc[(lam, npl)]
                aw += w; an += n
                parts.append(f"{npl}p {w}/{n}" + (f" WR={w/n:.2f}" if n else ""))
            p, lo, hi = _wilson(aw, an)
            verdict = "BEATS V15" if lo > 0.5 else "~tie" if hi > 0.5 else "loses"
            print(f"  lam={lam}: {' | '.join(parts)} | "
                  f"AGG {aw}/{an} WR={p:.3f} CI=[{lo:.3f},{hi:.3f}] -> {verdict}",
                  flush=True)

    from concurrent.futures import as_completed
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.budget, args.ckpt,
                                       args.esc_weights,
                                       args.policy_ckpt)) as pool:
        futs = {pool.submit(_play, t): t for t in tasks}
        for fut in as_completed(futs):
            lam, npl, win = fut.result()
            acc[(lam, npl)][0] += win
            acc[(lam, npl)][1] += 1
            done += 1
            print(f"[{done}/{len(tasks)}] lam={lam} {npl}p win={win} "
                  f"({acc[(lam,npl)][0]}/{acc[(lam,npl)][1]})", flush=True)
            if done % args.workers == 0:
                _print_running()

    print(f"\n=== FINAL  budget={args.budget}s  games/cell={args.games} ===")
    _print_running()
    print(f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
