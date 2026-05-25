"""wo_diag — why does a better predictor make the search worse?

The milestone benchmark showed V15+net(lambda=0.5) LOSING to raw V15, yet the
net correlates with game outcome far better than ESC on replay states. The
resolution hypothesis: the search does not query the net on replay-like
states — it queries it on POST-CONTINUATION leaf states (apply a combo, run a
quiescence horizon, evaluate). If the net is accurate on live replay states
but unreliable on those quiet leaves, a search optimising against it drives
into its blind spots.

This script measures, per (state, player): ESC and net correlation with the
outcome, on the RAW replay state AND on the leaf after a passive horizon-H
continuation (what the search actually evaluates). A large net-correlation
drop from raw to leaf confirms the distribution-shift diagnosis.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import glob
import json

import numpy as np

import v15_eval
import v15_fast_sim as fsim
from wo_dataset import _board_obs, _episode_id, _episode_values
from wo_value import load_value_fn

GLOBS = ["replays/top1-05-05/episode-*.json",
         "V15LogsReplays/episode-*.json",
         "logs/logsfromVPS/episode-*.json"]
HORIZON = 24                                         # v15_search default


def _corr(x, y):
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main():
    value_fn = load_value_fn()
    files = sorted({f for g in GLOBS for f in glob.glob(g)})
    seen = set()
    rows = []                  # (npl, esc_raw, net_raw, esc_leaf, net_leaf, out)
    for f in files:
        try:
            ep = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        eid = _episode_id(ep, f)
        if eid in seen:
            continue
        seen.add(eid)
        steps = ep.get("steps") or []
        if not steps:
            continue
        npl = len(steps[0])
        es = int((ep.get("configuration") or {}).get("episodeSteps", 500))
        val = _episode_values(ep, npl)
        for t, st in enumerate(steps):
            obs = _board_obs(st)
            if obs is None:
                continue
            fs = fsim.from_obs(obs, n_players=npl, episode_steps=es)
            fs.n_players = npl
            fs.step = t
            if len(fs.planets) == 0:
                continue
            # passive quiescence continuation — what _eval_combo evaluates
            leaf = fs
            for _ in range(HORIZON):
                if leaf.done:
                    break
                leaf = fsim.step(leaf, [[] for _ in range(npl)])
            for p in range(npl):
                rows.append((npl,
                             v15_eval.evaluate(fs, p, v15_eval.ESC),
                             value_fn(fs, p),
                             v15_eval.evaluate(leaf, p, v15_eval.ESC),
                             value_fn(leaf, p),
                             val[p]))

    a = np.array(rows, dtype=np.float64)
    npl = a[:, 0]
    print(f"[wo_diag] {len(a)} (state,player) rows from {len(seen)} episodes "
          f"(corr is train-inflated — the net trained on most of these)")

    def report(name, mask):
        sub = a[mask]
        if len(sub) == 0:
            return
        o = sub[:, 5]
        er, nr, el, nl = sub[:, 1], sub[:, 2], sub[:, 3], sub[:, 4]
        print(f"[{name}] n={len(sub)}")
        print(f"  RAW replay state : corr(ESC)={_corr(er, o):+.3f}  "
              f"corr(net)={_corr(nr, o):+.3f}")
        print(f"  LEAF after {HORIZON}-step continuation : "
              f"corr(ESC)={_corr(el, o):+.3f}  corr(net)={_corr(nl, o):+.3f}")
        drop_n = _corr(nr, o) - _corr(nl, o)
        drop_e = _corr(er, o) - _corr(el, o)
        print(f"  -> net corr drop raw->leaf = {drop_n:+.3f}   "
              f"ESC corr drop = {drop_e:+.3f}")

    report("ALL", np.ones(len(a), bool))
    report("2p", npl == 2)
    report("4p", npl == 4)


if __name__ == "__main__":
    main()
