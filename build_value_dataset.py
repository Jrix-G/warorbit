"""Phase 1 — extract a 4p win-probability dataset from the top-10 replay corpus.

For each sampled (episode, step, player), compute position features and label
with whether that player finished #1 (rewards[player] == 1). Output: a feature
matrix X and label vector y for fitting a calibrated value function V ~ P(win).

Run:
    python build_value_dataset.py
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

RAW = "D:/warorbit_kaggle_raw/orbit-wars-top10-episodes-2026-05-04/episodes/episodes"
OUT = "analysis/v15_value_dataset_4p.npz"
SAMPLE_EVERY = 5          # sample one state every N steps
SKIP_HEAD = 5             # skip the opening (board not settled)
SKIP_TAIL = 4             # skip the very end (outcome trivially decided)

FEATURE_NAMES = [
    "ship_share", "prod_share", "planet_share",
    "max_opp_ship_share", "max_opp_prod_share",
    "ship_margin", "prod_margin",
    "rank_norm", "step_frac", "alive_frac", "eliminated",
]


def _player_totals(obs, n):
    ships = [0.0] * n
    prod = [0.0] * n
    planets = [0] * n
    for p in obs.get("planets") or []:
        o = int(p[1])
        if 0 <= o < n:
            ships[o] += p[5]
            prod[o] += p[6]
            planets[o] += 1
    for f in obs.get("fleets") or []:
        o = int(f[1])
        if 0 <= o < n:
            ships[o] += f[6]
    return ships, prod, planets


def _features(ships, prod, planets, n, player, step):
    tot_s = sum(ships) or 1.0
    tot_p = sum(prod) or 1.0
    tot_pl = sum(planets) or 1
    ss = ships[player] / tot_s
    ps = prod[player] / tot_p
    pls = planets[player] / tot_pl
    opp_s = [ships[q] / tot_s for q in range(n) if q != player]
    opp_p = [prod[q] / tot_p for q in range(n) if q != player]
    max_os = max(opp_s) if opp_s else 0.0
    max_op = max(opp_p) if opp_p else 0.0
    rank = sorted(range(n), key=lambda q: -ships[q]).index(player)
    alive = sum(1 for q in range(n) if ships[q] > 0 or planets[q] > 0)
    elim = 1.0 if (ships[player] <= 0 and planets[player] == 0) else 0.0
    return [
        ss, ps, pls,
        max_os, max_op,
        ss - max_os, ps - max_op,
        rank / (n - 1) if n > 1 else 0.0,
        min(step / 500.0, 1.0),
        alive / n,
        elim,
    ]


def main():
    files = [p for p in sorted(glob.glob(os.path.join(RAW, "*.json")))
             if os.path.basename(p)[:-5].isdigit()]
    X, y = [], []
    eps = skipped = 0
    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            skipped += 1
            continue
        steps = d.get("steps") or []
        rewards = d.get("rewards") or []
        if not steps or not rewards:
            skipped += 1
            continue
        n = len(steps[0])
        if n != 4:                       # 4p corpus only
            continue
        eps += 1
        T = len(steps)
        for t in range(SKIP_HEAD, T - SKIP_TAIL, SAMPLE_EVERY):
            obs = steps[t][0].get("observation") or {}
            if not obs.get("planets"):
                continue
            ships, prod, planets = _player_totals(obs, n)
            step = int(obs.get("step", t) or t)
            for p in range(n):
                X.append(_features(ships, prod, planets, n, p, step))
                y.append(1 if rewards[p] == 1 else 0)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    os.makedirs("analysis", exist_ok=True)
    np.savez(OUT, X=X, y=y, feature_names=np.array(FEATURE_NAMES))
    print(f"episodes used={eps} skipped={skipped}")
    print(f"dataset: X={X.shape} y={y.shape} positive_rate={y.mean():.3f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
