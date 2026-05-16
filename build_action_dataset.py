"""P3 phase 1 — extract a behavioral-cloning dataset from the top-10 corpus.

For every (replay, step, player) we look at the move the player made and, for
every candidate (source planet, target planet) pair, emit a feature vector
and a binary label: did that player launch a fleet src->tgt this step?

The launch's stored angle aims at the *intercept point*, not the target's
current position, so we match a launch to the planet whose current bearing
from the source is angularly closest (tolerance ANGLE_TOL).

Positives are kept in full; negatives are subsampled to NEG_RATIO x positives
to keep the logistic fit from collapsing to the majority class.

Output: analysis/v15_action_dataset.npz  (X, y, feature_names)

Run:
    python build_action_dataset.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import random

import numpy as np

RAW = "D:/warorbit_kaggle_raw/orbit-wars-top10-episodes-2026-05-04/episodes/episodes"
OUT = "analysis/v15_action_dataset.npz"

SAMPLE_EVERY = 3          # sample one step every N
SKIP_HEAD = 4
SKIP_TAIL = 4
ANGLE_TOL = 0.35          # rad — launch-to-target bearing match tolerance
NEG_RATIO = 4             # negatives kept per positive (final dataset)
NEG_PRESAMPLE = 0.03      # on-the-fly keep prob for negatives (caps memory)
SEED = 0

FEATURE_NAMES = [
    "src_ship_share", "tgt_prod_share", "dist_norm",
    "tgt_is_enemy", "tgt_is_neutral", "combat_margin",
    "step_frac", "my_ship_share", "my_prod_share",
    "tgt_defense_ratio", "src_prod_norm", "is_four_player",
]

BOARD_DIAG = 141.42


def _totals(planets, fleets, n):
    ships = [0.0] * n
    prod = [0.0] * n
    for p in planets:
        o = int(p[1])
        if 0 <= o < n:
            ships[o] += float(p[5])
            prod[o] += float(p[6])
    for f in fleets:
        o = int(f[1])
        if 0 <= o < n:
            ships[o] += float(f[6])
    return ships, prod


def _features(src, tgt, player, n, step, ships, prod):
    tot_s = sum(ships) or 1.0
    tot_p = sum(prod) or 1.0
    sx, sy = float(src[2]), float(src[3])
    tx, ty = float(tgt[2]), float(tgt[3])
    dist = math.hypot(tx - sx, ty - sy)
    s_ships = float(src[5])
    t_ships = float(tgt[5])
    t_owner = int(tgt[1])
    return [
        s_ships / tot_s,
        float(tgt[6]) / tot_p,
        dist / BOARD_DIAG,
        1.0 if (t_owner >= 0 and t_owner != player) else 0.0,
        1.0 if t_owner == -1 else 0.0,
        (s_ships - t_ships) / (s_ships + t_ships + 1.0),
        min(step / 500.0, 1.0),
        ships[player] / tot_s,
        prod[player] / tot_p,
        t_ships / (s_ships + 1.0),
        min(float(src[6]) / 10.0, 1.0),
        1.0 if n >= 4 else 0.0,
    ]


def _match_target(src, planets_by_id, angle):
    """Return the planet id whose bearing from `src` best matches `angle`."""
    sx, sy = float(src[2]), float(src[3])
    best_id, best_diff = None, 999.0
    for pid, pp in planets_by_id.items():
        if pid == int(src[0]):
            continue
        a = math.atan2(float(pp[3]) - sy, float(pp[2]) - sx)
        diff = abs(a - angle)
        if diff > math.pi:
            diff = 2 * math.pi - diff
        if diff < best_diff:
            best_diff, best_id = diff, pid
    return best_id if best_diff < ANGLE_TOL else None


def main():
    rng = random.Random(SEED)
    files = [p for p in sorted(glob.glob(os.path.join(RAW, "*.json")))
             if os.path.basename(p)[:-5].isdigit()]
    pos_X, neg_X = [], []
    eps = skipped = 0

    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            skipped += 1
            continue
        steps = d.get("steps") or []
        if not steps:
            skipped += 1
            continue
        n = len(steps[0])
        if n not in (2, 4):
            continue
        eps += 1
        T = len(steps)
        for t in range(SKIP_HEAD, T - SKIP_TAIL, SAMPLE_EVERY):
            obs = steps[t][0].get("observation") or {}
            planets = obs.get("planets") or []
            fleets = obs.get("fleets") or []
            if not planets:
                continue
            planets_by_id = {int(p[0]): p for p in planets}
            step = int(obs.get("step", t) or t)
            ships, prod = _totals(planets, fleets, n)

            for p in range(n):
                action = steps[t][p].get("action") or []
                # planets owned by p
                mine = [pl for pl in planets if int(pl[1]) == p]
                if not mine:
                    continue
                # which (src,tgt) pairs were actually launched
                launched = set()
                for mv in action:
                    if not (isinstance(mv, list) and len(mv) >= 2):
                        continue
                    src = planets_by_id.get(int(mv[0]))
                    if src is None or int(src[1]) != p:
                        continue
                    tgt_id = _match_target(src, planets_by_id, float(mv[1]))
                    if tgt_id is not None:
                        launched.add((int(mv[0]), tgt_id))
                # emit a sample for every candidate (src,tgt) pair
                for src in mine:
                    for tgt in planets:
                        if int(tgt[0]) == int(src[0]):
                            continue
                        is_pos = (int(src[0]), int(tgt[0])) in launched
                        # negatives: probabilistic keep to cap memory; the
                        # exact NEG_RATIO trim happens after the full pass
                        if not is_pos and rng.random() > NEG_PRESAMPLE:
                            continue
                        feat = _features(src, tgt, p, n, step, ships, prod)
                        if is_pos:
                            pos_X.append(feat)
                        else:
                            neg_X.append(feat)

    rng.shuffle(neg_X)
    keep_neg = neg_X[: NEG_RATIO * len(pos_X)]
    X = np.array(pos_X + keep_neg, dtype=np.float64)
    y = np.array([1.0] * len(pos_X) + [0.0] * len(keep_neg), dtype=np.float64)
    idx = np.array(range(len(X)))
    np.random.default_rng(SEED).shuffle(idx)
    X, y = X[idx], y[idx]

    os.makedirs("analysis", exist_ok=True)
    np.savez(OUT, X=X, y=y, feature_names=np.array(FEATURE_NAMES))
    print(f"episodes used={eps} skipped={skipped}")
    print(f"positives={len(pos_X)} negatives kept={len(keep_neg)} "
          f"(of {len(neg_X)} presampled @ {NEG_PRESAMPLE})")
    print(f"dataset: X={X.shape} y={y.shape} positive_rate={y.mean():.3f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
