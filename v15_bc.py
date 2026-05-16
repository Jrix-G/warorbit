"""v15_bc — behavioral-cloning policy (BC-Policy) for the RCC continuation.

A logistic model fitted on 2.5M (source, target) decisions from 2630 top-10
replays (train_bc_policy.py, val AUC 0.778). It answers, for any candidate
launch: P(a top-10 bot launches src->tgt in this state).

It is used as the DETERMINISTIC continuation policy inside the RCC combo
evaluation: after a combo is applied, both sides continue by launching only
their highest-confidence move per planet (P above LAUNCH_THRESHOLD). Unlike a
fully passive continuation this models opponent reinforcement / counter-play,
so combos that only look good against a do-nothing opponent are correctly
penalised. It stays deterministic — no RNG — so combo comparisons keep zero
variance.

Weights are embedded so the module is self-contained for Kaggle packaging.
"""

from __future__ import annotations

import math

import numpy as np

import v15_fast_sim as fsim

ID, OWNER, X, Y, R, SHIPS, PROD = range(7)
F_OWNER, F_SHIPS = fsim.F_OWNER, fsim.F_SHIPS

BOARD_DIAG = 141.42
LAUNCH_THRESHOLD = 0.5      # min P(launch) for the continuation to act
_DET_MARGIN = 2.0           # min source garrison to consider launching

# fitted weights (train_bc_policy.py, val AUC 0.778) — feature order:
# src_ship_share, tgt_prod_share, dist_norm, tgt_is_enemy, tgt_is_neutral,
# combat_margin, step_frac, my_ship_share, my_prod_share, tgt_defense_ratio,
# src_prod_norm, is_four_player
_W = np.array([-1.0108, 0.2763, -0.6452, 0.2163, 0.0953, -0.0625,
               -0.3072, 0.5150, -0.4708, 0.1650, 0.5178, 0.0000])
_B = -1.7701757603993615
_MEAN = np.array([0.0248, 0.0393, 0.3973, 0.4487, 0.1219, -0.0413,
                  0.3587, 0.4839, 0.4935, 3.6383, 0.2699, 1.0000])
_STD = np.array([0.0270, 0.0427, 0.1907, 0.4974, 0.3272, 0.5032,
                 0.2378, 0.2847, 0.2798, 29.1325, 0.1449, 1.0000])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def launch_probs(fs: fsim.FastState, player: int):
    """P(launch src->tgt) for every (owned src, other tgt) pair.

    Returns (src_idx, tgt_idx, prob) arrays — flat over all candidate pairs."""
    planets = fs.planets
    N = len(planets)
    if N == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)

    owners = planets[:, OWNER].astype(np.int64)
    ships = planets[:, SHIPS]
    prod = planets[:, PROD]
    px = planets[:, X]
    py = planets[:, Y]

    tot_s = float(ships.sum())
    # in-flight fleets count toward player ship totals
    fleet_s = np.zeros(fs.n_players)
    if len(fs.fleets):
        fo = fs.fleets[:, F_OWNER].astype(np.int64)
        for p in range(fs.n_players):
            m = fo == p
            if m.any():
                fleet_s[p] += fs.fleets[m, F_SHIPS].sum()
    tot_s += float(fleet_s.sum())
    tot_p = float(prod[owners >= 0].sum()) or 1.0
    tot_s = tot_s or 1.0

    my_ships = float(ships[owners == player].sum()) + fleet_s[player]
    my_prod = float(prod[owners == player].sum())
    step_frac = min(fs.step / 500.0, 1.0)
    is_4p = 1.0 if fs.n_players >= 4 else 0.0

    mine = np.where(owners == player)[0]
    others = np.where(np.arange(N) != -1)[0]  # all planets
    src_list, tgt_list, feat_rows = [], [], []
    for i in mine:
        if ships[i] < _DET_MARGIN:
            continue
        s_ships = float(ships[i])
        for j in range(N):
            if j == i:
                continue
            t_ships = float(ships[j])
            t_owner = int(owners[j])
            dist = math.hypot(px[j] - px[i], py[j] - py[i])
            feat_rows.append([
                s_ships / tot_s,
                float(prod[j]) / tot_p,
                dist / BOARD_DIAG,
                1.0 if (t_owner >= 0 and t_owner != player) else 0.0,
                1.0 if t_owner == -1 else 0.0,
                (s_ships - t_ships) / (s_ships + t_ships + 1.0),
                step_frac,
                my_ships / tot_s,
                my_prod / tot_p,
                t_ships / (s_ships + 1.0),
                min(float(prod[i]) / 10.0, 1.0),
                is_4p,
            ])
            src_list.append(i)
            tgt_list.append(j)

    if not feat_rows:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    F = np.array(feat_rows)
    Z = (F - _MEAN) / _STD
    probs = _sigmoid(Z @ _W + _B)
    return np.array(src_list), np.array(tgt_list), probs


def bc_policy(fs: fsim.FastState) -> list[list]:
    """Deterministic top-10-style continuation: each player launches its
    single highest-probability move per source planet, if P >= threshold.
    Ship amount = a capture-sized fraction of the garrison."""
    actions: list[list] = [[] for _ in range(fs.n_players)]
    planets = fs.planets
    N = len(planets)
    if N == 0:
        return actions
    owners = planets[:, OWNER].astype(np.int64)
    for player in range(fs.n_players):
        src, tgt, probs = launch_probs(fs, player)
        if len(src) == 0:
            continue
        best_by_src: dict[int, tuple] = {}
        for k in range(len(src)):
            i = int(src[k])
            cur = best_by_src.get(i)
            if cur is None or probs[k] > cur[1]:
                best_by_src[i] = (int(tgt[k]), float(probs[k]))
        for i, (j, p) in best_by_src.items():
            if p < LAUNCH_THRESHOLD:
                continue
            ang = math.atan2(planets[j, Y] - planets[i, Y],
                             planets[j, X] - planets[i, X])
            # capture-sized: defenders + small margin, capped at garrison
            need = int(planets[j, SHIPS] + 5)
            send = min(int(planets[i, SHIPS]), max(need, 1))
            if send > 0:
                actions[player].append([int(planets[i, ID]), float(ang), send])
    return actions
