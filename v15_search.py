"""v15_search — flat Monte-Carlo search on top of v15_fast_sim.

Pure numpy, no compiled dependency. Design:
  * Candidates at the root: V7's move, the empty move, prefixes of V7's plan,
    plus a few fast-policy samples (genuinely different targets).
  * Each candidate is scored by R Monte-Carlo rollouts. Rollouts use a fast
    stochastic heuristic policy for ALL players (NOT V7 — too slow as a
    rollout policy; classic MCTS uses a cheap rollout policy).
  * Leaf value = our ship-share. Best average wins.

V7 is called once per turn (to seed candidates) and is the fallback.
"""

from __future__ import annotations

import math
import time

import numpy as np

import bot_v7
import v15_fast_sim as fsim

ID, OWNER, X, Y, R, SHIPS, PROD = range(7)

_POLICY_MARGIN = 20.0
_POLICY_SEND_LO = 0.5
_POLICY_SEND_HI = 0.9
_POLICY_NEAR_K = 3

# 4p win-probability value function (logistic regression on top-10 replays;
# val AUC 0.956, calibrated). Used as the leaf evaluation in 4p games only;
# 2p keeps the plain ship-share leaf eval (which already works there).
# Features: ship_share, prod_share, planet_share, max_opp_ship_share,
# max_opp_prod_share, ship_margin, prod_margin, rank_norm, step_frac,
# alive_frac, eliminated.
_VF_W = np.array([1.248060, 1.668613, -0.380990, -0.760989, -0.738031,
                  1.076146, 1.287865, -0.096036, 0.048232, 0.189244,
                  0.335709])
_VF_B = -3.341067887385429
_VF_MEAN = np.array([0.249949, 0.249928, 0.249966, 0.482475, 0.484964,
                     -0.232526, -0.235036, 0.500264, 0.327224, 0.827213,
                     0.172802])
_VF_STD = np.array([0.240608, 0.241091, 0.236827, 0.253622, 0.250501,
                    0.458392, 0.455922, 0.372551, 0.245244, 0.202419,
                    0.378076])


def _leaf_value(fs: fsim.FastState, player: int, use_vf: bool = True) -> float:
    """Leaf evaluation. 4p: calibrated P(player finishes #1) via the value
    function. 2p (or use_vf=False): plain ship-share."""
    n = fs.n_players
    sc = fsim.scores(fs)
    total = sum(sc)
    if n != 4 or not use_vf:
        return sc[player] / total if total > 0 else 0.0

    ships = [0.0] * n
    prod = [0.0] * n
    planets = [0] * n
    for p in fs.planets:
        o = int(p[OWNER])
        if 0 <= o < n:
            ships[o] += p[SHIPS]
            prod[o] += p[PROD]
            planets[o] += 1
    for f in fs.fleets:
        o = int(f[1])
        if 0 <= o < n:
            ships[o] += f[6]
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
    feats = np.array([ss, ps, pls, max_os, max_op, ss - max_os, ps - max_op,
                      rank / 3.0, min(fs.step / 500.0, 1.0), alive / n, elim])
    z = float(((feats - _VF_MEAN) / _VF_STD) @ _VF_W + _VF_B)
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def _infer_n_players(planets: np.ndarray) -> int:
    if len(planets) == 0:
        return 2
    return 4 if planets[:, OWNER].max() >= 2 else 2


def state_to_obs(fs: fsim.FastState, player: int) -> dict:
    planets = [[int(p[ID]), int(p[OWNER]), float(p[X]), float(p[Y]),
                float(p[R]), float(p[SHIPS]), float(p[PROD])]
               for p in fs.planets]
    init = [[int(fs.planets[i][ID]), int(fs.planets[i][OWNER]),
             float(fs.p_init[i][0]), float(fs.p_init[i][1]),
             float(fs.planets[i][R]), float(fs.planets[i][SHIPS]),
             float(fs.planets[i][PROD])]
            for i in range(len(fs.planets))]
    fleets = [[int(f[0]), int(f[1]), float(f[2]), float(f[3]),
               float(f[4]), int(f[5]), float(f[6])] for f in fs.fleets]
    comet_ids = [pid for g in fs.comets for pid in g["planet_ids"]]
    return {
        "player": player, "step": fs.step,
        "angular_velocity": fs.angular_velocity,
        "planets": planets, "initial_planets": init, "fleets": fleets,
        "next_fleet_id": fs.next_fleet_id, "comets": fs.comets,
        "comet_planet_ids": comet_ids, "remainingOverageTime": 60.0,
    }


def _fast_policy(fs: fsim.FastState, rng: np.random.Generator) -> list[list]:
    """Cheap stochastic rollout policy: each strong planet sends ships toward a
    nearby non-owned planet. Returns one action list per player."""
    planets = fs.planets
    N = len(planets)
    actions: list[list] = [[] for _ in range(fs.n_players)]
    if N == 0:
        return actions
    px = planets[:, X]
    py = planets[:, Y]
    owners = planets[:, OWNER].astype(np.int64)
    ships = planets[:, SHIPS]
    ids = planets[:, ID].astype(np.int64)
    dxm = px[:, None] - px[None, :]
    dym = py[:, None] - py[None, :]
    dist = np.sqrt(dxm * dxm + dym * dym)
    for i in range(N):
        p = int(owners[i])
        if p < 0 or p >= fs.n_players or ships[i] <= _POLICY_MARGIN:
            continue
        foreign = np.where(owners != p)[0]
        if len(foreign) == 0:
            continue
        order = foreign[np.argsort(dist[i, foreign])]
        k = min(_POLICY_NEAR_K, len(order))
        tgt = int(order[rng.integers(0, k)])
        ang = math.atan2(py[tgt] - py[i], px[tgt] - px[i])
        frac = rng.uniform(_POLICY_SEND_LO, _POLICY_SEND_HI)
        send = int(ships[i] * frac)
        if send > 0:
            actions[p].append([int(ids[i]), float(ang), send])
    return actions


def _rollout(fs: fsim.FastState, our_player: int, horizon: int,
             rng: np.random.Generator, use_vf: bool = True) -> float:
    """Play `horizon` steps with the fast policy; return the leaf value."""
    cur = fs
    for _ in range(horizon):
        if cur.done:
            break
        cur = fsim.step(cur, _fast_policy(cur, rng))
    return _leaf_value(cur, our_player, use_vf)


def _mc_value(fs: fsim.FastState, our_player: int, n_rollouts: int,
              horizon: int, rng: np.random.Generator,
              deadline: float | None = None) -> float:
    acc = 0.0
    done = 0
    for _ in range(n_rollouts):
        if deadline is not None and time.monotonic() > deadline:
            break
        acc += _rollout(fs, our_player, horizon, rng)
        done += 1
    return acc / max(1, done)


def _candidates(v7_move: list) -> list[list]:
    cands: list[list] = [[]]
    if v7_move:
        ordered = list(v7_move)
        cands.append(ordered)
        for k in range(1, len(ordered)):
            cands.append(ordered[:k])
    seen = []
    uniq = []
    for c in cands:
        key = repr(c)
        if key not in seen:
            seen.append(key)
            uniq.append(c)
    return uniq


def search(obs, config=None, *, time_budget: float = 0.7,
           horizon: int = 18, n_policy_samples: int = 6,
           seed: int = 0, use_value_fn: bool = False) -> list:
    """V15.1-A — flat Monte-Carlo with sequential-halving budget allocation.

    Candidates are pruned in rounds: every survivor gets a slice of rollouts,
    the worst half is dropped, repeat. The best candidate ends up with most of
    the budget — far better best-arm identification than uniform allocation
    at the same total cost. Falls back to V7 on error / time pressure."""
    t0 = time.monotonic()
    deadline = t0 + time_budget
    try:
        if isinstance(obs, dict):
            our_player = int(obs.get("player", 0) or 0)
        else:
            our_player = int(getattr(obs, "player", 0) or 0)

        v7_move = bot_v7.agent(obs, config)
        if not isinstance(v7_move, list):
            v7_move = []

        fs = fsim.from_obs(obs, n_players=2)
        fs.n_players = _infer_n_players(fs.planets)
        n_players = fs.n_players

        rng = np.random.default_rng(seed + fs.step)
        cands = _candidates(v7_move)
        for _ in range(n_policy_samples):
            sample = _fast_policy(fs, rng)[our_player]
            if sample and repr(sample) not in [repr(c) for c in cands]:
                cands.append(sample)
        if len(cands) <= 1:
            return v7_move

        # Pre-compute each candidate's child state once.
        opp_template = _fast_policy(fs, rng)
        children = []
        for cand in cands:
            actions = [list(opp_template[p]) for p in range(n_players)]
            actions[our_player] = cand
            children.append(fsim.step(fs, actions))

        # --- Sequential halving ---
        K = len(cands)
        sums = [0.0] * K
        counts = [0] * K
        survivors = list(range(K))
        n_rounds = max(1, math.ceil(math.log2(K)))
        for r in range(n_rounds):
            if len(survivors) <= 1 or time.monotonic() > deadline:
                break
            round_deadline = t0 + time_budget * (r + 1) / n_rounds
            per_arm = max(0.0, (round_deadline - time.monotonic()) / len(survivors))
            for idx in survivors:
                arm_deadline = min(round_deadline, time.monotonic() + per_arm)
                while time.monotonic() < arm_deadline:
                    sums[idx] += _rollout(children[idx], our_player, horizon,
                                          rng, use_value_fn)
                    counts[idx] += 1
            survivors.sort(
                key=lambda i: (sums[i] / counts[i]) if counts[i] else -1.0,
                reverse=True,
            )
            survivors = survivors[: max(1, len(survivors) // 2)]

        best = max(range(K),
                   key=lambda i: (sums[i] / counts[i]) if counts[i] else -1.0)
        return cands[best] if counts[best] else v7_move
    except Exception:
        try:
            return bot_v7.agent(obs, config)
        except Exception:
            return []
