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
             rng: np.random.Generator) -> float:
    """Play `horizon` steps with the fast policy; return our ship-share [0,1]."""
    cur = fs
    for _ in range(horizon):
        if cur.done:
            break
        cur = fsim.step(cur, _fast_policy(cur, rng))
    sc = fsim.scores(cur)
    total = sum(sc)
    if total <= 0:
        return 0.0
    return sc[our_player] / total


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


def search(obs, config=None, *, time_budget: float = 0.8,
           n_rollouts: int = 120, horizon: int = 25,
           seed: int = 0) -> list:
    """Return the chosen move for the current player. Falls back to V7 on
    time pressure or error."""
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
        n_players = _infer_n_players(fs.planets)
        fs.n_players = n_players

        rng = np.random.default_rng(seed + fs.step)
        cands = _candidates(v7_move)
        for _ in range(3):
            sample = _fast_policy(fs, rng)[our_player]
            if sample and repr(sample) not in [repr(c) for c in cands]:
                cands.append(sample)
        if len(cands) <= 1:
            return v7_move

        opp_template = _fast_policy(fs, rng)
        # Split the time budget evenly across candidates.
        per_cand = time_budget / max(1, len(cands))

        best_move = v7_move
        best_val = -1.0
        for ci, cand in enumerate(cands):
            if time.monotonic() > deadline:
                break
            actions = [list(opp_template[p]) for p in range(n_players)]
            actions[our_player] = cand
            child = fsim.step(fs, actions)
            cand_deadline = min(deadline, time.monotonic() + per_cand)
            val = _mc_value(child, our_player, n_rollouts, horizon, rng,
                            deadline=cand_deadline)
            if val > best_val:
                best_val = val
                best_move = cand
        return best_move
    except Exception:
        try:
            return bot_v7.agent(obs, config)
        except Exception:
            return []
