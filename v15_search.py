"""v15_search — flat Monte-Carlo search on top of v15_fast_sim.

Design (P2 v1):
  * Candidates at the root are derived from V7's move (V7 is the proven base):
    V7's full move, the empty move, and prefixes of V7's plan. The search
    therefore decides *how much* of V7's plan to commit to.
  * Each candidate is evaluated by R Monte-Carlo rollouts. Rollouts use a fast
    stochastic heuristic policy for ALL players (NOT V7 — V7 is too slow to be
    a rollout policy; classic MCTS uses a cheap rollout policy).
  * Leaf value = our ship-share. Best average wins.

V7 is called exactly once per turn (to seed candidates), so the search keeps
V7's strategic priors while fixing its myopic over-/under-commitment mistakes.
"""

from __future__ import annotations

import math
import time

import numpy as np

import bot_v7
import v15_fast_sim as fsim

ID, OWNER, X, Y, R, SHIPS, PROD = range(7)

# Rollout policy tunables.
_POLICY_MARGIN = 20.0      # only launch from planets with more ships than this
_POLICY_SEND_LO = 0.5
_POLICY_SEND_HI = 0.9
_POLICY_NEAR_K = 3         # pick target randomly among the K nearest


def _infer_n_players(planets: np.ndarray) -> int:
    if len(planets) == 0:
        return 2
    return 4 if planets[:, OWNER].max() >= 2 else 2


def state_to_obs(fs: fsim.FastState, player: int) -> dict:
    """Build a Kaggle-style observation dict for V7."""
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
        "player": player,
        "step": fs.step,
        "angular_velocity": fs.angular_velocity,
        "planets": planets,
        "initial_planets": init,
        "fleets": fleets,
        "next_fleet_id": fs.next_fleet_id,
        "comets": fs.comets,
        "comet_planet_ids": comet_ids,
        "remainingOverageTime": 60.0,
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
    # pairwise distance matrix
    dxm = px[:, None] - px[None, :]
    dym = py[:, None] - py[None, :]
    dist = np.sqrt(dxm * dxm + dym * dym)
    for i in range(N):
        p = int(owners[i])
        if p < 0 or p >= fs.n_players:
            continue
        if ships[i] <= _POLICY_MARGIN:
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
              horizon: int, rng: np.random.Generator) -> float:
    return sum(_rollout(fs, our_player, horizon, rng)
               for _ in range(n_rollouts)) / max(1, n_rollouts)


def _candidates(v7_move: list) -> list[list]:
    """Candidate root moves: subsets of V7's plan (commit-level decisions)."""
    cands: list[list] = [[]]                       # do nothing
    if v7_move:
        ordered = list(v7_move)
        cands.append(ordered)                      # full V7 move
        for k in range(1, len(ordered)):           # prefixes
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
           n_rollouts: int = 12, horizon: int = 20,
           seed: int = 0) -> list:
    """Return the chosen move for the current player. Falls back to V7 on
    time pressure or error."""
    t0 = time.monotonic()
    try:
        if isinstance(obs, dict):
            our_player = int(obs.get("player", 0) or 0)
        else:
            our_player = int(getattr(obs, "player", 0) or 0)

        v7_move = bot_v7.agent(obs, config)
        if not isinstance(v7_move, list):
            v7_move = []

        fs = fsim.from_obs(obs, n_players=2)        # n_players fixed up below
        n_players = _infer_n_players(fs.planets)
        fs.n_players = n_players

        cands = _candidates(v7_move)
        if len(cands) <= 1:
            return v7_move

        rng = np.random.default_rng(seed + fs.step)
        opp_template = _fast_policy(fs, rng)         # opponents' move this turn

        best_move = v7_move
        best_val = -1.0
        for cand in cands:
            if time.monotonic() - t0 > time_budget:
                break
            actions = [list(opp_template[p]) for p in range(n_players)]
            actions[our_player] = cand
            child = fsim.step(fs, actions)
            val = _mc_value(child, our_player, n_rollouts, horizon, rng)
            if val > best_val:
                best_val = val
                best_move = cand
        return best_move
    except Exception:
        try:
            return bot_v7.agent(obs, config)
        except Exception:
            return []
