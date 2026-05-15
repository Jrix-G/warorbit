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
from numba import njit

import bot_v7
import v15_fast_sim as fsim
from v15_fast_sim import _seg_pt

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


@njit(cache=True, fastmath=False)
def _rollout_njit(planets0, p_init, p_comet, fleets0, ang_vel, step0,
                  ship_speed, n_players, horizon, seed):
    """Fully-compiled rollout: H turns of (fast policy -> engine), return scores.

    Comets are frozen (no comet movement) — a rollout-only approximation."""
    np.random.seed(seed)
    N = planets0.shape[0]
    planets = planets0.copy()
    cap = N * (horizon + 2) + fleets0.shape[0] + 16
    fleets = np.zeros((cap, 7), dtype=np.float64)
    n_f = fleets0.shape[0]
    for i in range(n_f):
        for c in range(7):
            fleets[i, c] = fleets0[i, c]
    next_fid = 0.0
    step = step0
    log1000 = math.log(1000.0)

    for _h in range(horizon):
        # --- policy: each strong planet launches at a nearby foreign planet ---
        for j in range(N):
            p = int(planets[j, 1])
            if p < 0 or p >= n_players:
                continue
            sh = planets[j, 5]
            if sh <= _POLICY_MARGIN:
                continue
            best = -1
            best_d = 1e18
            best2 = -1
            best2_d = 1e18
            for t in range(N):
                if int(planets[t, 1]) == p:
                    continue
                ddx = planets[t, 2] - planets[j, 2]
                ddy = planets[t, 3] - planets[j, 3]
                d = ddx * ddx + ddy * ddy
                if d < best_d:
                    best2 = best
                    best2_d = best_d
                    best = t
                    best_d = d
                elif d < best2_d:
                    best2 = t
                    best2_d = d
            if best < 0:
                continue
            tgt = best
            if best2 >= 0 and np.random.random() < 0.4:
                tgt = best2
            ang = math.atan2(planets[tgt, 3] - planets[j, 3],
                             planets[tgt, 2] - planets[j, 2])
            frac = _POLICY_SEND_LO + (_POLICY_SEND_HI - _POLICY_SEND_LO) * np.random.random()
            send = float(int(sh * frac))
            if send <= 0.0:
                continue
            planets[j, 5] -= send
            sx = planets[j, 2] + math.cos(ang) * (planets[j, 4] + 0.1)
            sy = planets[j, 3] + math.sin(ang) * (planets[j, 4] + 0.1)
            if n_f < cap:
                fleets[n_f, 0] = next_fid
                fleets[n_f, 1] = float(p)
                fleets[n_f, 2] = sx
                fleets[n_f, 3] = sy
                fleets[n_f, 4] = ang
                fleets[n_f, 5] = planets[j, 0]
                fleets[n_f, 6] = send
                n_f += 1
                next_fid += 1.0

        # --- production ---
        for j in range(N):
            if planets[j, 1] != -1.0:
                planets[j, 5] += planets[j, 6]

        # --- movement + collision ---
        removed = np.zeros(n_f, dtype=np.bool_)
        caught = np.full(n_f, -1, dtype=np.int64)
        for i in range(n_f):
            ox = fleets[i, 2]
            oy = fleets[i, 3]
            sh = fleets[i, 6]
            speed = 1.0 + (ship_speed - 1.0) * (math.log(sh) / log1000) ** 1.5
            if speed > ship_speed:
                speed = ship_speed
            ang = fleets[i, 4]
            nx = ox + math.cos(ang) * speed
            ny = oy + math.sin(ang) * speed
            fleets[i, 2] = nx
            fleets[i, 3] = ny
            if not (0.0 <= nx <= 100.0 and 0.0 <= ny <= 100.0):
                removed[i] = True
                continue
            if _seg_pt(50.0, 50.0, ox, oy, nx, ny) < 10.0:
                removed[i] = True
                continue
            for j in range(N):
                if _seg_pt(planets[j, 2], planets[j, 3], ox, oy, nx, ny) < planets[j, 4]:
                    caught[i] = j
                    removed[i] = True
                    break

        # --- rotation + sweep (comets frozen) ---
        for j in range(N):
            if p_comet[j]:
                continue
            dx = p_init[j, 0] - 50.0
            dy = p_init[j, 1] - 50.0
            r = math.sqrt(dx * dx + dy * dy)
            opx = planets[j, 2]
            opy = planets[j, 3]
            if r + planets[j, 4] < 50.0:
                ca = math.atan2(dy, dx) + ang_vel * step
                planets[j, 2] = 50.0 + r * math.cos(ca)
                planets[j, 3] = 50.0 + r * math.sin(ca)
            npx = planets[j, 2]
            npy = planets[j, 3]
            if opx == npx and opy == npy:
                continue
            for i in range(n_f):
                if removed[i]:
                    continue
                if _seg_pt(fleets[i, 2], fleets[i, 3], opx, opy, npx, npy) < planets[j, 4]:
                    caught[i] = j
                    removed[i] = True

        # --- combat ---
        acc = np.zeros((N, n_players), dtype=np.float64)
        has = np.zeros(N, dtype=np.bool_)
        for i in range(n_f):
            j = caught[i]
            if j >= 0:
                o = int(fleets[i, 1])
                if 0 <= o < n_players:
                    acc[j, o] += fleets[i, 6]
                    has[j] = True
        for j in range(N):
            if not has[j]:
                continue
            n_pos = 0
            top_o = -1
            top_s = 0.0
            sec_s = 0.0
            for o in range(n_players):
                v = acc[j, o]
                if v > 0.0:
                    n_pos += 1
                if v > top_s:
                    sec_s = top_s
                    top_s = v
                    top_o = o
                elif v > sec_s:
                    sec_s = v
            if n_pos >= 2:
                surv_s = top_s - sec_s
                if top_s == sec_s:
                    surv_s = 0.0
                surv_o = top_o if surv_s > 0.0 else -1
            else:
                surv_o = top_o
                surv_s = top_s
            if surv_s > 0.0:
                if int(planets[j, 1]) == surv_o:
                    planets[j, 5] += surv_s
                else:
                    planets[j, 5] -= surv_s
                    if planets[j, 5] < 0.0:
                        planets[j, 1] = surv_o
                        planets[j, 5] = -planets[j, 5]

        # --- compact surviving fleets ---
        w = 0
        for i in range(n_f):
            if not removed[i]:
                if w != i:
                    for c in range(7):
                        fleets[w, c] = fleets[i, c]
                w += 1
        n_f = w
        step += 1

        # --- terminal ---
        seen = np.zeros(n_players, dtype=np.bool_)
        alive = 0
        for j in range(N):
            o = int(planets[j, 1])
            if 0 <= o < n_players and not seen[o]:
                seen[o] = True
                alive += 1
        for i in range(n_f):
            o = int(fleets[i, 1])
            if 0 <= o < n_players and not seen[o]:
                seen[o] = True
                alive += 1
        if alive <= 1:
            break

    sc = np.zeros(n_players, dtype=np.float64)
    for j in range(N):
        o = int(planets[j, 1])
        if 0 <= o < n_players:
            sc[o] += planets[j, 5]
    for i in range(n_f):
        o = int(fleets[i, 1])
        if 0 <= o < n_players:
            sc[o] += fleets[i, 6]
    return sc


def _mc_value(fs: fsim.FastState, our_player: int, n_rollouts: int,
              horizon: int, base_seed: int) -> float:
    """Average our ship-share over n_rollouts compiled rollouts."""
    planets = np.ascontiguousarray(fs.planets)
    p_init = np.ascontiguousarray(fs.p_init)
    p_comet = np.ascontiguousarray(fs.p_comet)
    fleets = np.ascontiguousarray(fs.fleets)
    acc = 0.0
    for r in range(n_rollouts):
        sc = _rollout_njit(planets, p_init, p_comet, fleets,
                           fs.angular_velocity, float(fs.step), fs.ship_speed,
                           fs.n_players, horizon, base_seed + r)
        total = sc.sum()
        if total > 0:
            acc += sc[our_player] / total
    return acc / max(1, n_rollouts)


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
           n_rollouts: int = 280, horizon: int = 30,
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

        rng = np.random.default_rng(seed + fs.step)

        # Candidates: V7-move subsets (commit-level decisions) + a few fast-policy
        # samples (genuinely different targets the search can prefer over V7).
        cands = _candidates(v7_move)
        for _ in range(3):
            sample = _fast_policy(fs, rng)[our_player]
            if sample and repr(sample) not in [repr(c) for c in cands]:
                cands.append(sample)
        if len(cands) <= 1:
            return v7_move

        opp_template = _fast_policy(fs, rng)         # opponents' move this turn

        best_move = v7_move
        best_val = -1.0
        for ci, cand in enumerate(cands):
            if time.monotonic() - t0 > time_budget:
                break
            actions = [list(opp_template[p]) for p in range(n_players)]
            actions[our_player] = cand
            child = fsim.step(fs, actions)
            val = _mc_value(child, our_player, n_rollouts, horizon,
                            seed + fs.step * 1000 + ci * 100000)
            if val > best_val:
                best_val = val
                best_move = cand
        return best_move
    except Exception:
        try:
            return bot_v7.agent(obs, config)
        except Exception:
            return []
