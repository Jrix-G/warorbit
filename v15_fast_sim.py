"""v15_fast_sim — Numba-JIT Orbit Wars step engine for MCTS rollouts.

Mirrors kaggle_environments.envs.orbit_wars.orbit_wars.interpreter, except it
does NOT spawn new extra-solar comets (generate_comet_paths is the RNG-heavy
rare path). Existing comets already in the state ARE advanced along their paths.

The numeric step (`_core`) is JIT-compiled; comet bookkeeping stays in Python.
Validated bit-for-bit by tests/test_fast_sim_equivalence.py.

State (FastState):
  planets : float64 [N,7]  = id, owner, x, y, radius, ships, production
  p_init  : float64 [N,2]  = initial x, y (for rotation)
  p_comet : bool    [N]    = is this planet a comet body
  fleets  : float64 [M,7]  = id, owner, x, y, angle, from_id, ships
  comets  : list of dict {planet_ids, paths, path_index}
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)

ID, OWNER, X, Y, R, SHIPS, PROD = range(7)
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = range(7)

_LOG1000 = math.log(1000.0)


@dataclass
class FastState:
    planets: np.ndarray
    p_init: np.ndarray
    p_comet: np.ndarray
    fleets: np.ndarray
    comets: list
    step: int
    angular_velocity: float
    next_fleet_id: int
    episode_steps: int
    ship_speed: float
    n_players: int
    done: bool = False

    def copy(self) -> "FastState":
        return FastState(
            planets=self.planets.copy(),
            p_init=self.p_init.copy(),
            p_comet=self.p_comet.copy(),
            fleets=self.fleets.copy(),
            comets=[{"planet_ids": list(g["planet_ids"]),
                     "paths": g["paths"],
                     "path_index": g["path_index"]} for g in self.comets],
            step=self.step,
            angular_velocity=self.angular_velocity,
            next_fleet_id=self.next_fleet_id,
            episode_steps=self.episode_steps,
            ship_speed=self.ship_speed,
            n_players=self.n_players,
            done=self.done,
        )


def from_obs(obs, *, n_players: int, episode_steps: int = 500,
             ship_speed: float = 6.0) -> FastState:
    def g(key, default):
        if isinstance(obs, dict):
            return obs.get(key, default)
        return getattr(obs, key, default)

    planets_raw = list(g("planets", []) or [])
    initial_raw = list(g("initial_planets", []) or [])
    fleets_raw = list(g("fleets", []) or [])
    comets_raw = list(g("comets", []) or [])
    comet_pids = set(g("comet_planet_ids", []) or [])

    N = len(planets_raw)
    planets = np.zeros((N, 7), dtype=np.float64)
    for i, p in enumerate(planets_raw):
        planets[i] = p
    init_by_id = {int(p[0]): (p[2], p[3]) for p in initial_raw}
    p_init = np.zeros((N, 2), dtype=np.float64)
    for i, p in enumerate(planets_raw):
        p_init[i] = init_by_id.get(int(p[0]), (p[2], p[3]))
    p_comet = np.array([int(p[0]) in comet_pids for p in planets_raw], dtype=np.bool_)

    M = len(fleets_raw)
    fleets = np.zeros((M, 7), dtype=np.float64)
    for i, f in enumerate(fleets_raw):
        fleets[i] = f

    comets = [{"planet_ids": list(grp["planet_ids"]),
               "paths": grp["paths"],
               "path_index": grp["path_index"]} for grp in comets_raw]

    return FastState(
        planets=planets, p_init=p_init, p_comet=p_comet, fleets=fleets,
        comets=comets,
        step=int(g("step", 0) or 0),
        angular_velocity=float(g("angular_velocity", 0.0) or 0.0),
        next_fleet_id=int(g("next_fleet_id", 0) or 0),
        episode_steps=int(episode_steps),
        ship_speed=float(ship_speed),
        n_players=int(n_players),
    )


@njit(cache=True, fastmath=False)
def _seg_pt(px, py, ax, ay, bx, by):
    """Point-to-segment distance (matches orbit_wars.point_to_segment_distance)."""
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        ex = px - ax
        ey = py - ay
        return math.sqrt(ex * ex + ey * ey)
    t = ((px - ax) * dx + (py - ay) * dy) / l2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    prx = ax + t * dx
    pry = ay + t * dy
    ex = px - prx
    ey = py - pry
    return math.sqrt(ex * ex + ey * ey)


@njit(cache=True, fastmath=False)
def _core(planets, p_init, p_comet, fleets, comet_move, expiring,
          ang_vel, step, ship_speed, n_players):
    """Numeric per-turn step. Mutates planets & fleets in place.

    Returns (removed[M] bool, done bool). Combat is resolved here; comet
    spawn/expiry bookkeeping is handled by the Python wrapper.
    """
    N = planets.shape[0]
    M = fleets.shape[0]
    removed = np.zeros(M, dtype=np.bool_)
    caught = np.full(M, -1, dtype=np.int64)

    # 1. Production
    for j in range(N):
        if planets[j, 1] != -1.0:
            planets[j, 5] += planets[j, 6]

    # 2. Fleet movement + continuous collision
    for i in range(M):
        ox = fleets[i, 2]
        oy = fleets[i, 3]
        sh = fleets[i, 6]
        speed = 1.0 + (ship_speed - 1.0) * (math.log(sh) / _LOG1000) ** 1.5
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
        if _seg_pt(50.0, 50.0, ox, oy, nx, ny) < SUN_RADIUS:
            removed[i] = True
            continue
        for j in range(N):
            if _seg_pt(planets[j, 2], planets[j, 3], ox, oy, nx, ny) < planets[j, 4]:
                caught[i] = j
                removed[i] = True
                break

    # 3. Planet rotation + sweep
    for j in range(N):
        if p_comet[j]:
            continue
        dx = p_init[j, 0] - CENTER
        dy = p_init[j, 1] - CENTER
        r = math.sqrt(dx * dx + dy * dy)
        opx = planets[j, 2]
        opy = planets[j, 3]
        if r + planets[j, 4] < ROTATION_RADIUS_LIMIT:
            ia = math.atan2(dy, dx)
            ca = ia + ang_vel * step
            planets[j, 2] = CENTER + r * math.cos(ca)
            planets[j, 3] = CENTER + r * math.sin(ca)
        npx = planets[j, 2]
        npy = planets[j, 3]
        if opx == npx and opy == npy:
            continue
        for i in range(M):
            if removed[i]:
                continue
            if _seg_pt(fleets[i, 2], fleets[i, 3], opx, opy, npx, npy) < planets[j, 4]:
                caught[i] = j
                removed[i] = True

    # 4. Comet movement + sweep (positions precomputed by the wrapper)
    for j in range(N):
        if math.isnan(comet_move[j, 0]):
            continue
        ocx = planets[j, 2]
        ocy = planets[j, 3]
        planets[j, 2] = comet_move[j, 0]
        planets[j, 3] = comet_move[j, 1]
        if ocx >= 0.0:
            for i in range(M):
                if removed[i]:
                    continue
                if _seg_pt(fleets[i, 2], fleets[i, 3], ocx, ocy,
                           planets[j, 2], planets[j, 3]) < planets[j, 4]:
                    caught[i] = j
                    removed[i] = True

    # 5. Combat resolution
    acc = np.zeros((N, n_players), dtype=np.float64)
    has = np.zeros(N, dtype=np.bool_)
    for i in range(M):
        j = caught[i]
        if j >= 0:
            o = int(fleets[i, 1])
            if 0 <= o < n_players:
                acc[j, o] += fleets[i, 6]
                has[j] = True
    for j in range(N):
        if not has[j] or expiring[j]:
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

    # 6. Terminal
    done = step >= 0  # placeholder, overwritten below
    alive = np.zeros(n_players, dtype=np.bool_)
    for j in range(N):
        o = int(planets[j, 1])
        if 0 <= o < n_players:
            alive[o] = True
    for i in range(M):
        if not removed[i]:
            o = int(fleets[i, 1])
            if 0 <= o < n_players:
                alive[o] = True
    cnt = 0
    for o in range(n_players):
        if alive[o]:
            cnt += 1
    return removed, cnt


def step(state: FastState, actions: list[list]) -> FastState:
    """Advance one turn. actions[player] = list of [from_id, angle, ships].

    Returns a NEW FastState (input is not mutated).
    """
    s = state.copy()
    if s.done:
        return s

    # --- Comet expiry BEFORE launch ---
    _expire_comets(s)
    planets = s.planets

    # --- 0. Fleet launch (Python: depends on the dynamic actions list) ---
    id_to_idx = {int(planets[i, ID]): i for i in range(len(planets))}
    new_fleets = []
    for player_id in range(s.n_players):
        action = actions[player_id] if player_id < len(actions) else None
        if not action or not isinstance(action, list):
            continue
        for move in action:
            if len(move) != 3:
                continue
            from_id, angle, ships = move
            ships = int(ships)
            idx = id_to_idx.get(int(from_id))
            if idx is None:
                continue
            if planets[idx, OWNER] != player_id:
                continue
            if planets[idx, SHIPS] >= ships and ships > 0:
                planets[idx, SHIPS] -= ships
                sx = planets[idx, X] + math.cos(angle) * (planets[idx, R] + 0.1)
                sy = planets[idx, Y] + math.sin(angle) * (planets[idx, R] + 0.1)
                new_fleets.append([float(s.next_fleet_id), float(player_id),
                                   sx, sy, float(angle), float(from_id),
                                   float(ships)])
                s.next_fleet_id += 1

    if new_fleets:
        nf = np.array(new_fleets, dtype=np.float64)
        fleets = np.vstack([s.fleets, nf]) if len(s.fleets) else nf
    else:
        fleets = np.ascontiguousarray(s.fleets)

    # --- Comet movement targets + expiry mask (Python bookkeeping) ---
    N = len(planets)
    comet_move = np.full((N, 2), np.nan, dtype=np.float64)
    expiring = np.zeros(N, dtype=np.bool_)
    expired_pids = []
    for grp in s.comets:                       # empty in the common rollout case
        grp["path_index"] += 1
        idx = grp["path_index"]
        for ci, pid in enumerate(grp["planet_ids"]):
            j = id_to_idx.get(int(pid))
            if j is None:
                continue
            p_path = grp["paths"][ci]
            if idx >= len(p_path):
                expiring[j] = True
                expired_pids.append(pid)
            else:
                comet_move[j, 0] = p_path[idx][0]
                comet_move[j, 1] = p_path[idx][1]

    # --- Numeric core (JIT). planets/p_init/p_comet are already contiguous
    #     (fresh from FastState.copy()), so no ascontiguousarray needed. ---
    removed, alive_cnt = _core(
        planets, s.p_init, s.p_comet, fleets, comet_move, expiring,
        s.angular_velocity, float(s.step), s.ship_speed, s.n_players,
    )
    s.fleets = fleets[~removed]

    # --- Drop expired comet planets ---
    if expired_pids:
        _drop_planets(s, set(expired_pids))

    # --- Terminal ---
    terminated = s.step >= s.episode_steps - 2 or alive_cnt <= 1
    s.done = bool(terminated)
    s.step += 1
    return s


def _expire_comets(s: FastState) -> None:
    """Remove comet bodies whose path_index has already run off the end."""
    expired = []
    for grp in s.comets:
        idx = grp["path_index"]
        for ci, pid in enumerate(grp["planet_ids"]):
            if idx >= len(grp["paths"][ci]):
                expired.append(pid)
    if expired:
        _drop_planets(s, set(expired))


def _drop_planets(s: FastState, expired: set) -> None:
    keep = np.array([int(s.planets[i, ID]) not in expired
                     for i in range(len(s.planets))], dtype=np.bool_)
    s.planets = s.planets[keep]
    s.p_init = s.p_init[keep]
    s.p_comet = s.p_comet[keep]
    for grp in s.comets:
        grp["planet_ids"] = [p for p in grp["planet_ids"] if p not in expired]
    s.comets = [g for g in s.comets if g["planet_ids"]]


def scores(s: FastState) -> list[int]:
    out = [0] * s.n_players
    for i in range(len(s.planets)):
        o = int(s.planets[i, OWNER])
        if 0 <= o < s.n_players:
            out[o] += int(s.planets[i, SHIPS])
    for i in range(len(s.fleets)):
        o = int(s.fleets[i, F_OWNER])
        if 0 <= o < s.n_players:
            out[o] += int(s.fleets[i, F_SHIPS])
    return out
