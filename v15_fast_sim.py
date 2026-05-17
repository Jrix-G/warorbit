"""v15_fast_sim — vectorized-numpy Orbit Wars step engine for MCTS rollouts.

Pure numpy, no compiled dependency (Kaggle's agent runtime has no numba).
Mirrors kaggle_environments.envs.orbit_wars.orbit_wars.interpreter, except it
does NOT spawn new extra-solar comets (rare RNG-heavy path). Existing comets
already in the state ARE advanced along their paths.

Validated bit-for-bit against OfficialFastGame by tests/test_fast_sim_equivalence.

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
            # p_init / p_comet are immutable per game (only read in step, and
            # _drop_planets REASSIGNS them with a fresh array) -> share by ref.
            p_init=self.p_init,
            p_comet=self.p_comet,
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


def _seg_point_dist(px, py, ax, ay, bx, by):
    """Vectorized point-to-segment distance (broadcasts)."""
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * dx + (py - ay) * dy) / l2
    t = np.where(l2 == 0.0, 0.0, t)
    t = np.clip(t, 0.0, 1.0)
    projx = ax + t * dx
    projy = ay + t * dy
    return np.hypot(px - projx, py - projy)


def step(state: FastState, actions: list[list]) -> FastState:
    """Advance one turn. actions[player] = list of [from_id, angle, ships].
    Returns a NEW FastState (input is not mutated)."""
    s = state.copy()
    if s.done:
        return s

    _expire_comets(s)
    planets = s.planets

    # --- 0. Fleet launch ---
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
            if idx is None or planets[idx, OWNER] != player_id:
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

    # --- 1. Production ---
    owned = planets[:, OWNER] != -1
    planets[owned, SHIPS] += planets[owned, PROD]

    # --- 2. Fleet movement + continuous collision ---
    M = len(fleets)
    N = len(planets)
    planet_ids = (planets[:, ID].astype(np.int64) if N
                  else np.zeros(0, dtype=np.int64))
    caught_pid = np.full(M, -1, dtype=np.int64)
    removed = np.zeros(M, dtype=bool)

    if M:
        old_x = fleets[:, F_X].copy()
        old_y = fleets[:, F_Y].copy()
        sh = fleets[:, F_SHIPS]
        with np.errstate(invalid="ignore", divide="ignore"):
            speeds = 1.0 + (s.ship_speed - 1.0) * (np.log(sh) / _LOG1000) ** 1.5
        speeds = np.minimum(speeds, s.ship_speed)
        new_x = old_x + np.cos(fleets[:, F_ANGLE]) * speeds
        new_y = old_y + np.sin(fleets[:, F_ANGLE]) * speeds
        fleets[:, F_X] = new_x
        fleets[:, F_Y] = new_y

        oob = ~((new_x >= 0) & (new_x <= BOARD_SIZE) &
                (new_y >= 0) & (new_y <= BOARD_SIZE))
        sun = _seg_point_dist(np.full(M, CENTER), np.full(M, CENTER),
                              old_x, old_y, new_x, new_y) < SUN_RADIUS
        removed = oob | sun

        if N:
            D = _seg_point_dist(planets[:, X][None, :], planets[:, Y][None, :],
                                old_x[:, None], old_y[:, None],
                                new_x[:, None], new_y[:, None])
            hitM = D < planets[:, R][None, :]
            mhit = hitM.any(axis=1) & ~removed
            first = np.argmax(hitM, axis=1)
            caught_pid[mhit] = planet_ids[first[mhit]]
            removed = removed | mhit

    # --- 3. Planet rotation + sweep ---
    if N:
        dx = s.p_init[:, 0] - CENTER
        dy = s.p_init[:, 1] - CENTER
        orb_r = np.hypot(dx, dy)
        rotating = (~s.p_comet) & (orb_r + planets[:, R] < ROTATION_RADIUS_LIMIT)
        old_px = planets[:, X].copy()
        old_py = planets[:, Y].copy()
        cur_angle = np.arctan2(dy, dx) + s.angular_velocity * s.step
        planets[rotating, X] = CENTER + orb_r[rotating] * np.cos(cur_angle[rotating])
        planets[rotating, Y] = CENTER + orb_r[rotating] * np.sin(cur_angle[rotating])

        moved = (old_px != planets[:, X]) | (old_py != planets[:, Y])
        if M and moved.any():
            DS = _seg_point_dist(fleets[:, F_X][None, :], fleets[:, F_Y][None, :],
                                 old_px[:, None], old_py[:, None],
                                 planets[:, X][:, None], planets[:, Y][:, None])
            hitS = (DS < planets[:, R][:, None]) & moved[:, None]
            shit = hitS.any(axis=0) & ~removed
            firstS = np.argmax(hitS, axis=0)
            caught_pid[shit] = planet_ids[firstS[shit]]
            removed = removed | shit

    # --- Comet movement along precomputed paths + sweep ---
    expired_pids = []
    for grp in s.comets:
        grp["path_index"] += 1
        idx = grp["path_index"]
        for ci, pid in enumerate(grp["planet_ids"]):
            j = id_to_idx.get(int(pid))
            if j is None:
                continue
            p_path = grp["paths"][ci]
            if idx >= len(p_path):
                expired_pids.append(pid)
            else:
                old_cx, old_cy = planets[j, X], planets[j, Y]
                planets[j, X] = p_path[idx][0]
                planets[j, Y] = p_path[idx][1]
                if old_cx >= 0 and M:
                    dvec = _seg_point_dist(fleets[:, F_X], fleets[:, F_Y],
                                           old_cx, old_cy,
                                           planets[j, X], planets[j, Y])
                    chit = (dvec < planets[j, R]) & ~removed
                    caught_pid[chit] = int(pid)
                    removed = removed | chit

    if expired_pids:
        _drop_planets(s, set(expired_pids))
        planets = s.planets

    s.fleets = fleets[~removed]

    # --- 4. Combat resolution ---
    contested = caught_pid[caught_pid >= 0]
    id_to_idx = {int(planets[i, ID]): i for i in range(len(planets))}
    for pid in np.unique(contested):
        j = id_to_idx.get(int(pid))
        if j is None:
            continue
        mask = caught_pid == pid
        player_ships: dict[int, int] = {}
        for owner, shp in zip(fleets[mask, F_OWNER], fleets[mask, F_SHIPS]):
            o = int(owner)
            player_ships[o] = player_ships.get(o, 0) + int(shp)
        if not player_ships:
            continue
        ranked = sorted(player_ships.items(), key=lambda kv: kv[1], reverse=True)
        top_player, top_ships = ranked[0]
        if len(ranked) > 1:
            second = ranked[1][1]
            survivor_ships = top_ships - second
            if ranked[0][1] == ranked[1][1]:
                survivor_ships = 0
            survivor_owner = top_player if survivor_ships > 0 else -1
        else:
            survivor_owner = top_player
            survivor_ships = top_ships
        if survivor_ships > 0:
            if int(planets[j, OWNER]) == survivor_owner:
                planets[j, SHIPS] += survivor_ships
            else:
                planets[j, SHIPS] -= survivor_ships
                if planets[j, SHIPS] < 0:
                    planets[j, OWNER] = survivor_owner
                    planets[j, SHIPS] = abs(planets[j, SHIPS])

    # --- Terminal ---
    terminated = s.step >= s.episode_steps - 2
    alive = set()
    for i in range(len(planets)):
        if planets[i, OWNER] != -1:
            alive.add(int(planets[i, OWNER]))
    for i in range(len(s.fleets)):
        alive.add(int(s.fleets[i, F_OWNER]))
    if len(alive) <= 1:
        terminated = True
    s.done = bool(terminated)
    s.step += 1
    return s


def _expire_comets(s: FastState) -> None:
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
