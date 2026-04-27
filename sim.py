"""Orbit Wars numpy simulator.

Faithful port of kaggle_environments orbit_wars.py engine. Used by V6 beam
search and self-play training. numpy + stdlib only.

Engine ordering per turn (matches Kaggle):
  1. Fleet launch from actions
  2. Production on owned planets
  3. Fleet movement + sun/planet collision + OOB
  4. Planet rotation + planet-sweep
  5. Comet movement + comet-sweep
  6. Combat resolution per planet (using fleet snapshot from step 3)
  7. Drop dead fleets / expired comets
  8. step += 1

Notes on faithfulness:
  - Comet spawning matches Kaggle when the caller preserves Python RNG state.
  - Existing comets continue along their pre-computed paths.
  - Fleet collision detection matches engine: continuous segment intersection
    with planet circles, sun, board edges.
  - Combat semantics match: per-planet, sum ships per player, top-second
    determines survivor; tied top → neutral with 0 ships.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np


# --- Physical constants (match kaggle_environments) -------------------------
BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0
COMET_SPEED = 4.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
TOTAL_TURNS = 500
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)

# Planet array column layout (float32):
#   0: id, 1: owner (-1 = neutral), 2: x, 3: y, 4: radius, 5: ships, 6: production
P_ID, P_OWNER, P_X, P_Y, P_R, P_SHIPS, P_PROD = range(7)

# Fleet array column layout (float32):
#   0: id, 1: owner, 2: x, 3: y, 4: angle, 5: from_id, 6: ships
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = range(7)


# --- State container --------------------------------------------------------

@dataclass
class GameState:
    """Compact numpy game state. All numeric arrays float32."""

    planets: np.ndarray              # (N_p, 7)
    fleets: np.ndarray               # (N_f, 7)
    init_positions: np.ndarray       # (N_p, 2) — initial x,y per row
    init_planet_ids: np.ndarray      # (N_p,)   — id of each row in init_positions
    angular_velocity: float
    comet_ids: Set[int]
    comet_groups: List[dict]
    step: int
    player: int
    next_fleet_id: int = 0

    def copy(self) -> "GameState":
        return GameState(
            planets=self.planets.copy() if len(self.planets) else self.planets.copy(),
            fleets=self.fleets.copy() if len(self.fleets) else np.zeros((0, 7), dtype=np.float32),
            init_positions=self.init_positions.copy(),
            init_planet_ids=self.init_planet_ids.copy(),
            angular_velocity=self.angular_velocity,
            comet_ids=set(self.comet_ids),
            comet_groups=[
                {"planet_ids": list(g["planet_ids"]),
                 "paths": g["paths"],  # paths are read-only references
                 "path_index": g["path_index"]}
                for g in self.comet_groups
            ],
            step=self.step,
            player=self.player,
            next_fleet_id=self.next_fleet_id,
        )


# --- Vectorized helpers -----------------------------------------------------

def fleet_speed(ships: float) -> float:
    """Scalar fleet speed (matches engine clamp to MAX_SPEED)."""
    if ships <= 1:
        return 1.0
    s = 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5
    return min(s, MAX_SPEED)


def fleet_speeds(ships_array: np.ndarray) -> np.ndarray:
    """Vectorized fleet speeds. Input shape (N,) → output (N,)."""
    speeds = np.ones_like(ships_array, dtype=np.float32)
    if len(speeds) == 0:
        return speeds
    mask = ships_array > 1
    if np.any(mask):
        log_ships = np.log(np.maximum(ships_array[mask], 1.0))
        speeds[mask] = 1.0 + (MAX_SPEED - 1.0) * (log_ships / math.log(1000.0)) ** 1.5
        np.minimum(speeds, MAX_SPEED, out=speeds)
    return speeds


def _seg_pt_dist_sq(px, py, x1, y1, x2, y2):
    """Min sq distance from point (px,py) to segments (x1,y1)→(x2,y2).

    Inputs may be scalars or arrays; broadcasting applies.
    """
    seg_dx = x2 - x1
    seg_dy = y2 - y1
    lsq = seg_dx * seg_dx + seg_dy * seg_dy
    if np.isscalar(lsq):
        if lsq < 1e-12:
            return (px - x1) ** 2 + (py - y1) ** 2
        t = ((px - x1) * seg_dx + (py - y1) * seg_dy) / lsq
        t = max(0.0, min(1.0, t))
        qx = x1 + t * seg_dx
        qy = y1 + t * seg_dy
        return (px - qx) ** 2 + (py - qy) ** 2
    safe = np.where(lsq < 1e-12, 1.0, lsq)
    t = ((px - x1) * seg_dx + (py - y1) * seg_dy) / safe
    t = np.clip(t, 0.0, 1.0)
    qx = x1 + t * seg_dx
    qy = y1 + t * seg_dy
    return (px - qx) ** 2 + (py - qy) ** 2


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _generate_comet_paths(initial_planets, angular_velocity, spawn_step,
                          comet_planet_ids=None, comet_speed=COMET_SPEED):
    """Faithful copy of Kaggle's comet path generator."""
    if comet_planet_ids is None:
        comet_planet_ids = set()
    else:
        comet_planet_ids = set(comet_planet_ids)

    for _ in range(300):
        e = random.uniform(0.75, 0.93)
        a = random.uniform(60, 150)
        perihelion = a * (1 - e)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            continue

        b = a * math.sqrt(1 - e ** 2)
        c_val = a * e
        phi = random.uniform(math.pi / 6, math.pi / 3)

        dense = []
        num = 5000
        for i in range(num):
            t = 0.3 * math.pi + 1.4 * math.pi * i / (num - 1)
            ex = c_val + a * math.cos(t)
            ey = b * math.sin(t)
            x = CENTER + ex * math.cos(phi) - ey * math.sin(phi)
            y = CENTER + ex * math.sin(phi) + ey * math.cos(phi)
            dense.append((x, y))

        path = [dense[0]]
        cum = 0.0
        target = comet_speed
        for i in range(1, len(dense)):
            cum += _dist(dense[i], dense[i - 1])
            if cum >= target:
                path.append(dense[i])
                target += comet_speed

        board_start = None
        board_end = None
        for i, (x, y) in enumerate(path):
            if 0 <= x <= BOARD_SIZE and 0 <= y <= BOARD_SIZE:
                if board_start is None:
                    board_start = i
                board_end = i

        if board_start is None:
            continue
        visible = path[board_start: board_end + 1]
        if not (5 <= len(visible) <= 40):
            continue

        paths = [
            [[x, y] for x, y in visible],
            [[BOARD_SIZE - x, y] for x, y in visible],
            [[x, BOARD_SIZE - y] for x, y in visible],
            [[BOARD_SIZE - x, BOARD_SIZE - y] for x, y in visible],
        ]

        static_planets = []
        orbiting_planets = []
        for planet in initial_planets:
            if int(planet[0]) in comet_planet_ids:
                continue
            pr = _dist((planet[2], planet[3]), (CENTER, CENTER))
            if pr + planet[4] < ROTATION_RADIUS_LIMIT:
                orbiting_planets.append(planet)
            else:
                static_planets.append(planet)

        valid = True
        buf = COMET_RADIUS + 0.5
        for k, (cx, cy) in enumerate(visible):
            if _dist((cx, cy), (CENTER, CENTER)) < SUN_RADIUS + COMET_RADIUS:
                valid = False
                break

            sym_pts = [
                (cx, cy),
                (BOARD_SIZE - cx, cy),
                (cx, BOARD_SIZE - cy),
                (BOARD_SIZE - cx, BOARD_SIZE - cy),
            ]
            for planet in static_planets:
                for sp in sym_pts:
                    if _dist(sp, (planet[2], planet[3])) < planet[4] + buf:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

            game_step = spawn_step - 1 + k
            for planet in orbiting_planets:
                dx = planet[2] - CENTER
                dy = planet[3] - CENTER
                orb_r = math.sqrt(dx ** 2 + dy ** 2)
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + angular_velocity * game_step
                px = CENTER + orb_r * math.cos(cur_angle)
                py = CENTER + orb_r * math.sin(cur_angle)
                for sp in sym_pts:
                    if _dist(sp, (px, py)) < planet[4] + COMET_RADIUS:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if valid:
            return paths
    return None


def _initial_planet_rows(state: GameState) -> List[list]:
    """Build Kaggle-style initial planet rows from compact state."""
    init_pos_by_id = {
        int(state.init_planet_ids[i]): state.init_positions[i]
        for i in range(len(state.init_planet_ids))
    }
    rows = []
    for p in state.planets:
        pid = int(p[P_ID])
        pos = init_pos_by_id.get(pid)
        if pos is None:
            x, y = float(p[P_X]), float(p[P_Y])
        else:
            x, y = float(pos[0]), float(pos[1])
        rows.append([
            pid, int(p[P_OWNER]), x, y, float(p[P_R]),
            int(p[P_SHIPS]), int(p[P_PROD]),
        ])
    return rows


def _spawn_comets_if_needed(state: GameState) -> None:
    spawn_step = int(state.step) + 1
    if spawn_step not in COMET_SPAWN_STEPS or len(state.planets) == 0:
        return

    initial_planets = _initial_planet_rows(state)
    comet_paths = _generate_comet_paths(
        initial_planets,
        state.angular_velocity,
        spawn_step,
        state.comet_ids,
        COMET_SPEED,
    )
    if not comet_paths:
        return

    next_id = int(np.max(state.planets[:, P_ID])) + 1
    comet_ships = min(
        random.randint(1, 99),
        random.randint(1, 99),
        random.randint(1, 99),
        random.randint(1, 99),
    )

    group = {"planet_ids": [], "paths": comet_paths, "path_index": -1}
    new_planets = []
    new_init_ids = []
    new_init_pos = []
    for i, _ in enumerate(comet_paths):
        pid = next_id + i
        group["planet_ids"].append(pid)
        state.comet_ids.add(pid)
        new_planets.append([
            pid, -1, -99.0, -99.0, COMET_RADIUS,
            comet_ships, COMET_PRODUCTION,
        ])
        new_init_ids.append(pid)
        new_init_pos.append([-99.0, -99.0])

    state.planets = np.vstack([state.planets, np.array(new_planets, dtype=np.float32)])
    state.init_planet_ids = np.concatenate([
        state.init_planet_ids,
        np.array(new_init_ids, dtype=np.int32),
    ])
    state.init_positions = np.vstack([
        state.init_positions,
        np.array(new_init_pos, dtype=np.float32),
    ])
    state.comet_groups.append(group)


# --- Planet position update -------------------------------------------------

def update_planet_positions(state: GameState) -> None:
    """Update non-comet planet x,y to current orbital positions in place."""
    planets = state.planets
    if len(planets) == 0:
        return

    n = len(planets)
    # Build id → init_idx
    init_id_to_idx = {int(state.init_planet_ids[i]): i for i in range(len(state.init_planet_ids))}

    init_dx = np.zeros(n, dtype=np.float32)
    init_dy = np.zeros(n, dtype=np.float32)
    has_init = np.zeros(n, dtype=bool)
    for i in range(n):
        pid = int(planets[i, P_ID])
        if pid in state.comet_ids:
            continue
        idx = init_id_to_idx.get(pid)
        if idx is None:
            continue
        init_dx[i] = state.init_positions[idx, 0] - CENTER
        init_dy[i] = state.init_positions[idx, 1] - CENTER
        has_init[i] = True

    r = np.hypot(init_dx, init_dy)
    in_orbit = has_init & ((r + planets[:, P_R]) < ROTATION_RADIUS_LIMIT)
    if not np.any(in_orbit):
        return

    angle_init = np.arctan2(init_dy[in_orbit], init_dx[in_orbit])
    angle_now = angle_init + state.angular_velocity * state.step
    state.planets[in_orbit, P_X] = CENTER + r[in_orbit] * np.cos(angle_now)
    state.planets[in_orbit, P_Y] = CENTER + r[in_orbit] * np.sin(angle_now)


def predict_planet_position(state: GameState, planet_id: int, future_step: int) -> tuple:
    """Predict (x, y) of a planet at a future absolute step number."""
    pid_int = int(planet_id)
    cur_idx = np.where(state.planets[:, P_ID] == planet_id)[0]
    if pid_int in state.comet_ids:
        if len(cur_idx) == 0:
            return (0.0, 0.0)
        return (float(state.planets[cur_idx[0], P_X]), float(state.planets[cur_idx[0], P_Y]))

    init_idx = None
    for i in range(len(state.init_planet_ids)):
        if int(state.init_planet_ids[i]) == pid_int:
            init_idx = i
            break
    if init_idx is None or len(cur_idx) == 0:
        if len(cur_idx) == 0:
            return (0.0, 0.0)
        return (float(state.planets[cur_idx[0], P_X]), float(state.planets[cur_idx[0], P_Y]))

    dx = float(state.init_positions[init_idx, 0]) - CENTER
    dy = float(state.init_positions[init_idx, 1]) - CENTER
    r = math.hypot(dx, dy)
    pradius = float(state.planets[cur_idx[0], P_R])

    if r + pradius >= ROTATION_RADIUS_LIMIT:
        return (float(state.init_positions[init_idx, 0]),
                float(state.init_positions[init_idx, 1]))

    angle = math.atan2(dy, dx) + state.angular_velocity * future_step
    return (CENTER + r * math.cos(angle), CENTER + r * math.sin(angle))


# --- Step function ----------------------------------------------------------

def step(state: GameState, actions_by_player: Dict[int, list]) -> GameState:
    """Simulate 1 turn. Returns NEW state (input not mutated)."""
    return step_inplace(state.copy(), actions_by_player)


def step_inplace(s: GameState, actions_by_player: Dict[int, list]) -> GameState:
    """Simulate 1 turn, mutating `s`."""
    _spawn_comets_if_needed(s)
    planets = s.planets

    # --- 1. Fleet launch --------------------------------------------------
    new_fleet_rows = []
    if len(planets) > 0 and actions_by_player:
        pid_to_row = {int(planets[i, P_ID]): i for i in range(len(planets))}
        for player_id, action in actions_by_player.items():
            if not action:
                continue
            for move in action:
                if not move or len(move) != 3:
                    continue
                from_id, angle, ships_req = move
                try:
                    ships_req = int(ships_req)
                except (ValueError, TypeError):
                    continue
                if ships_req <= 0:
                    continue
                row = pid_to_row.get(int(from_id))
                if row is None:
                    continue
                if int(planets[row, P_OWNER]) != int(player_id):
                    continue
                avail = int(planets[row, P_SHIPS])
                if avail < ships_req:
                    continue
                planets[row, P_SHIPS] -= ships_req
                pradius = float(planets[row, P_R])
                start_x = float(planets[row, P_X]) + math.cos(angle) * (pradius + 0.1)
                start_y = float(planets[row, P_Y]) + math.sin(angle) * (pradius + 0.1)
                new_fleet_rows.append([
                    s.next_fleet_id, int(player_id), start_x, start_y,
                    float(angle), int(from_id), ships_req,
                ])
                s.next_fleet_id += 1

    if new_fleet_rows:
        new_arr = np.array(new_fleet_rows, dtype=np.float32)
        s.fleets = new_arr if len(s.fleets) == 0 else np.vstack([s.fleets, new_arr])
    fleets = s.fleets

    # --- 2. Production ----------------------------------------------------
    if len(planets) > 0:
        owned = planets[:, P_OWNER] >= 0
        if np.any(owned):
            planets[owned, P_SHIPS] += planets[owned, P_PROD]

    # Snapshot fleets BEFORE movement (combat needs original ships/owner;
    # movement only updates x/y so a ref to fleets is fine — but combat
    # references row indices, so we save a copy in case the array is replaced).
    fleet_snapshot = fleets.copy() if len(fleets) > 0 else np.zeros((0, 7), dtype=np.float32)

    # --- 3. Fleet movement + collisions ----------------------------------
    combat_lists: Dict[int, List[int]] = {}
    dead_set: Set[int] = set()

    if len(fleets) > 0:
        speeds = fleet_speeds(fleets[:, F_SHIPS])
        old_x = fleets[:, F_X].copy()
        old_y = fleets[:, F_Y].copy()
        cos_a = np.cos(fleets[:, F_ANGLE])
        sin_a = np.sin(fleets[:, F_ANGLE])
        fleets[:, F_X] = old_x + cos_a * speeds
        fleets[:, F_Y] = old_y + sin_a * speeds
        new_x = fleets[:, F_X]
        new_y = fleets[:, F_Y]

        oob = (new_x < 0) | (new_x > BOARD_SIZE) | (new_y < 0) | (new_y > BOARD_SIZE)
        sun_d2 = _seg_pt_dist_sq(
            np.full_like(new_x, CENTER), np.full_like(new_y, CENTER),
            old_x, old_y, new_x, new_y,
        )
        sun_hit = sun_d2 < SUN_RADIUS * SUN_RADIUS

        for fi in range(len(fleets)):
            if oob[fi] or sun_hit[fi]:
                dead_set.add(fi)
                continue
            if len(planets) == 0:
                continue
            ox, oy = float(old_x[fi]), float(old_y[fi])
            nx, ny = float(new_x[fi]), float(new_y[fi])
            seg_dx = nx - ox
            seg_dy = ny - oy
            lsq = seg_dx * seg_dx + seg_dy * seg_dy
            px = planets[:, P_X]
            py = planets[:, P_Y]
            pr = planets[:, P_R]
            if lsq < 1e-12:
                d2 = (px - ox) ** 2 + (py - oy) ** 2
            else:
                t = ((px - ox) * seg_dx + (py - oy) * seg_dy) / lsq
                t = np.clip(t, 0.0, 1.0)
                qx = ox + t * seg_dx
                qy = oy + t * seg_dy
                d2 = (px - qx) ** 2 + (py - qy) ** 2
            hits = d2 < pr * pr
            if np.any(hits):
                pi = int(np.argmax(hits))
                combat_lists.setdefault(pi, []).append(fi)
                dead_set.add(fi)

    # --- 4. Planet rotation + sweep --------------------------------------
    if len(planets) > 0:
        old_planet_x = planets[:, P_X].copy()
        old_planet_y = planets[:, P_Y].copy()
        update_planet_positions(s)
        if len(fleets) > 0:
            for pi in range(len(planets)):
                ox, oy = float(old_planet_x[pi]), float(old_planet_y[pi])
                nx, ny = float(planets[pi, P_X]), float(planets[pi, P_Y])
                if ox == nx and oy == ny:
                    continue
                pr = float(planets[pi, P_R])
                survivors = [fi for fi in range(len(fleets)) if fi not in dead_set]
                if not survivors:
                    break
                fx = fleets[survivors, F_X]
                fy = fleets[survivors, F_Y]
                seg_dx = nx - ox
                seg_dy = ny - oy
                lsq = seg_dx * seg_dx + seg_dy * seg_dy
                if lsq < 1e-12:
                    d2 = (fx - ox) ** 2 + (fy - oy) ** 2
                else:
                    t = ((fx - ox) * seg_dx + (fy - oy) * seg_dy) / lsq
                    t = np.clip(t, 0.0, 1.0)
                    qx = ox + t * seg_dx
                    qy = oy + t * seg_dy
                    d2 = (qx - fx) ** 2 + (qy - fy) ** 2
                hits = d2 < pr * pr
                if np.any(hits):
                    for hi in np.where(hits)[0]:
                        global_fi = survivors[int(hi)]
                        combat_lists.setdefault(pi, []).append(global_fi)
                        dead_set.add(global_fi)

    # --- 5. Comet movement + sweep + expiry -----------------------------
    expired_pids: List[int] = []
    for group in s.comet_groups:
        group["path_index"] += 1
        idx = group["path_index"]
        for ci, pid in enumerate(group["planet_ids"]):
            row = -1
            for k in range(len(s.planets)):
                if int(s.planets[k, P_ID]) == pid:
                    row = k
                    break
            if row < 0:
                continue
            p_path = group["paths"][ci]
            if idx >= len(p_path):
                expired_pids.append(pid)
                continue
            ox = float(s.planets[row, P_X])
            oy = float(s.planets[row, P_Y])
            nx, ny = float(p_path[idx][0]), float(p_path[idx][1])
            s.planets[row, P_X] = nx
            s.planets[row, P_Y] = ny
            if ox < 0 or len(s.fleets) == 0:
                continue
            survivors = [fi for fi in range(len(s.fleets)) if fi not in dead_set]
            if not survivors:
                continue
            pr = float(s.planets[row, P_R])
            fx = s.fleets[survivors, F_X]
            fy = s.fleets[survivors, F_Y]
            seg_dx = nx - ox
            seg_dy = ny - oy
            lsq = seg_dx * seg_dx + seg_dy * seg_dy
            if lsq < 1e-12:
                d2 = (fx - ox) ** 2 + (fy - oy) ** 2
            else:
                t = ((fx - ox) * seg_dx + (fy - oy) * seg_dy) / lsq
                t = np.clip(t, 0.0, 1.0)
                qx = ox + t * seg_dx
                qy = oy + t * seg_dy
                d2 = (qx - fx) ** 2 + (qy - fy) ** 2
            hits = d2 < pr * pr
            if np.any(hits):
                for hi in np.where(hits)[0]:
                    global_fi = survivors[int(hi)]
                    combat_lists.setdefault(row, []).append(global_fi)
                    dead_set.add(global_fi)

    # --- 6. Combat resolution -------------------------------------------
    for pi, fl_idx_list in combat_lists.items():
        if pi >= len(s.planets):
            continue
        per_player: Dict[int, int] = {}
        for fi in fl_idx_list:
            if fi >= len(fleet_snapshot):
                continue
            owner = int(fleet_snapshot[fi, F_OWNER])
            ships = int(fleet_snapshot[fi, F_SHIPS])
            per_player[owner] = per_player.get(owner, 0) + ships
        if not per_player:
            continue
        sorted_p = sorted(per_player.items(), key=lambda kv: kv[1], reverse=True)
        top_player, top_ships = sorted_p[0]
        if len(sorted_p) > 1:
            second_ships = sorted_p[1][1]
            survivor_ships = top_ships - second_ships
            survivor_owner = top_player if survivor_ships > 0 else -1
        else:
            survivor_owner = top_player
            survivor_ships = top_ships

        if survivor_ships > 0:
            planet_owner = int(s.planets[pi, P_OWNER])
            if planet_owner == survivor_owner:
                s.planets[pi, P_SHIPS] += survivor_ships
            else:
                s.planets[pi, P_SHIPS] -= survivor_ships
                if s.planets[pi, P_SHIPS] < 0:
                    s.planets[pi, P_OWNER] = survivor_owner
                    s.planets[pi, P_SHIPS] = -s.planets[pi, P_SHIPS]

    # --- Drop dead fleets -----------------------------------------------
    if dead_set and len(s.fleets) > 0:
        keep = np.ones(len(s.fleets), dtype=bool)
        for fi in dead_set:
            if 0 <= fi < len(keep):
                keep[fi] = False
        s.fleets = s.fleets[keep] if np.any(keep) else np.zeros((0, 7), dtype=np.float32)

    # --- Drop expired comets --------------------------------------------
    if expired_pids:
        expired = set(expired_pids)
        keep_p = np.ones(len(s.planets), dtype=bool)
        for k in range(len(s.planets)):
            if int(s.planets[k, P_ID]) in expired:
                keep_p[k] = False
        s.planets = s.planets[keep_p]
        keep_i = np.ones(len(s.init_planet_ids), dtype=bool)
        for k in range(len(s.init_planet_ids)):
            if int(s.init_planet_ids[k]) in expired:
                keep_i[k] = False
        s.init_planet_ids = s.init_planet_ids[keep_i]
        s.init_positions = s.init_positions[keep_i]
        s.comet_ids -= expired
        new_groups = []
        for g in s.comet_groups:
            g["planet_ids"] = [pid for pid in g["planet_ids"] if pid not in expired]
            if g["planet_ids"]:
                new_groups.append(g)
        s.comet_groups = new_groups

    # --- 7. Step counter ------------------------------------------------
    s.step += 1
    return s


# --- Conversions ------------------------------------------------------------

def state_from_obs(obs, player: Optional[int] = None) -> GameState:
    """Convert a Kaggle observation (dict or SimpleNamespace) to GameState."""
    def get(o, k, default):
        if isinstance(o, dict):
            return o.get(k, default)
        return getattr(o, k, default)

    raw_planets = list(get(obs, "planets", []) or [])
    raw_fleets = list(get(obs, "fleets", []) or [])
    raw_init = list(get(obs, "initial_planets", []) or [])
    ang = float(get(obs, "angular_velocity", 0.0) or 0.0)
    step_n = int(get(obs, "step", 0) or 0)
    plyr = player if player is not None else int(get(obs, "player", 0) or 0)
    next_fleet_id = int(get(obs, "next_fleet_id", 0) or 0)
    raw_comets = list(get(obs, "comets", []) or [])
    raw_comet_pids = list(get(obs, "comet_planet_ids", []) or [])

    if raw_planets:
        planets_arr = np.array([[float(v) for v in p] for p in raw_planets], dtype=np.float32)
    else:
        planets_arr = np.zeros((0, 7), dtype=np.float32)

    if raw_fleets:
        fleets_arr = np.array([[float(v) for v in f] for f in raw_fleets], dtype=np.float32)
    else:
        fleets_arr = np.zeros((0, 7), dtype=np.float32)

    if raw_init:
        init_ids = np.array([int(p[0]) for p in raw_init], dtype=np.int32)
        init_pos = np.array([[float(p[2]), float(p[3])] for p in raw_init], dtype=np.float32)
    else:
        if len(planets_arr) > 0:
            init_ids = planets_arr[:, P_ID].astype(np.int32)
            init_pos = planets_arr[:, [P_X, P_Y]].copy()
        else:
            init_ids = np.zeros(0, dtype=np.int32)
            init_pos = np.zeros((0, 2), dtype=np.float32)

    comet_groups = []
    for g in raw_comets:
        if isinstance(g, dict):
            pids = list(g.get("planet_ids", []) or [])
            paths = g.get("paths", []) or []
            idx = int(g.get("path_index", -1))
        else:
            pids = list(getattr(g, "planet_ids", []) or [])
            paths = getattr(g, "paths", []) or []
            idx = int(getattr(g, "path_index", -1))
        norm_paths = []
        for p in paths:
            np_path = []
            for pt in p:
                np_path.append([float(pt[0]), float(pt[1])])
            norm_paths.append(np_path)
        comet_groups.append({
            "planet_ids": [int(x) for x in pids],
            "paths": norm_paths,
            "path_index": idx,
        })

    return GameState(
        planets=planets_arr,
        fleets=fleets_arr,
        init_positions=init_pos,
        init_planet_ids=init_ids,
        angular_velocity=ang,
        comet_ids=set(int(x) for x in raw_comet_pids),
        comet_groups=comet_groups,
        step=step_n,
        player=plyr,
        next_fleet_id=next_fleet_id,
    )


def player_total_ships(state: GameState, player: int) -> int:
    """Total ships (planets + fleets) owned by `player`."""
    p_total = 0
    if len(state.planets) > 0:
        mask = state.planets[:, P_OWNER] == player
        if np.any(mask):
            p_total = int(state.planets[mask, P_SHIPS].sum())
    f_total = 0
    if len(state.fleets) > 0:
        mask = state.fleets[:, F_OWNER] == player
        if np.any(mask):
            f_total = int(state.fleets[mask, F_SHIPS].sum())
    return p_total + f_total


def is_terminal(state: GameState) -> bool:
    """Game over: max steps reached or only one player has any presence."""
    if state.step >= TOTAL_TURNS - 1:
        return True
    alive = set()
    if len(state.planets) > 0:
        for o in state.planets[:, P_OWNER]:
            o_int = int(o)
            if o_int >= 0:
                alive.add(o_int)
    if len(state.fleets) > 0:
        for o in state.fleets[:, F_OWNER]:
            alive.add(int(o))
    return len(alive) <= 1


def winner(state: GameState, n_players: int = 2) -> int:
    """Return winning player id, or -1 on tie / no ships."""
    scores = [0] * n_players
    if len(state.planets) > 0:
        for o, sh in zip(state.planets[:, P_OWNER], state.planets[:, P_SHIPS]):
            o_int = int(o)
            if 0 <= o_int < n_players:
                scores[o_int] += int(sh)
    if len(state.fleets) > 0:
        for o, sh in zip(state.fleets[:, F_OWNER], state.fleets[:, F_SHIPS]):
            o_int = int(o)
            if 0 <= o_int < n_players:
                scores[o_int] += int(sh)
    mx = max(scores)
    winners = [i for i, s in enumerate(scores) if s == mx and mx > 0]
    return winners[0] if len(winners) == 1 else -1
