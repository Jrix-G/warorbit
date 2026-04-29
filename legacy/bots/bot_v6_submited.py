"""Orbit Wars V6.5 single-file Kaggle agent."""

import math
import time
import base64
import io
import collections

import numpy as np


BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0
COMET_SPEED = 4.0
TOTAL_TURNS = 500
COMET_TURNS = (50, 150, 250, 350, 450)
P_ID, P_OWNER, P_X, P_Y, P_R, P_SHIPS, P_PROD = range(7)
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = range(7)


def _get(o, k, d=None):
    return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)


class GameState:
    def __init__(self, planets, fleets, init_positions, init_planet_ids,
                 angular_velocity, comet_ids, comet_groups, step, player,
                 next_fleet_id=0):
        self.planets = planets
        self.fleets = fleets
        self.init_positions = init_positions
        self.init_planet_ids = init_planet_ids
        self.angular_velocity = angular_velocity
        self.comet_ids = comet_ids
        self.comet_groups = comet_groups
        self.step = step
        self.player = player
        self.next_fleet_id = next_fleet_id

    def copy(self):
        return GameState(
            self.planets.copy(),
            self.fleets.copy() if len(self.fleets) else np.zeros((0, 7), dtype=np.float32),
            self.init_positions.copy(),
            self.init_planet_ids.copy(),
            self.angular_velocity,
            set(self.comet_ids),
            [{"planet_ids": list(g["planet_ids"]), "paths": g["paths"], "path_index": g["path_index"]}
             for g in self.comet_groups],
            self.step,
            self.player,
            self.next_fleet_id,
        )


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    s = 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5
    return min(s, MAX_SPEED)


def fleet_speeds(ships_array):
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


def update_planet_positions(state):
    if len(state.planets) == 0:
        return
    init_id_to_idx = {int(state.init_planet_ids[i]): i for i in range(len(state.init_planet_ids))}
    n = len(state.planets)
    init_dx = np.zeros(n, dtype=np.float32)
    init_dy = np.zeros(n, dtype=np.float32)
    has_init = np.zeros(n, dtype=bool)
    for i in range(n):
        pid = int(state.planets[i, P_ID])
        if pid in state.comet_ids:
            continue
        idx = init_id_to_idx.get(pid)
        if idx is None:
            continue
        init_dx[i] = state.init_positions[idx, 0] - CENTER
        init_dy[i] = state.init_positions[idx, 1] - CENTER
        has_init[i] = True
    r = np.hypot(init_dx, init_dy)
    in_orbit = has_init & ((r + state.planets[:, P_R]) < ROTATION_RADIUS_LIMIT)
    if np.any(in_orbit):
        angle_init = np.arctan2(init_dy[in_orbit], init_dx[in_orbit])
        angle_now = angle_init + state.angular_velocity * state.step
        state.planets[in_orbit, P_X] = CENTER + r[in_orbit] * np.cos(angle_now)
        state.planets[in_orbit, P_Y] = CENTER + r[in_orbit] * np.sin(angle_now)


def state_from_obs(obs, player=None):
    raw_planets = list(_get(obs, "planets", []) or [])
    raw_fleets = list(_get(obs, "fleets", []) or [])
    raw_init = list(_get(obs, "initial_planets", []) or [])
    raw_comets = list(_get(obs, "comets", []) or [])
    raw_comet_pids = list(_get(obs, "comet_planet_ids", []) or [])
    planets = np.array([[float(v) for v in p] for p in raw_planets], dtype=np.float32) if raw_planets else np.zeros((0, 7), dtype=np.float32)
    fleets = np.array([[float(v) for v in f] for f in raw_fleets], dtype=np.float32) if raw_fleets else np.zeros((0, 7), dtype=np.float32)
    if raw_init:
        init_ids = np.array([int(p[0]) for p in raw_init], dtype=np.int32)
        init_pos = np.array([[float(p[2]), float(p[3])] for p in raw_init], dtype=np.float32)
    elif len(planets) > 0:
        init_ids = planets[:, P_ID].astype(np.int32)
        init_pos = planets[:, [P_X, P_Y]].copy()
    else:
        init_ids = np.zeros(0, dtype=np.int32)
        init_pos = np.zeros((0, 2), dtype=np.float32)
    comet_groups = []
    for g in raw_comets:
        pids = list(_get(g, "planet_ids", []) or [])
        paths = _get(g, "paths", []) or []
        idx = int(_get(g, "path_index", -1))
        comet_groups.append({
            "planet_ids": [int(x) for x in pids],
            "paths": [[[float(pt[0]), float(pt[1])] for pt in path] for path in paths],
            "path_index": idx,
        })
    plyr = int(player if player is not None else (_get(obs, "player", 0) or 0))
    return GameState(planets, fleets, init_pos, init_ids,
                     float(_get(obs, "angular_velocity", 0.0) or 0.0),
                     set(int(x) for x in raw_comet_pids), comet_groups,
                     int(_get(obs, "step", 0) or 0), plyr,
                     int(_get(obs, "next_fleet_id", 0) or 0))


def step_inplace(s, actions_by_player):
    planets = s.planets
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
                except Exception:
                    continue
                row = pid_to_row.get(int(from_id))
                if ships_req <= 0 or row is None or int(planets[row, P_OWNER]) != int(player_id):
                    continue
                avail = int(planets[row, P_SHIPS])
                if avail < ships_req:
                    continue
                planets[row, P_SHIPS] -= ships_req
                pr = float(planets[row, P_R])
                new_fleet_rows.append([
                    s.next_fleet_id, int(player_id),
                    float(planets[row, P_X]) + math.cos(angle) * (pr + 0.1),
                    float(planets[row, P_Y]) + math.sin(angle) * (pr + 0.1),
                    float(angle), int(from_id), ships_req,
                ])
                s.next_fleet_id += 1
    if new_fleet_rows:
        arr = np.array(new_fleet_rows, dtype=np.float32)
        s.fleets = arr if len(s.fleets) == 0 else np.vstack([s.fleets, arr])
    fleets = s.fleets
    if len(planets) > 0:
        owned = planets[:, P_OWNER] >= 0
        if np.any(owned):
            planets[owned, P_SHIPS] += planets[owned, P_PROD]
    fleet_snapshot = fleets.copy() if len(fleets) else np.zeros((0, 7), dtype=np.float32)
    combat_lists = {}
    dead_set = set()
    if len(fleets) > 0:
        speeds = fleet_speeds(fleets[:, F_SHIPS])
        old_x = fleets[:, F_X].copy()
        old_y = fleets[:, F_Y].copy()
        fleets[:, F_X] = old_x + np.cos(fleets[:, F_ANGLE]) * speeds
        fleets[:, F_Y] = old_y + np.sin(fleets[:, F_ANGLE]) * speeds
        new_x = fleets[:, F_X]
        new_y = fleets[:, F_Y]
        oob = (new_x < 0) | (new_x > BOARD_SIZE) | (new_y < 0) | (new_y > BOARD_SIZE)
        sun_hit = _seg_pt_dist_sq(np.full_like(new_x, CENTER), np.full_like(new_y, CENTER),
                                  old_x, old_y, new_x, new_y) < SUN_RADIUS * SUN_RADIUS
        for fi in range(len(fleets)):
            if oob[fi] or sun_hit[fi]:
                dead_set.add(fi)
                continue
            if len(planets) == 0:
                continue
            ox, oy, nx, ny = float(old_x[fi]), float(old_y[fi]), float(new_x[fi]), float(new_y[fi])
            seg_dx, seg_dy = nx - ox, ny - oy
            lsq = seg_dx * seg_dx + seg_dy * seg_dy
            if lsq < 1e-12:
                d2 = (planets[:, P_X] - ox) ** 2 + (planets[:, P_Y] - oy) ** 2
            else:
                t = ((planets[:, P_X] - ox) * seg_dx + (planets[:, P_Y] - oy) * seg_dy) / lsq
                t = np.clip(t, 0.0, 1.0)
                d2 = (planets[:, P_X] - (ox + t * seg_dx)) ** 2 + (planets[:, P_Y] - (oy + t * seg_dy)) ** 2
            hits = d2 < planets[:, P_R] * planets[:, P_R]
            if np.any(hits):
                pi = int(np.argmax(hits))
                combat_lists.setdefault(pi, []).append(fi)
                dead_set.add(fi)
    if len(planets) > 0:
        old_px = planets[:, P_X].copy()
        old_py = planets[:, P_Y].copy()
        update_planet_positions(s)
        if len(s.fleets) > 0:
            for pi in range(len(planets)):
                ox, oy, nx, ny = float(old_px[pi]), float(old_py[pi]), float(planets[pi, P_X]), float(planets[pi, P_Y])
                if ox == nx and oy == ny:
                    continue
                survivors = [fi for fi in range(len(s.fleets)) if fi not in dead_set]
                if not survivors:
                    break
                seg_dx, seg_dy = nx - ox, ny - oy
                lsq = seg_dx * seg_dx + seg_dy * seg_dy
                fx, fy = s.fleets[survivors, F_X], s.fleets[survivors, F_Y]
                if lsq < 1e-12:
                    d2 = (fx - ox) ** 2 + (fy - oy) ** 2
                else:
                    t = ((fx - ox) * seg_dx + (fy - oy) * seg_dy) / lsq
                    t = np.clip(t, 0.0, 1.0)
                    d2 = ((ox + t * seg_dx) - fx) ** 2 + ((oy + t * seg_dy) - fy) ** 2
                for hi in np.where(d2 < planets[pi, P_R] * planets[pi, P_R])[0]:
                    fi = survivors[int(hi)]
                    combat_lists.setdefault(pi, []).append(fi)
                    dead_set.add(fi)
    expired = []
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
            path = group["paths"][ci]
            if idx >= len(path):
                expired.append(pid)
                continue
            ox, oy = float(s.planets[row, P_X]), float(s.planets[row, P_Y])
            nx, ny = float(path[idx][0]), float(path[idx][1])
            s.planets[row, P_X], s.planets[row, P_Y] = nx, ny
            if ox < 0 or len(s.fleets) == 0:
                continue
            survivors = [fi for fi in range(len(s.fleets)) if fi not in dead_set]
            if not survivors:
                continue
            seg_dx, seg_dy = nx - ox, ny - oy
            lsq = seg_dx * seg_dx + seg_dy * seg_dy
            fx, fy = s.fleets[survivors, F_X], s.fleets[survivors, F_Y]
            if lsq < 1e-12:
                d2 = (fx - ox) ** 2 + (fy - oy) ** 2
            else:
                t = ((fx - ox) * seg_dx + (fy - oy) * seg_dy) / lsq
                t = np.clip(t, 0.0, 1.0)
                d2 = ((ox + t * seg_dx) - fx) ** 2 + ((oy + t * seg_dy) - fy) ** 2
            for hi in np.where(d2 < s.planets[row, P_R] * s.planets[row, P_R])[0]:
                fi = survivors[int(hi)]
                combat_lists.setdefault(row, []).append(fi)
                dead_set.add(fi)
    for pi, fl_idx_list in combat_lists.items():
        if pi >= len(s.planets):
            continue
        per_player = {}
        for fi in fl_idx_list:
            if fi >= len(fleet_snapshot):
                continue
            owner = int(fleet_snapshot[fi, F_OWNER])
            per_player[owner] = per_player.get(owner, 0) + int(fleet_snapshot[fi, F_SHIPS])
        if not per_player:
            continue
        sorted_p = sorted(per_player.items(), key=lambda kv: kv[1], reverse=True)
        top_player, top_ships = sorted_p[0]
        if len(sorted_p) > 1:
            survivor_ships = top_ships - sorted_p[1][1]
            survivor_owner = top_player if survivor_ships > 0 else -1
        else:
            survivor_owner, survivor_ships = top_player, top_ships
        if survivor_ships > 0:
            if int(s.planets[pi, P_OWNER]) == survivor_owner:
                s.planets[pi, P_SHIPS] += survivor_ships
            else:
                s.planets[pi, P_SHIPS] -= survivor_ships
                if s.planets[pi, P_SHIPS] < 0:
                    s.planets[pi, P_OWNER] = survivor_owner
                    s.planets[pi, P_SHIPS] = -s.planets[pi, P_SHIPS]
    if dead_set and len(s.fleets) > 0:
        keep = np.ones(len(s.fleets), dtype=bool)
        for fi in dead_set:
            if 0 <= fi < len(keep):
                keep[fi] = False
        s.fleets = s.fleets[keep] if np.any(keep) else np.zeros((0, 7), dtype=np.float32)
    if expired:
        expired_set = set(expired)
        keep_p = np.array([int(p[P_ID]) not in expired_set for p in s.planets], dtype=bool)
        s.planets = s.planets[keep_p]
        keep_i = np.array([int(pid) not in expired_set for pid in s.init_planet_ids], dtype=bool)
        s.init_planet_ids = s.init_planet_ids[keep_i]
        s.init_positions = s.init_positions[keep_i]
        s.comet_ids -= expired_set
        s.comet_groups = [g for g in s.comet_groups if [pid for pid in g["planet_ids"] if pid not in expired_set]]
    s.step += 1
    return s


def is_terminal(state):
    if state.step >= TOTAL_TURNS - 1:
        return True
    alive = set()
    if len(state.planets):
        for o in state.planets[:, P_OWNER]:
            if int(o) >= 0:
                alive.add(int(o))
    if len(state.fleets):
        for o in state.fleets[:, F_OWNER]:
            alive.add(int(o))
    return len(alive) <= 1


def extract_features(state, me):
    features = np.zeros(24, dtype=np.float32)
    planets, fleets = state.planets, state.fleets
    total_ships = np.zeros(4, dtype=np.float64)
    for p in range(4):
        if len(planets):
            m = planets[:, P_OWNER] == p
            if np.any(m):
                total_ships[p] += float(np.sum(planets[m, P_SHIPS]))
        if len(fleets):
            m = fleets[:, F_OWNER] == p
            if np.any(m):
                total_ships[p] += float(np.sum(fleets[m, F_SHIPS]))
    my_ships = total_ships[me]
    all_ships = total_ships.sum() + 1e-8
    features[0] = my_ships / all_ships
    features[1] = my_ships / (max(total_ships) + 1e-8)
    n_planets = max(len(planets), 1)
    my_mask = planets[:, P_OWNER] == me if len(planets) else np.array([], dtype=bool)
    my_planets = planets[my_mask] if len(planets) and np.any(my_mask) else np.zeros((0, 7), dtype=np.float32)
    features[2] = len(my_planets) / n_planets
    features[3] = float(np.sum(my_planets[:, P_PROD])) / max(n_planets * 3.0, 1.0) if len(my_planets) else 0.0
    incoming = 0.0
    if len(fleets) and len(my_planets):
        for ef in fleets[fleets[:, F_OWNER] != me]:
            for mp in my_planets:
                if math.hypot(float(ef[F_X] - mp[P_X]), float(ef[F_Y] - mp[P_Y])) < 30.0:
                    incoming += float(ef[F_SHIPS])
    features[4] = incoming / (my_ships + 1e-8)
    features[5] = state.step / 500.0
    features[6] = 1.0 if state.step < 40 else 0.0
    features[7] = 1.0 if 40 <= state.step < 150 else 0.0
    features[8] = 1.0 if state.step >= 150 else 0.0
    features[9] = 1.0 if len(state.comet_ids) else 0.0
    if len(state.comet_ids) and len(planets):
        owned = 0
        for cid in state.comet_ids:
            if np.any((planets[:, P_ID] == cid) & (planets[:, P_OWNER] == me)):
                owned += 1
        features[10] = owned / max(len(state.comet_ids), 1)
    for i, s in enumerate(sorted([total_ships[p] for p in range(4) if p != me], reverse=True)[:3]):
        features[11 + i] = s / all_ships
    if len(my_planets) > 1:
        cx, cy = float(np.mean(my_planets[:, P_X])), float(np.mean(my_planets[:, P_Y]))
        features[14] = float(np.mean(np.hypot(my_planets[:, P_X] - cx, my_planets[:, P_Y] - cy))) / 70.0
    if len(my_planets):
        features[15] = float(np.min(np.hypot(my_planets[:, P_X] - CENTER, my_planets[:, P_Y] - CENTER))) / 70.0
    remaining = (500 - state.step) / 500.0
    features[16] = remaining
    features[17] = 1.0 if remaining < 0.1 else 0.0
    if len(fleets):
        m = fleets[:, F_OWNER] == me
        if np.any(m):
            features[18] = float(np.sum(fleets[m, F_SHIPS])) / (my_ships + 1e-8)
    neutral = planets[planets[:, P_OWNER] == -1] if len(planets) else np.zeros((0, 7), dtype=np.float32)
    if len(neutral) and len(my_planets):
        cx, cy = float(np.mean(my_planets[:, P_X])), float(np.mean(my_planets[:, P_Y]))
        d = np.hypot(neutral[:, P_X] - cx, neutral[:, P_Y] - cy)
        features[19] = float(np.sum(neutral[d < 40.0, P_PROD])) / 20.0
    return features


class NumpyEvaluator:
    INPUT_DIM = 24
    H1 = 32
    H2 = 16

    def __init__(self, seed=None):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(self.INPUT_DIM, self.H1).astype(np.float32) * math.sqrt(2.0 / self.INPUT_DIM)
        self.b1 = np.zeros(self.H1, dtype=np.float32)
        self.W2 = rng.randn(self.H1, self.H2).astype(np.float32) * math.sqrt(2.0 / self.H1)
        self.b2 = np.zeros(self.H2, dtype=np.float32)
        self.W3 = rng.randn(self.H2, 1).astype(np.float32) * math.sqrt(2.0 / self.H2)
        self.b3 = np.zeros(1, dtype=np.float32)

    def predict(self, features):
        h1 = np.maximum(0.0, features @ self.W1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.W2 + self.b2)
        logit = float((h2 @ self.W3 + self.b3)[0])
        logit = max(-40.0, min(40.0, logit))
        return float(1.0 / (1.0 + math.exp(-logit)))

    def get_params(self):
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2, self.W3.ravel(), self.b3])

    def set_params(self, params):
        params = params.astype(np.float32)
        i = 0
        for attr, shape in [("W1", (24, 32)), ("b1", (32,)), ("W2", (32, 16)), ("b2", (16,)), ("W3", (16, 1)), ("b3", (1,))]:
            size = int(np.prod(shape))
            setattr(self, attr, params[i:i + size].reshape(shape))
            i += size


def decode_evaluator(b64_string):
    data = base64.b64decode(b64_string)
    params = np.load(io.BytesIO(data))
    ev = NumpyEvaluator()
    ev.set_params(params)
    return ev


def _heuristic_eval(state, me):
    total_s = my_s = total_p = my_p = 0.0
    for p in state.planets:
        total_s += float(p[P_SHIPS])
        total_p += float(p[P_PROD])
        if int(p[P_OWNER]) == me:
            my_s += float(p[P_SHIPS])
            my_p += float(p[P_PROD])
    for f in state.fleets:
        total_s += float(f[F_SHIPS])
        if int(f[F_OWNER]) == me:
            my_s += float(f[F_SHIPS])
    if total_s < 1e-6:
        return 0.5
    ship_ratio = my_s / total_s
    prod_ratio = my_p / max(total_p, 1.0)
    t = state.step / 500.0
    return (1 - t) * (0.4 * prod_ratio + 0.6 * ship_ratio) + t * (0.6 * ship_ratio + 0.4 * prod_ratio)


def evaluate_state(state, me, evaluator=None):
    if evaluator is not None:
        return evaluator.predict(extract_features(state, me))
    return _heuristic_eval(state, me)


def _planet_rows(state, owner):
    if len(state.planets) == 0:
        return np.zeros((0, 7), dtype=np.float32)
    m = state.planets[:, P_OWNER] == owner
    return state.planets[m] if np.any(m) else np.zeros((0, 7), dtype=np.float32)


def _planet_future_position(state, planet, turns):
    pid = int(planet[P_ID])
    if pid in state.comet_ids:
        for g in state.comet_groups:
            if pid not in g["planet_ids"]:
                continue
            ci = g["planet_ids"].index(pid)
            path = g["paths"][ci] if ci < len(g["paths"]) else None
            if not path:
                break
            idx = int(g["path_index"]) + int(round(turns))
            if idx < 0:
                idx = 0
            if idx >= len(path):
                idx = len(path) - 1
            return float(path[idx][0]), float(path[idx][1])
        return float(planet[P_X]), float(planet[P_Y])
    for i in range(len(state.init_planet_ids)):
        if int(state.init_planet_ids[i]) != pid:
            continue
        ix = float(state.init_positions[i, 0])
        iy = float(state.init_positions[i, 1])
        r = math.hypot(ix - CENTER, iy - CENTER)
        if r + float(planet[P_R]) < ROTATION_RADIUS_LIMIT:
            a = math.atan2(iy - CENTER, ix - CENTER) + state.angular_velocity * (state.step + turns)
            return CENTER + r * math.cos(a), CENTER + r * math.sin(a)
        break
    return float(planet[P_X]), float(planet[P_Y])


def _line_clears_sun(sx, sy, tx, ty, margin=1.25):
    return _segment_min_dist_to_sun(sx, sy, tx, ty) > SUN_RADIUS + margin


def _angle_hits_target(state, src, target, angle, ships, max_turns=140):
    sx = float(src[P_X]) + math.cos(angle) * (float(src[P_R]) + 0.1)
    sy = float(src[P_Y]) + math.sin(angle) * (float(src[P_R]) + 0.1)
    speed = fleet_speed(max(1, int(ships)))
    target_id = int(target[P_ID])

    for turn in range(int(max_turns)):
        nx = sx + math.cos(angle) * speed
        ny = sy + math.sin(angle) * speed
        if _segment_min_dist_to_sun(sx, sy, nx, ny) < SUN_RADIUS:
            return False
        if nx < 0.0 or nx > BOARD_SIZE or ny < 0.0 or ny > BOARD_SIZE:
            return False

        for planet in state.planets:
            px, py = _planet_future_position(state, planet, turn)
            if _seg_pt_dist_sq(px, py, sx, sy, nx, ny) < float(planet[P_R]) * float(planet[P_R]):
                return int(planet[P_ID]) == target_id

        for planet in state.planets:
            old_px, old_py = _planet_future_position(state, planet, turn)
            new_px, new_py = _planet_future_position(state, planet, turn + 1)
            if _seg_pt_dist_sq(nx, ny, old_px, old_py, new_px, new_py) < float(planet[P_R]) * float(planet[P_R]):
                return int(planet[P_ID]) == target_id

        sx, sy = nx, ny
    return False


def _intercept_aim(state, src, target, ships):
    sx = float(src[P_X])
    sy = float(src[P_Y])
    src_r = float(src[P_R])
    tgt_r = float(target[P_R])
    speed = fleet_speed(max(1, int(ships)))
    tx = float(target[P_X])
    ty = float(target[P_Y])
    eff_offset = src_r + 0.1 + tgt_r
    turns = max(0.0, _dist(sx, sy, tx, ty) - eff_offset) / max(speed, 1e-6)

    best = None
    for _ in range(8):
        tx, ty = _planet_future_position(state, target, turns)
        d = max(0.0, _dist(sx, sy, tx, ty) - eff_offset)
        new_turns = d / max(speed, 1e-6)
        if abs(new_turns - turns) < 0.05:
            turns = new_turns
            break
        turns = 0.55 * turns + 0.45 * new_turns

    for dt in (0.0, -1.0, 1.0, -2.0, 2.0, -4.0, 4.0, -7.0, 7.0, -11.0, 11.0, -18.0, 18.0):
        cand_turns = max(0.0, turns + dt)
        tx, ty = _planet_future_position(state, target, cand_turns)
        if not (0.0 <= tx <= BOARD_SIZE and 0.0 <= ty <= BOARD_SIZE):
            continue
        if not _line_clears_sun(sx, sy, tx, ty):
            continue
        angle = math.atan2(ty - sy, tx - sx)
        if not _angle_hits_target(state, src, target, angle, ships):
            continue
        eta = max(0.0, _dist(sx, sy, tx, ty) - eff_offset) / max(speed, 1e-6)
        err = abs(eta - cand_turns)
        if best is None or err < best[0]:
            best = (err, angle)

    if best is not None:
        return best[1]
    return None


def _heuristic_actions(state, player):
    my_planets = _planet_rows(state, player)
    targets = state.planets[state.planets[:, P_OWNER] != player] if len(state.planets) else np.zeros((0, 7), dtype=np.float32)
    if len(my_planets) == 0 or len(targets) == 0:
        return []
    actions = []
    committed = {}
    for src in my_planets:
        ships = int(src[P_SHIPS])
        if ships <= 5:
            continue
        d = np.hypot(targets[:, P_X] - src[P_X], targets[:, P_Y] - src[P_Y])
        score = targets[:, P_PROD] * 30.0 - targets[:, P_SHIPS] * 0.5 - d * 0.5
        order = np.argsort(-score)
        for idx in order:
            target = targets[int(idx)]
            tid = int(target[P_ID])
            already = committed.get(tid, 0)
            eta = float(d[int(idx)]) / 4.0
            need = max(1, int(math.ceil(float(target[P_SHIPS]) + float(target[P_PROD]) * eta + 1 - already)))
            send = min(ships - 1, max(need + 2, int(ships * 0.5)))
            if send < need:
                continue
            angle = _intercept_aim(state, src, target, send)
            if angle is None:
                continue
            actions.append([int(src[P_ID]), angle, send])
            committed[tid] = already + send
            break
    return actions


def _best_target(state, src, me, mode):
    if mode == 2:
        pool = state.planets[state.planets[:, P_OWNER] == -1] if len(state.planets) else np.zeros((0, 7), dtype=np.float32)
    elif mode == 3 and len(state.comet_ids):
        pool = np.array([p for p in state.planets if int(p[P_ID]) in state.comet_ids], dtype=np.float32)
    elif mode == 1:
        pool = state.planets[(state.planets[:, P_OWNER] == me) & (state.planets[:, P_ID] != src[P_ID])] if len(state.planets) else np.zeros((0, 7), dtype=np.float32)
    else:
        pool = state.planets[state.planets[:, P_OWNER] != me] if len(state.planets) else np.zeros((0, 7), dtype=np.float32)
    if len(pool) == 0:
        pool = state.planets[state.planets[:, P_OWNER] != me] if len(state.planets) else np.zeros((0, 7), dtype=np.float32)
    if len(pool) == 0:
        return None
    d = np.hypot(pool[:, P_X] - src[P_X], pool[:, P_Y] - src[P_Y])
    score = pool[:, P_PROD] * 30.0 - pool[:, P_SHIPS] * 0.4 - d * 0.7
    if mode == 1:
        score = -pool[:, P_SHIPS] - d * 0.2
    return pool[int(np.argmax(score))]


def _generate_action_set(state, me, strategy, best_actions=None):
    my_planets = _planet_rows(state, me)
    if len(my_planets) == 0:
        return []
    if strategy == 4 and best_actions:
        pid_to_ships = {int(p[P_ID]): int(p[P_SHIPS]) for p in my_planets}
        out = []
        for from_id, angle, ships in best_actions:
            avail = pid_to_ships.get(int(from_id), 0)
            if avail > 5:
                out.append([int(from_id), float(angle),
                            min(avail, max(1, int(float(ships) * float(np.random.uniform(0.8, 1.2)))))])
        if out:
            return out
        strategy = 0
    actions = []
    committed = {}
    for src in my_planets:
        ships = int(src[P_SHIPS])
        if ships <= 5:
            continue
        if strategy == 0:
            ratio_max = float(np.random.uniform(0.70, 0.85))
        elif strategy == 1:
            ratio_max = float(np.random.uniform(0.35, 0.55))
        elif strategy == 2:
            ratio_max = float(np.random.uniform(0.55, 0.70))
        elif strategy == 3:
            ratio_max = 0.85 if any(abs(state.step - ct) < 15 for ct in COMET_TURNS) else 0.70
        else:
            ratio_max = 0.70
        target = _best_target(state, src, me, strategy)
        if target is None:
            continue
        tid = int(target[P_ID])
        already = committed.get(tid, 0)
        d = math.hypot(float(target[P_X]) - float(src[P_X]), float(target[P_Y]) - float(src[P_Y]))
        eta = d / 4.0
        need = max(1, int(math.ceil(float(target[P_SHIPS]) + float(target[P_PROD]) * eta + 1 - already)))
        send = min(ships - 1, max(need + 2, int(ships * ratio_max * 0.5)))
        send = min(send, int(ships * ratio_max))
        if send < need or send < 1:
            continue
        angle = _intercept_aim(state, src, target, send)
        if angle is None:
            continue
        actions.append([int(src[P_ID]), angle, send])
        committed[tid] = already + send
    return actions


def generate_candidates(state, me, n_candidates):
    candidates, best_actions = [], None
    for i in range(int(n_candidates)):
        actions = _generate_action_set(state, me, i % 5, best_actions)
        candidates.append(actions)
        if best_actions is None and actions:
            best_actions = actions
    return candidates


def beam_search(state, time_budget=0.85, evaluator=None):
    t_start = time.time()
    me = int(state.player)
    n_candidates = min(100, max(20, int(50 * float(time_budget) / 0.85)))
    horizon = min(25, max(10, int(20 * float(time_budget) / 0.85)))
    players = set()
    for p in state.planets:
        if int(p[P_OWNER]) >= 0 and int(p[P_OWNER]) != me:
            players.add(int(p[P_OWNER]))
    for f in state.fleets:
        if int(f[F_OWNER]) >= 0 and int(f[F_OWNER]) != me:
            players.add(int(f[F_OWNER]))
    other_players = sorted(players)
    best_score, best_actions = -1.0e18, []
    for my_actions in generate_candidates(state, me, n_candidates):
        if time.time() - t_start > float(time_budget) * 0.9:
            break
        st = state.copy()
        score, discount = 0.0, 1.0
        for _ in range(horizon):
            all_actions = {me: my_actions}
            for opp in other_players:
                all_actions[opp] = _heuristic_actions(st, opp)
            step_inplace(st, all_actions)
            score += evaluate_state(st, me, evaluator) * discount
            discount *= 0.97
            if is_terminal(st) or time.time() - t_start > float(time_budget) * 0.9:
                break
        score += evaluate_state(st, me, evaluator) * discount * 5.0
        if score > best_score:
            best_score, best_actions = score, my_actions
    return best_actions if best_actions is not None else []


# V5 fallback.
SUN_X, SUN_Y = 50.0, 50.0
ROT_LIMIT = 50.0
EARLY_GAME_LIMIT = 40
LATE_GAME_LIMIT = 350
VERY_LATE_GAME_LIMIT = 450
FLEET_SEND_RATIO = 0.889
THREAT_THRESHOLD = 0.275
EARLY_GAME_DEFENSE = 0.09
MID_GAME_DEFENSE = 0.296
LATE_GAME_DEFENSE = 0.245
EARLY_HORIZON = 72
MID_HORIZON = 105
LATE_HORIZON = 120
VERY_LATE_HORIZON = 50


def _dist(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def _segment_min_dist_to_sun(x1, y1, x2, y2):
    seg_dx, seg_dy = x2 - x1, y2 - y1
    lsq = seg_dx * seg_dx + seg_dy * seg_dy
    if lsq < 1e-9:
        return _dist(x1, y1, SUN_X, SUN_Y)
    t = max(0.0, min(1.0, ((SUN_X - x1) * seg_dx + (SUN_Y - y1) * seg_dy) / lsq))
    return _dist(SUN_X, SUN_Y, x1 + t * seg_dx, y1 + t * seg_dy)


def _safe_angle(sx, sy, tx, ty):
    direct = math.atan2(ty - sy, tx - sx)
    d = _dist(sx, sy, tx, ty)
    if _segment_min_dist_to_sun(sx, sy, sx + math.cos(direct) * d, sy + math.sin(direct) * d) > SUN_RADIUS + 1.5:
        return direct
    for delta in (0.3, 0.6, 0.9):
        for sign in (1, -1):
            a = direct + sign * delta
            if _segment_min_dist_to_sun(sx, sy, sx + math.cos(a) * d, sy + math.sin(a) * d) > SUN_RADIUS + 1.0:
                return a
    return direct


def _obs_planet_future_position(obs, planet, turns):
    pid = int(planet[0])
    comet_ids = set(_get(obs, "comet_planet_ids", []) or [])
    if pid in comet_ids:
        for g in (_get(obs, "comets", []) or []):
            pids = list(_get(g, "planet_ids", []) or [])
            if pid not in pids:
                continue
            ci = pids.index(pid)
            paths = _get(g, "paths", []) or []
            if ci >= len(paths) or not paths[ci]:
                break
            path = paths[ci]
            base = int(_get(g, "path_index", 0) or 0)
            idx = base + int(round(turns))
            if idx < 0:
                idx = 0
            if idx >= len(path):
                idx = len(path) - 1
            return float(path[idx][0]), float(path[idx][1])
        return float(planet[2]), float(planet[3])
    step = int(_get(obs, "step", 0) or 0)
    ang_vel = float(_get(obs, "angular_velocity", 0.0) or 0.0)
    for init in (_get(obs, "initial_planets", []) or []):
        if int(init[0]) != pid:
            continue
        ix = float(init[2])
        iy = float(init[3])
        r = math.hypot(ix - SUN_X, iy - SUN_Y)
        if r + float(planet[4]) < ROT_LIMIT:
            a = math.atan2(iy - SUN_Y, ix - SUN_X) + ang_vel * (step + turns)
            return SUN_X + r * math.cos(a), SUN_Y + r * math.sin(a)
        break
    return float(planet[2]), float(planet[3])


def _obs_angle_hits_target(obs, src, target, angle, ships, max_turns=140):
    planets = _get(obs, "planets", []) or []
    sx = float(src[2]) + math.cos(angle) * (float(src[4]) + 0.1)
    sy = float(src[3]) + math.sin(angle) * (float(src[4]) + 0.1)
    speed = fleet_speed(max(1, int(ships)))
    target_id = int(target[0])

    for turn in range(int(max_turns)):
        nx = sx + math.cos(angle) * speed
        ny = sy + math.sin(angle) * speed
        if _segment_min_dist_to_sun(sx, sy, nx, ny) < SUN_RADIUS:
            return False
        if nx < 0.0 or nx > BOARD_SIZE or ny < 0.0 or ny > BOARD_SIZE:
            return False

        for planet in planets:
            px, py = _obs_planet_future_position(obs, planet, turn)
            if _seg_pt_dist_sq(px, py, sx, sy, nx, ny) < float(planet[4]) * float(planet[4]):
                return int(planet[0]) == target_id

        for planet in planets:
            old_px, old_py = _obs_planet_future_position(obs, planet, turn)
            new_px, new_py = _obs_planet_future_position(obs, planet, turn + 1)
            if _seg_pt_dist_sq(nx, ny, old_px, old_py, new_px, new_py) < float(planet[4]) * float(planet[4]):
                return int(planet[0]) == target_id

        sx, sy = nx, ny
    return False


def _obs_intercept_angle(obs, src, target, ships):
    sx = float(src[2])
    sy = float(src[3])
    src_r = float(src[4])
    tgt_r = float(target[4])
    eff_offset = src_r + 0.1 + tgt_r
    speed = fleet_speed(max(1, int(ships)))
    turns = max(0.0, _dist(sx, sy, float(target[2]), float(target[3])) - eff_offset) / max(speed, 1e-6)
    best = None

    for _ in range(8):
        tx, ty = _obs_planet_future_position(obs, target, turns)
        new_turns = max(0.0, _dist(sx, sy, tx, ty) - eff_offset) / max(speed, 1e-6)
        if abs(new_turns - turns) < 0.05:
            turns = new_turns
            break
        turns = 0.55 * turns + 0.45 * new_turns

    for dt in (0.0, -1.0, 1.0, -2.0, 2.0, -4.0, 4.0, -7.0, 7.0, -11.0, 11.0, -18.0, 18.0):
        cand_turns = max(0.0, turns + dt)
        tx, ty = _obs_planet_future_position(obs, target, cand_turns)
        if not (0.0 <= tx <= BOARD_SIZE and 0.0 <= ty <= BOARD_SIZE):
            continue
        if _segment_min_dist_to_sun(sx, sy, tx, ty) <= SUN_RADIUS + 1.25:
            continue
        angle = math.atan2(ty - sy, tx - sx)
        if not _obs_angle_hits_target(obs, src, target, angle, ships):
            continue
        eta = max(0.0, _dist(sx, sy, tx, ty) - eff_offset) / max(speed, 1e-6)
        err = abs(eta - cand_turns)
        if best is None or err < best[0]:
            best = (err, angle)

    if best is not None:
        return best[1]
    return None


def _v5_fallback(obs):
    planets = _get(obs, "planets", []) or []
    fleets = _get(obs, "fleets", []) or []
    me = int(_get(obs, "player", 0) or 0)
    turn = int(_get(obs, "step", 0) or 0)
    ang_vel = float(_get(obs, "angular_velocity", 0.0) or 0.0)
    comet_ids = set(_get(obs, "comet_planet_ids", []) or [])
    my_planets = [p for p in planets if int(p[1]) == me]
    if not my_planets:
        return []
    my_ships = sum(float(p[5]) for p in my_planets) + sum(float(f[6]) for f in fleets if int(f[1]) == me)
    cx = sum(float(p[2]) for p in my_planets) / len(my_planets)
    cy = sum(float(p[3]) for p in my_planets) / len(my_planets)
    phase = "early" if turn < EARLY_GAME_LIMIT else ("late" if 500 - turn <= LATE_GAME_LIMIT else "mid")
    base_def = EARLY_GAME_DEFENSE if phase == "early" else (LATE_GAME_DEFENSE if phase == "late" else MID_GAME_DEFENSE)
    total_threat = 0.0
    for p in planets:
        if int(p[1]) >= 0 and int(p[1]) != me:
            total_threat += float(p[5])
    for f in fleets:
        if int(f[1]) != me:
            total_threat += float(f[6])
    if my_ships > 0 and total_threat / my_ships > THREAT_THRESHOLD:
        base_def = min(0.35, base_def + 0.10)
    available = my_ships * (1.0 - base_def)
    in_comet_window = any(abs(turn - ct) <= 15 for ct in COMET_TURNS)
    horizon = EARLY_HORIZON if phase == "early" else (LATE_HORIZON if phase == "late" else MID_HORIZON)
    target_values = []
    for p in planets:
        if int(p[1]) == me:
            continue
        value = float(p[6]) * horizon - _dist(cx, cy, float(p[2]), float(p[3])) * 0.6 - float(p[5]) * 2.0
        if int(p[0]) in comet_ids and in_comet_window:
            value += 1000.0
        if int(p[1]) == -1:
            value *= 1.15
        if value > 0:
            target_values.append((value, p))
    target_values.sort(key=lambda x: -x[0])
    orders, committed_total = [], 0
    target_committed = {}
    for src in my_planets:
        if committed_total >= available or float(src[5]) <= 1:
            continue
        for _, target in target_values:
            tid = int(target[0])
            already = target_committed.get(tid, 0)
            ships_available = min(float(src[5]) - 1, available - committed_total)
            if ships_available <= 1:
                break
            d = _dist(float(src[2]), float(src[3]), float(target[2]), float(target[3]))
            eta = d / 4.0
            enemy_ships = float(target[5])
            enemy_prod = float(target[6]) if int(target[1]) >= 0 else 0.0
            need = max(1, int(math.ceil(enemy_ships + enemy_prod * eta + 1 - already)))
            ratio_cap = 0.85 if tid in comet_ids and in_comet_window else FLEET_SEND_RATIO
            cap = int(ships_available * ratio_cap)
            ships = min(cap, max(need + 2, min(int(ships_available * 0.4), cap)))
            if ships < need or ships < 1:
                continue
            angle = _obs_intercept_angle(obs, src, target, ships)
            if angle is None:
                continue
            orders.append([int(src[0]), angle, ships])
            committed_total += ships
            target_committed[tid] = already + ships
            break
    return orders


EVALUATOR_B64 = "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"
_EVALUATOR = None


def _load_evaluator():
    if EVALUATOR_B64:
        return decode_evaluator(EVALUATOR_B64)
    return None


def agent(obs, config=None):
    global _EVALUATOR
    t_start = time.time()
    if _EVALUATOR is None:
        _EVALUATOR = _load_evaluator()
    remaining = _get(obs, "remainingOverageTime", 60.0) or 60.0
    if remaining < 5.0:
        return _v5_fallback(obs)
    time_budget = min(0.85, 0.4 + float(remaining) * 0.01)
    try:
        state = state_from_obs(obs)
        elapsed = time.time() - t_start
        if time_budget - elapsed > 0.15:
            actions = beam_search(state, time_budget - elapsed, _EVALUATOR)
            if actions:
                return actions
    except Exception:
        pass
    return _v5_fallback(obs)
