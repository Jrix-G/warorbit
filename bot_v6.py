"""Orbit Wars V6 single-file Kaggle agent."""

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


def _angle_to(src, target):
    return math.atan2(float(target[P_Y]) - float(src[P_Y]), float(target[P_X]) - float(src[P_X]))


def _heuristic_actions(state, player):
    my_planets = _planet_rows(state, player)
    targets = state.planets[state.planets[:, P_OWNER] != player] if len(state.planets) else np.zeros((0, 7), dtype=np.float32)
    if len(my_planets) == 0 or len(targets) == 0:
        return []
    actions = []
    for src in my_planets:
        ships = int(src[P_SHIPS])
        if ships <= 5:
            continue
        d = np.hypot(targets[:, P_X] - src[P_X], targets[:, P_Y] - src[P_Y])
        score = targets[:, P_PROD] * 30.0 - d * 0.5
        target = targets[int(np.argmax(score))]
        actions.append([int(src[P_ID]), _angle_to(src, target), int(ships * 0.6)])
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
                out.append([int(from_id), float(angle) + float(np.random.uniform(-0.2, 0.2)),
                            min(avail, max(1, int(float(ships) * float(np.random.uniform(0.8, 1.2)))))])
        if out:
            return out
        strategy = 0
    actions = []
    for src in my_planets:
        ships = int(src[P_SHIPS])
        if ships <= 5:
            continue
        if strategy == 0:
            ratio = float(np.random.uniform(0.65, 0.80))
        elif strategy == 1:
            ratio = float(np.random.uniform(0.30, 0.50))
        elif strategy == 2:
            ratio = float(np.random.uniform(0.50, 0.65))
        elif strategy == 3:
            ratio = 0.78 if any(abs(state.step - ct) < 15 for ct in COMET_TURNS) else 0.60
        else:
            ratio = 0.65
        target = _best_target(state, src, me, strategy)
        if target is not None:
            actions.append([int(src[P_ID]), _angle_to(src, target), int(ships * ratio)])
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
FLEET_SEND_RATIO = 0.65
THREAT_THRESHOLD = 0.25
EARLY_GAME_DEFENSE = 0.25
MID_GAME_DEFENSE = 0.20
LATE_GAME_DEFENSE = 0.15
EARLY_HORIZON = 40
MID_HORIZON = 100
LATE_HORIZON = 180
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
    orders, committed = [], 0
    for src in my_planets:
        if committed >= available or float(src[5]) <= 1:
            continue
        for _, target in target_values:
            ships_available = min(float(src[5]), available - committed)
            if ships_available <= 1:
                break
            ratio = 0.70 if int(target[0]) in comet_ids and in_comet_window else FLEET_SEND_RATIO
            ships = int(ships_available * ratio)
            if ships >= 1:
                orders.append([int(src[0]), _safe_angle(float(src[2]), float(src[3]), float(target[2]), float(target[3])), ships])
                committed += ships
                break
    return orders


EVALUATOR_B64 = ""
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
            return beam_search(state, time_budget - elapsed, _EVALUATOR)
    except Exception:
        pass
    return _v5_fallback(obs)
