"""Orbit Wars V7 — WorldModel + arrival ledger + 2p aggression tuning."""

import math
import time
import base64
import io
import collections
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------

BOARD_SIZE = 100.0
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_R = 10.0
SUN_SAFETY = 1.5
ROTATION_LIMIT = 50.0
MAX_SPEED = 6.0
TOTAL_STEPS = 500
HORIZON = 80

P_ID, P_OWNER, P_X, P_Y, P_R, P_SHIPS, P_PROD = range(7)
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = range(7)

# Phase thresholds
EARLY_TURN_LIMIT = 40
OPENING_TURN_LIMIT = 80
LATE_REMAINING_TURNS = 60
VERY_LATE_REMAINING_TURNS = 25

# Intercept
AIM_ITERATIONS = 10
INTERCEPT_TOLERANCE = 1
LAUNCH_CLEARANCE = 0.1

# Defense
SAFE_NEUTRAL_MARGIN = 2
CONTESTED_NEUTRAL_MARGIN = 2
DOOMED_EVAC_TURN_LIMIT = 24
DOOMED_MIN_SHIPS = 8

# Scoring weights
ATTACK_COST_TURN_WEIGHT = 0.55
SNIPE_COST_TURN_WEIGHT = 0.45
INDIRECT_VALUE_SCALE = 0.15
INDIRECT_FRIENDLY_WEIGHT = 0.35
INDIRECT_NEUTRAL_WEIGHT = 0.9
INDIRECT_ENEMY_WEIGHT = 1.25

# Value multipliers
STATIC_NEUTRAL_VALUE_MULT = 1.4
STATIC_HOSTILE_VALUE_MULT = 1.55
ROTATING_OPENING_VALUE_MULT = 0.9
HOSTILE_TARGET_VALUE_MULT = 1.85
OPENING_HOSTILE_TARGET_VALUE_MULT = 1.45
SAFE_NEUTRAL_VALUE_MULT = 1.2
CONTESTED_NEUTRAL_VALUE_MULT = 0.7
EARLY_NEUTRAL_VALUE_MULT = 1.2
COMET_VALUE_MULT = 0.65
SNIPE_VALUE_MULT = 1.12
SWARM_VALUE_MULT = 1.05
FINISHING_HOSTILE_VALUE_MULT = 1.15
BEHIND_ROTATING_NEUTRAL_VALUE_MULT = 0.92

# Margin constants
NEUTRAL_MARGIN_BASE = 2
NEUTRAL_MARGIN_PROD_WEIGHT = 2
NEUTRAL_MARGIN_CAP = 8
HOSTILE_MARGIN_BASE = 3
HOSTILE_MARGIN_PROD_WEIGHT = 2
HOSTILE_MARGIN_CAP = 12
STATIC_TARGET_MARGIN = 4
CONTESTED_TARGET_MARGIN = 5
FOUR_PLAYER_TARGET_MARGIN = 2
LONG_TRAVEL_MARGIN_START = 18
LONG_TRAVEL_MARGIN_DIVISOR = 3
LONG_TRAVEL_MARGIN_CAP = 8
COMET_MARGIN_RELIEF = 6
FINISHING_HOSTILE_SEND_BONUS = 3

# Score modifiers
STATIC_TARGET_SCORE_MULT = 1.18
EARLY_STATIC_NEUTRAL_SCORE_MULT = 1.25
FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT = 0.92
DENSE_STATIC_NEUTRAL_COUNT = 4
DENSE_ROTATING_NEUTRAL_SCORE_MULT = 0.86
SNIPE_SCORE_MULT = 1.12
SWARM_SCORE_MULT = 1.06

# Opening filter
SAFE_OPENING_PROD_THRESHOLD = 4
SAFE_OPENING_TURN_LIMIT = 10
ROTATING_OPENING_MAX_TURNS = 13
ROTATING_OPENING_LOW_PROD = 2
FOUR_PLAYER_ROTATING_REACTION_GAP = 1
FOUR_PLAYER_ROTATING_SEND_RATIO = 0.72
FOUR_PLAYER_ROTATING_TURN_LIMIT = 14

# Multi-source swarms
PARTIAL_SOURCE_MIN_SHIPS = 6
MULTI_SOURCE_TOP_K = 10
MULTI_SOURCE_ETA_TOLERANCE = 2
MULTI_SOURCE_PLAN_PENALTY = 0.97
HOSTILE_SWARM_ETA_TOLERANCE = 1
THREE_SOURCE_SWARM_ENABLED = True
THREE_SOURCE_MIN_TARGET_SHIPS = 20
THREE_SOURCE_ETA_TOLERANCE = 2
THREE_SOURCE_PLAN_PENALTY = 0.94
FOUR_SOURCE_SWARM_ENABLED = True
FOUR_SOURCE_MIN_TARGET_SHIPS = 40
FOUR_SOURCE_ETA_TOLERANCE = 2
FOUR_SOURCE_PLAN_PENALTY = 0.91

# Reinforcement
REINFORCE_ENABLED = True
REINFORCE_MIN_PRODUCTION = 2
REINFORCE_MAX_TRAVEL_TURNS = 22
REINFORCE_SAFETY_MARGIN = 2
REINFORCE_VALUE_MULT = 1.35
REINFORCE_MAX_SOURCE_FRACTION = 0.75
REINFORCE_MIN_FUTURE_TURNS = 40

# Proactive defense
MULTI_ENEMY_PROACTIVE_HORIZON = 14
MULTI_ENEMY_PROACTIVE_RATIO = 0.22
MULTI_ENEMY_STACK_WINDOW = 3
PROACTIVE_DEFENSE_HORIZON = 12
PROACTIVE_DEFENSE_RATIO = 0.18

# Late game
LATE_IMMEDIATE_SHIP_VALUE = 0.6
WEAK_ENEMY_THRESHOLD = 45
ELIMINATION_BONUS = 18.0
FOLLOWUP_MIN_SHIPS = 8
LOW_VALUE_COMET_PRODUCTION = 1
LATE_CAPTURE_BUFFER = 5
VERY_LATE_CAPTURE_BUFFER = 3
COMET_MAX_CHASE_TURNS = 15

# Domination
BEHIND_DOMINATION = -0.20
AHEAD_DOMINATION = 0.18
FINISHING_DOMINATION = 0.35
FINISHING_PROD_RATIO = 1.25
AHEAD_ATTACK_MARGIN_BONUS = 0.08
BEHIND_ATTACK_MARGIN_PENALTY = 0.05
FINISHING_ATTACK_MARGIN_BONUS = 0.08

# Rear staging
REAR_SOURCE_MIN_SHIPS = 16
REAR_DISTANCE_RATIO = 1.25
REAR_STAGE_PROGRESS = 0.78
REAR_SEND_RATIO_TWO_PLAYER = 0.62
REAR_SEND_RATIO_FOUR_PLAYER = 0.7
REAR_SEND_MIN_SHIPS = 10
REAR_MAX_TRAVEL_TURNS = 40

# 4-player specifics
FOUR_PLAYER_PROACTIVE_RATIO_MULT = 0.85
FOUR_PLAYER_HOSTILE_AGGRESSION_BOOST = 1.20
FOUR_PLAYER_OPPORTUNISTIC_BOOST = 1.35
FOUR_PLAYER_OPPORTUNISTIC_GARRISON_LIMIT = 6
FOUR_PLAYER_SAFE_NEUTRAL_BOOST = 1.20
FOUR_PLAYER_WEAKEST_ENEMY_BOOST = 1.55
FOUR_PLAYER_STRONGEST_ENEMY_PENALTY = 0.85
FOUR_PLAYER_ELIMINATION_BONUS = 35.0
FOUR_PLAYER_WEAK_ENEMY_THRESHOLD = 75
FOUR_PLAYER_AGGRESSIVE_FINISHING_DOMINATION = 0.20
FOUR_PLAYER_AGGRESSIVE_FINISHING_PROD_RATIO = 0.95

# --- 2-player tuning (derived from replay analysis: slow scaling = main loss cause) ---
# More aggressive neutral expansion + hostile pressure in 2p
TWO_PLAYER_HOSTILE_AGGRESSION_BOOST = 1.35   # attack opponent harder
TWO_PLAYER_NEUTRAL_MARGIN_BASE = 1           # lower garrison margin on neutrals
TWO_PLAYER_OPENING_TURN_LIMIT = 60           # exit cautious opening mode faster
TWO_PLAYER_NEUTRAL_VALUE_MULT = 1.35         # neutrals more valuable (faster scaling)
TWO_PLAYER_SAFE_NEUTRAL_BOOST = 1.30         # chase safe neutrals aggressively


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
Fleet = namedtuple("Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])


@dataclass(frozen=True)
class ShotOption:
    score: float
    src_id: int
    target_id: int
    angle: float
    turns: int
    needed: int
    send_cap: int
    mission: str = "capture"


@dataclass
class Mission:
    kind: str
    score: float
    target_id: int
    turns: int
    options: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def is_static_planet(planet):
    r = _dist(planet.x, planet.y, CENTER_X, CENTER_Y)
    return r + planet.radius >= ROTATION_LIMIT


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(max(ships, 1)) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


def _pt_seg_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    lsq = dx * dx + dy * dy
    if lsq <= 1e-9:
        return _dist(px, py, x1, y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / lsq))
    return _dist(px, py, x1 + t * dx, y1 + t * dy)


_aim_trig_cache = {}


def _cos_sin(angle):
    key = round(angle, 5)
    cached = _aim_trig_cache.get(key)
    if cached is not None:
        return cached
    result = (math.cos(angle), math.sin(angle))
    if len(_aim_trig_cache) > 4096:
        _aim_trig_cache.clear()
    _aim_trig_cache[key] = result
    return result


_SUN_SAFETY_SQ = (SUN_R + SUN_SAFETY) ** 2


def _seg_hits_sun(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    lsq = dx * dx + dy * dy
    if lsq <= 1e-9:
        d2 = (CENTER_X - x1) ** 2 + (CENTER_Y - y1) ** 2
    else:
        t = max(0.0, min(1.0, ((CENTER_X - x1) * dx + (CENTER_Y - y1) * dy) / lsq))
        d2 = (CENTER_X - x1 - t * dx) ** 2 + (CENTER_Y - y1 - t * dy) ** 2
    return d2 < _SUN_SAFETY_SQ


def _safe_angle_and_dist(sx, sy, sr, tx, ty, tr):
    angle = math.atan2(ty - sy, tx - sx)
    ca, sa = _cos_sin(angle)
    clearance = sr + LAUNCH_CLEARANCE
    lx, ly = sx + ca * clearance, sy + sa * clearance
    hit_d = max(0.0, _dist(sx, sy, tx, ty) - clearance - tr)
    if _seg_hits_sun(lx, ly, lx + ca * hit_d, ly + sa * hit_d):
        return None
    return angle, hit_d


def _predict_planet_pos(planet, initial_by_id, angular_velocity, turns):
    init = initial_by_id.get(planet.id)
    if init is None:
        return planet.x, planet.y
    r = _dist(init.x, init.y, CENTER_X, CENTER_Y)
    if r + init.radius >= ROTATION_LIMIT:
        return planet.x, planet.y
    cur_ang = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    new_ang = cur_ang + angular_velocity * turns
    return CENTER_X + r * math.cos(new_ang), CENTER_Y + r * math.sin(new_ang)


def _predict_comet_pos(planet_id, comets, turns):
    for group in comets:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        path_index = group.get("path_index", 0)
        if idx >= len(paths):
            return None
        path = paths[idx]
        fi = path_index + int(turns)
        if 0 <= fi < len(path):
            return path[fi][0], path[fi][1]
        return None
    return None


def _comet_remaining_life(planet_id, comets):
    for group in comets:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        path_index = group.get("path_index", 0)
        if idx < len(paths):
            return max(0, len(paths[idx]) - path_index)
    return 0


def _estimate_arrival(sx, sy, sr, tx, ty, tr, ships):
    res = _safe_angle_and_dist(sx, sy, sr, tx, ty, tr)
    if res is None:
        return None
    angle, total_d = res
    turns = max(1, int(math.ceil(total_d / fleet_speed(max(1, ships)))))
    return angle, turns


def _travel_time(sx, sy, sr, tx, ty, tr, ships):
    est = _estimate_arrival(sx, sy, sr, tx, ty, tr, ships)
    return est[1] if est else 10 ** 9


def _predict_target_pos(target, turns, initial_by_id, ang_vel, comets, comet_ids):
    if target.id in comet_ids:
        return _predict_comet_pos(target.id, comets, turns)
    return _predict_planet_pos(target, initial_by_id, ang_vel, turns)


def _aim_with_prediction(src, target, ships, initial_by_id, ang_vel, comets, comet_ids):
    est = _estimate_arrival(src.x, src.y, src.radius, target.x, target.y, target.radius, ships)
    if est is None:
        return _search_safe_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids)

    tx, ty = target.x, target.y
    for _ in range(AIM_ITERATIONS):
        _, turns = est
        pos = _predict_target_pos(target, turns, initial_by_id, ang_vel, comets, comet_ids)
        if pos is None:
            return None
        ntx, nty = pos
        next_est = _estimate_arrival(src.x, src.y, src.radius, ntx, nty, target.radius, ships)
        if next_est is None:
            return _search_safe_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids)
        if abs(ntx - tx) < 0.25 and abs(nty - ty) < 0.25 and abs(next_est[1] - turns) <= INTERCEPT_TOLERANCE:
            return next_est[0], next_est[1], ntx, nty
        tx, ty = ntx, nty
        est = next_est

    final = _estimate_arrival(src.x, src.y, src.radius, tx, ty, target.radius, ships)
    if final is None:
        return _search_safe_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids)
    return final[0], final[1], tx, ty


def _search_safe_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids):
    best = None
    best_score = None
    max_turns = HORIZON
    if target.id in comet_ids:
        max_turns = min(max_turns, max(0, _comet_remaining_life(target.id, comets) - 1))

    for cand in range(1, max_turns + 1):
        pos = _predict_target_pos(target, cand, initial_by_id, ang_vel, comets, comet_ids)
        if pos is None:
            continue
        est = _estimate_arrival(src.x, src.y, src.radius, pos[0], pos[1], target.radius, ships)
        if est is None:
            continue
        _, turns = est
        if abs(turns - cand) > INTERCEPT_TOLERANCE:
            continue
        actual = max(turns, cand)
        apos = _predict_target_pos(target, actual, initial_by_id, ang_vel, comets, comet_ids)
        if apos is None:
            continue
        confirm = _estimate_arrival(src.x, src.y, src.radius, apos[0], apos[1], target.radius, ships)
        if confirm is None:
            continue
        delta = abs(confirm[1] - actual)
        if delta > INTERCEPT_TOLERANCE:
            continue
        score = (delta, confirm[1], cand)
        if best is None or score < best_score:
            best_score = score
            best = (confirm[0], confirm[1], apos[0], apos[1])
    return best


# ---------------------------------------------------------------------------
# Fleet target detection
# ---------------------------------------------------------------------------

def _fleet_target_planet(fleet, planets):
    best_planet, best_time = None, 1e9
    dir_x, dir_y = math.cos(fleet.angle), math.sin(fleet.angle)
    speed = fleet_speed(fleet.ships)

    for planet in planets:
        dx, dy = planet.x - fleet.x, planet.y - fleet.y
        proj = dx * dir_x + dy * dir_y
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        r2 = planet.radius * planet.radius
        if perp_sq >= r2:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, r2 - perp_sq)))
        turns = hit_d / speed
        if turns <= HORIZON and turns < best_time:
            best_time = turns
            best_planet = planet

    if best_planet is None:
        return None, None
    return best_planet, int(math.ceil(best_time))


# ---------------------------------------------------------------------------
# Arrival ledger + timeline simulation
# ---------------------------------------------------------------------------

def build_arrival_ledger(fleets, planets):
    arrivals = {p.id: [] for p in planets}
    for fleet in fleets:
        target, eta = _fleet_target_planet(fleet, planets)
        if target is None:
            continue
        arrivals[target.id].append((eta, fleet.owner, int(fleet.ships)))
    return arrivals


def _resolve_arrival(owner, garrison, arrivals):
    by_owner = {}
    for _, o, s in arrivals:
        by_owner[o] = by_owner.get(o, 0) + s
    if not by_owner:
        return owner, max(0.0, garrison)

    sorted_p = sorted(by_owner.items(), key=lambda kv: kv[1], reverse=True)
    top_owner, top_ships = sorted_p[0]
    if len(sorted_p) > 1:
        second = sorted_p[1][1]
        if top_ships == second:
            survivor_owner, survivor_ships = -1, 0
        else:
            survivor_owner, survivor_ships = top_owner, top_ships - second
    else:
        survivor_owner, survivor_ships = top_owner, top_ships

    if survivor_ships <= 0:
        return owner, max(0.0, garrison)
    if owner == survivor_owner:
        return owner, garrison + survivor_ships
    garrison -= survivor_ships
    if garrison < 0:
        return survivor_owner, -garrison
    return owner, garrison


def simulate_planet_timeline(planet, arrivals, player, horizon):
    horizon = max(0, int(math.ceil(horizon)))
    events = []
    for turns, owner, ships in arrivals:
        if ships <= 0:
            continue
        eta = max(1, int(math.ceil(turns)))
        if eta <= horizon:
            events.append((eta, owner, int(ships)))
    events.sort(key=lambda x: x[0])
    by_turn = defaultdict(list)
    for item in events:
        by_turn[item[0]].append(item)

    owner = planet.owner
    garrison = float(planet.ships)
    owner_at = {0: owner}
    ships_at = {0: max(0.0, garrison)}
    min_owned = garrison if owner == player else 0.0
    first_enemy = None
    fall_turn = None

    for turn in range(1, horizon + 1):
        if owner != -1:
            garrison += planet.production
        group = by_turn.get(turn, [])
        prev_owner = owner
        if group:
            if prev_owner == player and first_enemy is None:
                if any(item[1] not in (-1, player) for item in group):
                    first_enemy = turn
            owner, garrison = _resolve_arrival(owner, garrison, group)
            if prev_owner == player and owner != player and fall_turn is None:
                fall_turn = turn
        owner_at[turn] = owner
        ships_at[turn] = max(0.0, garrison)
        if owner == player:
            min_owned = min(min_owned, garrison)

    keep_needed = 0
    holds_full = True

    if planet.owner == player:
        has_enemy_arrival = any(o not in (-1, player) for _, o, _ in events)
        if not has_enemy_arrival:
            return {
                "owner_at": owner_at,
                "ships_at": ships_at,
                "keep_needed": 0,
                "min_owned": max(0, int(math.floor(min_owned))),
                "first_enemy": None,
                "fall_turn": None,
                "holds_full": True,
                "horizon": horizon,
            }

        def _survives(keep):
            sim_owner = planet.owner
            sim_gar = float(keep)
            for t in range(1, horizon + 1):
                if sim_owner != -1:
                    sim_gar += planet.production
                grp = by_turn.get(t, [])
                if grp:
                    sim_owner, sim_gar = _resolve_arrival(sim_owner, sim_gar, grp)
                    if sim_owner != player:
                        return False
            return sim_owner == player

        if _survives(int(planet.ships)):
            lo, hi = 0, int(planet.ships)
            while lo < hi:
                mid = (lo + hi) // 2
                if _survives(mid):
                    hi = mid
                else:
                    lo = mid + 1
            keep_needed = lo
        else:
            holds_full = False
            keep_needed = int(planet.ships)

    return {
        "owner_at": owner_at,
        "ships_at": ships_at,
        "keep_needed": keep_needed,
        "min_owned": max(0, int(math.floor(min_owned))) if planet.owner == player else 0,
        "first_enemy": first_enemy,
        "fall_turn": fall_turn,
        "holds_full": holds_full,
        "horizon": horizon,
    }


def _state_at_timeline(timeline, arrival_turn):
    turn = max(0, min(int(math.ceil(arrival_turn)), timeline["horizon"]))
    owner = timeline["owner_at"].get(turn, timeline["owner_at"][timeline["horizon"]])
    ships = timeline["ships_at"].get(turn, timeline["ships_at"][timeline["horizon"]])
    return owner, max(0.0, ships)


# ---------------------------------------------------------------------------
# WorldModel
# ---------------------------------------------------------------------------

def _count_players(planets, fleets):
    owners = set()
    for p in planets:
        if p.owner != -1:
            owners.add(p.owner)
    for f in fleets:
        owners.add(f.owner)
    return max(2, len(owners))


def _indirect_wealth(planet, planets, player):
    wealth = 0.0
    for other in planets:
        if other.id == planet.id:
            continue
        d = _dist(planet.x, planet.y, other.x, other.y)
        if d < 1:
            continue
        factor = other.production / (d + 12.0)
        if other.owner == player:
            wealth += factor * INDIRECT_FRIENDLY_WEIGHT
        elif other.owner == -1:
            wealth += factor * INDIRECT_NEUTRAL_WEIGHT
        else:
            wealth += factor * INDIRECT_ENEMY_WEIGHT
    return wealth


class WorldModel:
    def __init__(self, player, step, planets, fleets, initial_by_id, ang_vel, comets, comet_ids):
        self.player = player
        self.step = step
        self.planets = planets
        self.fleets = fleets
        self.initial_by_id = initial_by_id
        self.ang_vel = ang_vel
        self.comets = comets
        self.comet_ids = set(comet_ids)

        self.planet_by_id = {p.id: p for p in planets}
        self.my_planets = [p for p in planets if p.owner == player]
        self.enemy_planets = [p for p in planets if p.owner not in (-1, player)]
        self.neutral_planets = [p for p in planets if p.owner == -1]
        self.static_neutral_planets = [p for p in self.neutral_planets if is_static_planet(p)]

        self.num_players = _count_players(planets, fleets)
        self.remaining_steps = max(1, TOTAL_STEPS - step)
        self.is_four_player = self.num_players >= 4

        # 2p mode uses a tighter opening window so we expand faster
        opening_limit = OPENING_TURN_LIMIT if self.is_four_player else TWO_PLAYER_OPENING_TURN_LIMIT
        self.is_early = step < EARLY_TURN_LIMIT
        self.is_opening = step < opening_limit
        self.is_late = self.remaining_steps < LATE_REMAINING_TURNS
        self.is_very_late = self.remaining_steps < VERY_LATE_REMAINING_TURNS

        self.owner_strength = defaultdict(int)
        self.owner_production = defaultdict(int)
        for p in planets:
            if p.owner != -1:
                self.owner_strength[p.owner] += int(p.ships)
                self.owner_production[p.owner] += int(p.production)
        for f in fleets:
            self.owner_strength[f.owner] += int(f.ships)

        self.my_total = self.owner_strength.get(player, 0)
        self.enemy_total = sum(s for o, s in self.owner_strength.items() if o != player)
        self.max_enemy_strength = max((s for o, s in self.owner_strength.items() if o != player), default=0)
        self.my_prod = self.owner_production.get(player, 0)
        self.enemy_prod = sum(p for o, p in self.owner_production.items() if o != player)

        enemy_owners = sorted((s, o) for o, s in self.owner_strength.items() if o != player and o != -1)
        self.weakest_enemy_id = enemy_owners[0][1] if enemy_owners else None
        self.strongest_enemy_id = enemy_owners[-1][1] if enemy_owners else None

        self.arrivals_by_planet = build_arrival_ledger(fleets, planets)
        self.base_timeline = {
            p.id: simulate_planet_timeline(p, self.arrivals_by_planet[p.id], player, HORIZON)
            for p in planets
        }
        self.indirect_wealth_map = {p.id: _indirect_wealth(p, planets, player) for p in planets}
        self.reaction_cache = {}
        self.base_need_cache = {}
        self.aim_cache = {}

        self.reserve, self.available, self.doomed_candidates, self.threatened_candidates = \
            self._compute_defense_buffers()

    def _multi_enemy_proactive_keep(self, planet):
        if not self.enemy_planets:
            return 0
        threats = []
        for enemy in self.enemy_planets:
            eta = _travel_time(enemy.x, enemy.y, enemy.radius,
                               planet.x, planet.y, planet.radius, max(1, enemy.ships))
            if eta <= MULTI_ENEMY_PROACTIVE_HORIZON:
                threats.append((eta, int(enemy.ships)))
        if not threats:
            return 0
        threats.sort()
        best_stacked = running = left = 0
        for right in range(len(threats)):
            running += threats[right][1]
            while threats[right][0] - threats[left][0] > MULTI_ENEMY_STACK_WINDOW:
                running -= threats[left][1]
                left += 1
            best_stacked = max(best_stacked, running)
        ratio_mult = FOUR_PLAYER_PROACTIVE_RATIO_MULT if self.is_four_player else 1.0
        proactive = int(best_stacked * MULTI_ENEMY_PROACTIVE_RATIO * ratio_mult)
        legacy = 0
        for eta, ships in threats:
            if eta <= PROACTIVE_DEFENSE_HORIZON:
                legacy = max(legacy, int(ships * PROACTIVE_DEFENSE_RATIO * ratio_mult))
        return max(proactive, legacy)

    def _compute_defense_buffers(self):
        reserve = {}
        available = {}
        doomed = set()
        threatened = {}
        for planet in self.my_planets:
            tl = self.base_timeline[planet.id]
            exact_keep = tl["keep_needed"]
            proactive_keep = self._multi_enemy_proactive_keep(planet)
            reserve[planet.id] = min(int(planet.ships), max(exact_keep, proactive_keep))
            available[planet.id] = max(0, int(planet.ships) - reserve[planet.id])

            if not tl["holds_full"] and tl["fall_turn"] is not None:
                fall_turn = tl["fall_turn"]
                if fall_turn <= DOOMED_EVAC_TURN_LIMIT and planet.ships >= DOOMED_MIN_SHIPS:
                    doomed.add(planet.id)
                if (REINFORCE_ENABLED and planet.production >= REINFORCE_MIN_PRODUCTION
                        and self.remaining_steps >= REINFORCE_MIN_FUTURE_TURNS):
                    deficit_hint = 0
                    for t in range(1, fall_turn + 1):
                        if tl["owner_at"].get(t) != self.player:
                            deficit_hint = max(deficit_hint, int(math.ceil(tl["ships_at"].get(t, 0))) + 1)
                            break
                    threatened[planet.id] = {"fall_turn": fall_turn, "deficit_hint": max(1, deficit_hint)}
        return reserve, available, doomed, threatened

    def is_static(self, planet_id):
        return is_static_planet(self.planet_by_id[planet_id])

    def comet_life(self, planet_id):
        return _comet_remaining_life(planet_id, self.comets)

    def source_inventory_left(self, source_id, spent_total):
        return max(0, int(self.planet_by_id[source_id].ships) - spent_total[source_id])

    def source_attack_left(self, source_id, spent_total):
        return max(0, self.available.get(source_id, 0) - spent_total[source_id])

    def plan_shot(self, src_id, target_id, ships):
        bucket = max(1, int(ships))
        if bucket <= 8:
            key_ships = bucket
        elif bucket <= 32:
            key_ships = (bucket // 4) * 4
        elif bucket <= 128:
            key_ships = (bucket // 8) * 8
        else:
            key_ships = (bucket // 16) * 16
        cache_key = (src_id, target_id, key_ships)
        cached = self.aim_cache.get(cache_key)
        if cached is not None:
            return cached if cached != "MISS" else None
        src = self.planet_by_id[src_id]
        target = self.planet_by_id[target_id]
        result = _aim_with_prediction(src, target, ships, self.initial_by_id,
                                      self.ang_vel, self.comets, self.comet_ids)
        self.aim_cache[cache_key] = result if result is not None else "MISS"
        return result

    def reaction_times(self, target_id):
        cached = self.reaction_cache.get(target_id)
        if cached is not None:
            return cached
        target = self.planet_by_id[target_id]
        my_t = min((_travel_time(p.x, p.y, p.radius, target.x, target.y, target.radius, max(1, p.ships))
                    for p in self.my_planets), default=10 ** 9)
        enemy_t = min((_travel_time(p.x, p.y, p.radius, target.x, target.y, target.radius, max(1, p.ships))
                       for p in self.enemy_planets), default=10 ** 9)
        result = (my_t, enemy_t)
        self.reaction_cache[target_id] = result
        return result

    def projected_state(self, target_id, arrival_turn, planned_commitments=None, extra_arrivals=()):
        planned_commitments = planned_commitments or {}
        cutoff = max(1, int(math.ceil(arrival_turn)))
        if not planned_commitments.get(target_id) and not extra_arrivals:
            return _state_at_timeline(self.base_timeline[target_id], cutoff)
        arrivals = [item for item in self.arrivals_by_planet.get(target_id, []) if item[0] <= cutoff]
        arrivals.extend(item for item in planned_commitments.get(target_id, []) if item[0] <= cutoff)
        arrivals.extend(item for item in extra_arrivals if item[0] <= cutoff)
        target = self.planet_by_id[target_id]
        dyn = simulate_planet_timeline(target, arrivals, self.player, cutoff)
        return _state_at_timeline(dyn, cutoff)

    def ships_needed_to_capture(self, target_id, arrival_turn, planned_commitments=None, extra_arrivals=()):
        planned_commitments = planned_commitments or {}
        cutoff = max(1, int(math.ceil(arrival_turn)))
        cache_key = None
        if not planned_commitments.get(target_id) and not extra_arrivals:
            cache_key = (target_id, cutoff)
            if cache_key in self.base_need_cache:
                return self.base_need_cache[cache_key]
        owner_t, ships_t = self.projected_state(target_id, cutoff,
                                                 planned_commitments=planned_commitments,
                                                 extra_arrivals=extra_arrivals)
        need = 0 if owner_t == self.player else int(math.ceil(ships_t)) + 1
        if cache_key is not None:
            self.base_need_cache[cache_key] = need
        return need

    def reinforcement_needed_for(self, planet_id, arrival_turn, planned_commitments=None):
        planned_commitments = planned_commitments or {}
        arrival_turn = max(1, int(math.ceil(arrival_turn)))
        planet = self.planet_by_id[planet_id]
        if planet.owner != self.player:
            return self.ships_needed_to_capture(planet_id, arrival_turn, planned_commitments)
        arrivals = list(self.arrivals_by_planet.get(planet_id, []))
        for item in planned_commitments.get(planet_id, []):
            arrivals.append(item)
        horizon = max(arrival_turn + 5, self.base_timeline[planet_id]["horizon"])
        timeline = simulate_planet_timeline(planet, arrivals, self.player, horizon)
        worst_deficit = 0
        for t in range(arrival_turn, min(horizon, arrival_turn + 20) + 1):
            if timeline["owner_at"].get(t) != self.player:
                worst_deficit = max(worst_deficit, int(math.ceil(timeline["ships_at"].get(t, 0))) + 1)
        return worst_deficit


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _build_modes(world):
    if world.is_four_player:
        domination = (world.my_total - world.max_enemy_strength) / max(1, world.my_total + world.max_enemy_strength)
    else:
        domination = (world.my_total - world.enemy_total) / max(1, world.my_total + world.enemy_total)

    is_behind = domination < BEHIND_DOMINATION
    is_ahead = domination > AHEAD_DOMINATION
    is_dominating = is_ahead or (world.max_enemy_strength > 0 and world.my_total > world.max_enemy_strength * 1.25)

    if world.is_four_player:
        max_other_prod = max((p for o, p in world.owner_production.items()
                              if o != world.player and o != -1), default=0)
        is_finishing = (domination > FOUR_PLAYER_AGGRESSIVE_FINISHING_DOMINATION
                        and world.my_prod > max_other_prod * FOUR_PLAYER_AGGRESSIVE_FINISHING_PROD_RATIO
                        and world.step > 80)
    else:
        is_finishing = (domination > FINISHING_DOMINATION
                        and world.my_prod > world.enemy_prod * FINISHING_PROD_RATIO
                        and world.step > 100)

    attack_margin_mult = 1.0
    if is_ahead:
        attack_margin_mult += AHEAD_ATTACK_MARGIN_BONUS
    if is_behind:
        attack_margin_mult -= BEHIND_ATTACK_MARGIN_PENALTY
    if is_finishing:
        attack_margin_mult += FINISHING_ATTACK_MARGIN_BONUS

    return {
        "domination": domination,
        "is_behind": is_behind,
        "is_ahead": is_ahead,
        "is_dominating": is_dominating,
        "is_finishing": is_finishing,
        "attack_margin_mult": attack_margin_mult,
    }


def _is_safe_neutral(target, world):
    if target.owner != -1:
        return False
    my_t, enemy_t = world.reaction_times(target.id)
    return my_t <= enemy_t - SAFE_NEUTRAL_MARGIN


def _is_contested_neutral(target, world):
    if target.owner != -1:
        return False
    my_t, enemy_t = world.reaction_times(target.id)
    return abs(my_t - enemy_t) <= CONTESTED_NEUTRAL_MARGIN


def _opening_filter(target, arrival_turns, needed, src_available, world):
    if not world.is_opening or target.owner != -1:
        return False
    if target.id in world.comet_ids or world.is_static(target.id):
        return False
    my_t, enemy_t = world.reaction_times(target.id)
    reaction_gap = enemy_t - my_t
    if (target.production >= SAFE_OPENING_PROD_THRESHOLD
            and arrival_turns <= SAFE_OPENING_TURN_LIMIT
            and reaction_gap >= SAFE_NEUTRAL_MARGIN):
        return False
    if world.is_four_player:
        affordable = needed <= max(PARTIAL_SOURCE_MIN_SHIPS, int(src_available * FOUR_PLAYER_ROTATING_SEND_RATIO))
        if affordable and arrival_turns <= FOUR_PLAYER_ROTATING_TURN_LIMIT and reaction_gap >= FOUR_PLAYER_ROTATING_REACTION_GAP:
            return False
        return True
    return arrival_turns > ROTATING_OPENING_MAX_TURNS or target.production <= ROTATING_OPENING_LOW_PROD


def _target_value(target, arrival_turns, mission, world, modes):
    turns_profit = max(1, world.remaining_steps - arrival_turns)
    if target.id in world.comet_ids:
        life = world.comet_life(target.id)
        turns_profit = max(0, min(turns_profit, life - arrival_turns))
        if turns_profit <= 0:
            return -1.0

    value = target.production * turns_profit
    value += world.indirect_wealth_map[target.id] * turns_profit * INDIRECT_VALUE_SCALE

    if world.is_static(target.id):
        value *= STATIC_NEUTRAL_VALUE_MULT if target.owner == -1 else STATIC_HOSTILE_VALUE_MULT
    elif world.is_opening:
        value *= ROTATING_OPENING_VALUE_MULT

    if target.owner not in (-1, world.player):
        value *= OPENING_HOSTILE_TARGET_VALUE_MULT if world.is_opening else HOSTILE_TARGET_VALUE_MULT

    if target.owner == -1:
        neutral_mult = TWO_PLAYER_NEUTRAL_VALUE_MULT if not world.is_four_player else 1.0
        value *= neutral_mult
        if _is_safe_neutral(target, world):
            safe_boost = TWO_PLAYER_SAFE_NEUTRAL_BOOST if not world.is_four_player else SAFE_NEUTRAL_VALUE_MULT
            value *= safe_boost
        elif _is_contested_neutral(target, world):
            value *= CONTESTED_NEUTRAL_VALUE_MULT
        if world.is_early:
            value *= EARLY_NEUTRAL_VALUE_MULT

    if target.id in world.comet_ids:
        value *= COMET_VALUE_MULT

    if mission == "snipe":
        value *= SNIPE_VALUE_MULT
    elif mission == "swarm":
        value *= SWARM_VALUE_MULT
    elif mission == "reinforce":
        value *= REINFORCE_VALUE_MULT

    if world.is_late:
        value += max(0, target.ships) * LATE_IMMEDIATE_SHIP_VALUE
        if target.owner not in (-1, world.player):
            es = world.owner_strength.get(target.owner, 0)
            threshold = FOUR_PLAYER_WEAK_ENEMY_THRESHOLD if world.is_four_player else WEAK_ENEMY_THRESHOLD
            bonus = FOUR_PLAYER_ELIMINATION_BONUS if world.is_four_player else ELIMINATION_BONUS
            if es <= threshold:
                value += bonus

    if modes["is_finishing"] and target.owner not in (-1, world.player):
        value *= FINISHING_HOSTILE_VALUE_MULT
    if modes["is_behind"] and target.owner == -1 and not world.is_static(target.id):
        value *= BEHIND_ROTATING_NEUTRAL_VALUE_MULT
    if modes["is_behind"] and target.owner == -1 and _is_safe_neutral(target, world):
        value *= 1.08
    if modes["is_dominating"] and target.owner == -1 and _is_contested_neutral(target, world):
        value *= 0.92

    if world.is_four_player:
        if target.owner not in (-1, world.player):
            if not world.is_opening:
                value *= FOUR_PLAYER_HOSTILE_AGGRESSION_BOOST
            if int(target.ships) <= FOUR_PLAYER_OPPORTUNISTIC_GARRISON_LIMIT:
                value *= FOUR_PLAYER_OPPORTUNISTIC_BOOST
            if target.owner == world.weakest_enemy_id:
                value *= FOUR_PLAYER_WEAKEST_ENEMY_BOOST
            elif target.owner == world.strongest_enemy_id and not world.is_late:
                value *= FOUR_PLAYER_STRONGEST_ENEMY_PENALTY
        elif target.owner == -1 and _is_safe_neutral(target, world):
            value *= FOUR_PLAYER_SAFE_NEUTRAL_BOOST
    else:
        # 2p: boost attack on enemy planets
        if target.owner not in (-1, world.player) and not world.is_opening:
            value *= TWO_PLAYER_HOSTILE_AGGRESSION_BOOST

    return value


def _preferred_send(target, base_needed, arrival_turns, src_available, world, modes):
    send = max(base_needed, int(math.ceil(base_needed * modes["attack_margin_mult"])))
    if target.owner == -1:
        margin_base = TWO_PLAYER_NEUTRAL_MARGIN_BASE if not world.is_four_player else NEUTRAL_MARGIN_BASE
        margin = min(NEUTRAL_MARGIN_CAP, margin_base + target.production * NEUTRAL_MARGIN_PROD_WEIGHT)
    else:
        margin = min(HOSTILE_MARGIN_CAP, HOSTILE_MARGIN_BASE + target.production * HOSTILE_MARGIN_PROD_WEIGHT)
    if world.is_static(target.id):
        margin += STATIC_TARGET_MARGIN
    if _is_contested_neutral(target, world):
        margin += CONTESTED_TARGET_MARGIN
    if world.is_four_player:
        margin += FOUR_PLAYER_TARGET_MARGIN
    if arrival_turns > LONG_TRAVEL_MARGIN_START:
        margin += min(LONG_TRAVEL_MARGIN_CAP, arrival_turns // LONG_TRAVEL_MARGIN_DIVISOR)
    if target.id in world.comet_ids:
        margin = max(0, margin - COMET_MARGIN_RELIEF)
    if modes["is_finishing"] and target.owner not in (-1, world.player):
        margin += FINISHING_HOSTILE_SEND_BONUS
    return min(src_available, send + margin)


def _apply_score_mods(base_score, target, mission, world):
    score = base_score
    if world.is_static(target.id):
        score *= STATIC_TARGET_SCORE_MULT
    if world.is_early and target.owner == -1 and world.is_static(target.id):
        score *= EARLY_STATIC_NEUTRAL_SCORE_MULT
    if world.is_four_player and target.owner == -1 and not world.is_static(target.id):
        score *= FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT
    if (len(world.static_neutral_planets) >= DENSE_STATIC_NEUTRAL_COUNT
            and target.owner == -1 and not world.is_static(target.id)):
        score *= DENSE_ROTATING_NEUTRAL_SCORE_MULT
    if mission == "snipe":
        score *= SNIPE_SCORE_MULT
    elif mission == "swarm":
        score *= SWARM_SCORE_MULT
    return score


# ---------------------------------------------------------------------------
# Mission builders
# ---------------------------------------------------------------------------

def _build_snipe_mission(src, target, src_available, world, planned_commitments, modes):
    if target.owner != -1:
        return None
    enemy_etas = sorted({
        int(math.ceil(eta))
        for eta, owner, ships in world.arrivals_by_planet.get(target.id, [])
        if owner not in (-1, world.player) and ships > 0
    })
    if not enemy_etas:
        return None
    probe = min(src_available, max(PARTIAL_SOURCE_MIN_SHIPS, int(target.ships) + 8))
    rough = world.plan_shot(src.id, target.id, probe)
    if rough is None:
        return None
    for enemy_eta in enemy_etas[:4]:
        if abs(rough[1] - enemy_eta) > 1:
            continue
        sync_turn = max(rough[1], enemy_eta)
        if target.id in world.comet_ids:
            life = world.comet_life(target.id)
            if sync_turn >= life or sync_turn > COMET_MAX_CHASE_TURNS:
                continue
        need = world.ships_needed_to_capture(target.id, sync_turn, planned_commitments)
        if need <= 0 or need > src_available:
            continue
        final = world.plan_shot(src.id, target.id, need)
        if final is None:
            continue
        angle, turns, _, _ = final
        if abs(turns - enemy_eta) > 1:
            continue
        sync_turn = max(turns, enemy_eta)
        need = world.ships_needed_to_capture(target.id, sync_turn, planned_commitments)
        if need <= 0 or need > src_available:
            continue
        value = _target_value(target, sync_turn, "snipe", world, modes)
        if value <= 0:
            continue
        score = _apply_score_mods(value / (need + sync_turn * SNIPE_COST_TURN_WEIGHT + 1.0),
                                   target, "snipe", world)
        option = ShotOption(score=score, src_id=src.id, target_id=target.id,
                             angle=angle, turns=turns, needed=need, send_cap=need, mission="snipe")
        return Mission(kind="snipe", score=score, target_id=target.id, turns=sync_turn, options=[option])
    return None


def _build_reinforcement_missions(world, planned_commitments, modes, source_budget_fn):
    if not REINFORCE_ENABLED or not world.threatened_candidates:
        return []
    missions = []
    for target_id, info in world.threatened_candidates.items():
        target = world.planet_by_id[target_id]
        fall_turn = info["fall_turn"]
        if fall_turn is None or fall_turn > REINFORCE_MAX_TRAVEL_TURNS + 5:
            continue
        best_mission = None
        for src in world.my_planets:
            if src.id == target_id:
                continue
            budget = source_budget_fn(src.id)
            if budget <= 0:
                continue
            source_cap = min(budget, int(src.ships * REINFORCE_MAX_SOURCE_FRACTION))
            if source_cap <= 0:
                continue
            probe = min(max(PARTIAL_SOURCE_MIN_SHIPS, int(info["deficit_hint"]) + REINFORCE_SAFETY_MARGIN), source_cap)
            if probe <= 0:
                continue
            aim = world.plan_shot(src.id, target.id, probe)
            if aim is None:
                continue
            angle, turns, _, _ = aim
            if turns > REINFORCE_MAX_TRAVEL_TURNS or turns > fall_turn:
                continue
            need = world.reinforcement_needed_for(target_id, turns, planned_commitments)
            if need <= 0:
                continue
            send = min(source_cap, need + REINFORCE_SAFETY_MARGIN)
            if send < need:
                continue
            final = world.plan_shot(src.id, target.id, send)
            if final is None:
                continue
            angle, turns, _, _ = final
            if turns > fall_turn:
                continue
            value = _target_value(target, turns, "reinforce", world, modes)
            if value <= 0:
                continue
            score = value / (send + turns * 0.35 + 1.0)
            option = ShotOption(score=score, src_id=src.id, target_id=target_id,
                                 angle=angle, turns=turns, needed=need, send_cap=send, mission="reinforce")
            mission = Mission(kind="reinforce", score=score, target_id=target_id, turns=turns, options=[option])
            if best_mission is None or mission.score > best_mission.score:
                best_mission = mission
        if best_mission is not None:
            missions.append(best_mission)
    return missions


# ---------------------------------------------------------------------------
# Main planning loop
# ---------------------------------------------------------------------------

def plan_moves(world):
    modes = _build_modes(world)
    planned_commitments = defaultdict(list)
    source_options_by_target = defaultdict(list)
    missions = []
    moves = []
    spent_total = defaultdict(int)

    def inv_left(sid):
        return world.source_inventory_left(sid, spent_total)

    def atk_left(sid):
        return world.source_attack_left(sid, spent_total)

    def append_move(src_id, angle, ships):
        send = min(int(ships), inv_left(src_id))
        if send < 1:
            return 0
        moves.append([src_id, float(angle), int(send)])
        spent_total[src_id] += send
        return send

    missions.extend(_build_reinforcement_missions(world, planned_commitments, modes, inv_left))

    for src in world.my_planets:
        src_available = atk_left(src.id)
        if src_available <= 0:
            continue
        for target in world.planets:
            if target.id == src.id or target.owner == world.player:
                continue
            rough_ships = max(1, min(src_available, max(PARTIAL_SOURCE_MIN_SHIPS, int(target.ships) + 1)))
            rough_aim = world.plan_shot(src.id, target.id, rough_ships)
            if rough_aim is None:
                continue
            rough_turns = rough_aim[1]
            if world.is_very_late and rough_turns > world.remaining_steps - VERY_LATE_CAPTURE_BUFFER:
                continue
            if target.id in world.comet_ids:
                life = world.comet_life(target.id)
                if rough_turns >= life or rough_turns > COMET_MAX_CHASE_TURNS:
                    continue
            rough_needed = world.ships_needed_to_capture(target.id, rough_turns, planned_commitments)
            if rough_needed <= 0:
                continue
            if _opening_filter(target, rough_turns, rough_needed, src_available, world):
                continue
            send_guess = _preferred_send(target, rough_needed, rough_turns, src_available, world, modes)
            aim = world.plan_shot(src.id, target.id, max(1, send_guess))
            if aim is None:
                continue
            angle, turns, _, _ = aim
            if world.is_very_late and turns > world.remaining_steps - VERY_LATE_CAPTURE_BUFFER:
                continue
            if target.id in world.comet_ids:
                life = world.comet_life(target.id)
                if turns >= life or turns > COMET_MAX_CHASE_TURNS:
                    continue
            needed = world.ships_needed_to_capture(target.id, turns, planned_commitments)
            if needed <= 0:
                continue
            if _opening_filter(target, turns, needed, src_available, world):
                continue
            send_cap = min(src_available, _preferred_send(target, needed, turns, src_available, world, modes))
            if send_cap < 1:
                continue
            if send_cap < needed and send_cap < PARTIAL_SOURCE_MIN_SHIPS:
                continue
            value = _target_value(target, turns, "capture", world, modes)
            if value <= 0:
                continue
            expected_send = max(needed, min(send_cap, _preferred_send(target, needed, turns, send_cap, world, modes)))
            score = _apply_score_mods(value / (expected_send + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                                       target, "capture", world)
            option = ShotOption(score=score, src_id=src.id, target_id=target.id,
                                 angle=angle, turns=turns, needed=needed, send_cap=send_cap, mission="capture")
            source_options_by_target[target.id].append(option)
            if send_cap >= needed:
                missions.append(Mission(kind="single", score=score, target_id=target.id,
                                        turns=turns, options=[option]))
            snipe = _build_snipe_mission(src, target, src_available, world, planned_commitments, modes)
            if snipe is not None:
                missions.append(snipe)

    for target_id, options in source_options_by_target.items():
        if len(options) < 2:
            continue
        target = world.planet_by_id[target_id]
        top = sorted(options, key=lambda o: -o.score)[:MULTI_SOURCE_TOP_K]
        hostile_target = target.owner not in (-1, world.player)
        eta_tol = HOSTILE_SWARM_ETA_TOLERANCE if hostile_target else MULTI_SOURCE_ETA_TOLERANCE

        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                a, b = top[i], top[j]
                if a.src_id == b.src_id or abs(a.turns - b.turns) > eta_tol:
                    continue
                jt = max(a.turns, b.turns)
                need = world.ships_needed_to_capture(target_id, jt, planned_commitments)
                if need <= 0 or a.send_cap >= need or b.send_cap >= need:
                    continue
                if a.send_cap + b.send_cap < need:
                    continue
                value = _target_value(target, jt, "swarm", world, modes)
                if value <= 0:
                    continue
                ps = _apply_score_mods(value / (need + jt * ATTACK_COST_TURN_WEIGHT + 1.0),
                                        target, "swarm", world) * MULTI_SOURCE_PLAN_PENALTY
                missions.append(Mission(kind="swarm", score=ps, target_id=target_id, turns=jt, options=[a, b]))

        if THREE_SOURCE_SWARM_ENABLED and hostile_target and int(target.ships) >= THREE_SOURCE_MIN_TARGET_SHIPS and len(top) >= 3:
            limit = min(len(top), 8)
            for i in range(limit):
                for j in range(i + 1, limit):
                    for k in range(j + 1, limit):
                        trio = [top[i], top[j], top[k]]
                        if len({o.src_id for o in trio}) < 3:
                            continue
                        if max(o.turns for o in trio) - min(o.turns for o in trio) > THREE_SOURCE_ETA_TOLERANCE:
                            continue
                        jt = max(o.turns for o in trio)
                        need = world.ships_needed_to_capture(target_id, jt, planned_commitments)
                        if need <= 0 or sum(o.send_cap for o in trio) < need:
                            continue
                        if any(trio[a].send_cap + trio[b].send_cap >= need for a in range(3) for b in range(a+1, 3)):
                            continue
                        value = _target_value(target, jt, "swarm", world, modes)
                        if value <= 0:
                            continue
                        ts = _apply_score_mods(value / (need + jt * ATTACK_COST_TURN_WEIGHT + 1.0),
                                                target, "swarm", world) * THREE_SOURCE_PLAN_PENALTY
                        missions.append(Mission(kind="swarm", score=ts, target_id=target_id, turns=jt, options=trio))

    missions.sort(key=lambda m: -m.score)

    for mission in missions:
        target = world.planet_by_id[mission.target_id]
        if mission.kind in ("single", "snipe", "reinforce"):
            option = mission.options[0]
            left = inv_left(option.src_id) if mission.kind == "reinforce" else atk_left(option.src_id)
            if left <= 0:
                continue
            if mission.kind == "reinforce":
                missing = world.reinforcement_needed_for(option.target_id, option.turns, planned_commitments)
            else:
                missing = world.ships_needed_to_capture(target.id, option.turns, planned_commitments)
            if missing <= 0:
                continue
            send_limit = min(left, option.send_cap)
            if send_limit < missing:
                continue
            if mission.kind == "snipe":
                send = missing
            elif mission.kind == "reinforce":
                send = min(send_limit, missing + REINFORCE_SAFETY_MARGIN)
            else:
                send = min(send_limit, max(missing, _preferred_send(target, missing, option.turns, send_limit, world, modes)))
            if send < missing:
                continue
            sent = append_move(option.src_id, option.angle, send)
            if sent < missing:
                continue
            planned_commitments[target.id].append((option.turns, world.player, int(sent)))
            continue

        limits = [min(atk_left(o.src_id), o.send_cap) for o in mission.options]
        if min(limits) <= 0:
            continue
        missing = world.ships_needed_to_capture(target.id, mission.turns, planned_commitments)
        if missing <= 0 or sum(limits) < missing:
            continue
        ordered = sorted(zip(mission.options, limits), key=lambda x: (x[0].turns, -x[1], x[0].src_id))
        remaining = missing
        sends = {}
        for idx, (option, limit) in enumerate(ordered):
            remaining_other = sum(lim for _, lim in ordered[idx+1:])
            send = min(limit, max(0, remaining - remaining_other))
            sends[option.src_id] = send
            remaining -= send
        if remaining > 0:
            continue
        committed = []
        for option, _ in ordered:
            send = sends.get(option.src_id, 0)
            if send <= 0:
                continue
            actual = append_move(option.src_id, option.angle, send)
            if actual > 0:
                committed.append((option.turns, world.player, int(actual)))
        if sum(item[2] for item in committed) < missing:
            continue
        planned_commitments[target.id].extend(committed)

    # Follow-up pass for leftover attack capacity
    if not world.is_very_late:
        for src in world.my_planets:
            src_left = atk_left(src.id)
            if src_left < FOLLOWUP_MIN_SHIPS:
                continue
            best = None
            for target in world.planets:
                if target.id == src.id or target.owner == world.player:
                    continue
                if target.id in world.comet_ids and target.production <= LOW_VALUE_COMET_PRODUCTION:
                    continue
                rough_ships = max(1, min(src_left, max(PARTIAL_SOURCE_MIN_SHIPS, int(target.ships) + 1)))
                rough_aim = world.plan_shot(src.id, target.id, rough_ships)
                if rough_aim is None:
                    continue
                est_turns = rough_aim[1]
                if world.is_late and est_turns > world.remaining_steps - LATE_CAPTURE_BUFFER:
                    continue
                rough_needed = world.ships_needed_to_capture(target.id, est_turns, planned_commitments)
                if rough_needed <= 0:
                    continue
                if _opening_filter(target, est_turns, rough_needed, src_left, world):
                    continue
                send = _preferred_send(target, rough_needed, est_turns, src_left, world, modes)
                if send < rough_needed:
                    continue
                value = _target_value(target, est_turns, "capture", world, modes)
                if value <= 0:
                    continue
                score = _apply_score_mods(value / (send + est_turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                                           target, "capture", world)
                if best is None or score > best[0]:
                    best = (score, target, send)
            if best is None:
                continue
            _, target, send = best
            aim = world.plan_shot(src.id, target.id, send)
            if aim is None:
                continue
            angle, turns, _, _ = aim
            missing = world.ships_needed_to_capture(target.id, turns, planned_commitments)
            if missing <= 0:
                continue
            src_left = atk_left(src.id)
            send = min(src_left, max(missing, _preferred_send(target, missing, turns, src_left, world, modes)))
            if send < missing:
                continue
            actual = append_move(src.id, angle, send)
            if actual >= missing:
                planned_commitments[target.id].append((turns, world.player, int(actual)))

    # Doomed planet evacuation
    if world.doomed_candidates:
        frontier_targets = world.enemy_planets or world.static_neutral_planets or world.neutral_planets
        if frontier_targets:
            frontier_distance = {p.id: min(_dist(p.x, p.y, t.x, t.y) for t in frontier_targets)
                                  for p in world.my_planets}
        else:
            frontier_distance = {p.id: 10 ** 9 for p in world.my_planets}

        for planet in world.my_planets:
            if planet.id not in world.doomed_candidates:
                continue
            if planned_commitments.get(planet.id):
                if sum(s for _, o, s in planned_commitments[planet.id] if o == world.player) > 0:
                    continue
            available_now = inv_left(planet.id)
            if available_now < world.reserve.get(planet.id, 0):
                continue
            best_capture = None
            for target in world.planets:
                if target.id == planet.id or target.owner == world.player:
                    continue
                probe_aim = world.plan_shot(planet.id, target.id, available_now)
                if probe_aim is None:
                    continue
                probe_turns = probe_aim[1]
                if probe_turns > world.remaining_steps - 2:
                    continue
                need = world.ships_needed_to_capture(target.id, probe_turns, planned_commitments)
                if need <= 0 or need > available_now:
                    continue
                final_aim = world.plan_shot(planet.id, target.id, need)
                if final_aim is None:
                    continue
                angle, turns, _, _ = final_aim
                score = _target_value(target, turns, "capture", world, modes) / (need + turns + 1.0)
                if target.owner not in (-1, world.player):
                    score *= 1.05
                if best_capture is None or score > best_capture[0]:
                    best_capture = (score, target.id, angle, turns, need)
            if best_capture is not None:
                _, tid, angle, turns, need = best_capture
                actual = append_move(planet.id, angle, need)
                if actual >= 1:
                    planned_commitments[tid].append((turns, world.player, int(actual)))
                continue
            safe_allies = [a for a in world.my_planets if a.id != planet.id and a.id not in world.doomed_candidates]
            if not safe_allies:
                continue
            retreat = min(safe_allies, key=lambda a: (frontier_distance.get(a.id, 10**9), _dist(planet.x, planet.y, a.x, a.y)))
            aim = world.plan_shot(planet.id, retreat.id, available_now)
            if aim is None:
                continue
            append_move(planet.id, aim[0], available_now)

    # Rear-to-front staging
    if (world.enemy_planets or world.neutral_planets) and len(world.my_planets) > 1 and not world.is_late:
        frontier_targets = world.enemy_planets or world.static_neutral_planets or world.neutral_planets
        frontier_distance = {p.id: min(_dist(p.x, p.y, t.x, t.y) for t in frontier_targets)
                              for p in world.my_planets}
        safe_fronts = [p for p in world.my_planets if p.id not in world.doomed_candidates]
        if safe_fronts:
            front_anchor = min(safe_fronts, key=lambda p: frontier_distance[p.id])
            send_ratio = REAR_SEND_RATIO_FOUR_PLAYER if world.is_four_player else REAR_SEND_RATIO_TWO_PLAYER
            if modes["is_finishing"]:
                send_ratio = max(send_ratio, REAR_SEND_RATIO_FOUR_PLAYER)
            for rear in sorted(world.my_planets, key=lambda p: -frontier_distance[p.id]):
                if rear.id == front_anchor.id or rear.id in world.doomed_candidates:
                    continue
                if atk_left(rear.id) < REAR_SOURCE_MIN_SHIPS:
                    continue
                if frontier_distance[rear.id] < frontier_distance[front_anchor.id] * REAR_DISTANCE_RATIO:
                    continue
                stage_candidates = [p for p in safe_fronts if p.id != rear.id
                                     and frontier_distance[p.id] < frontier_distance[rear.id] * REAR_STAGE_PROGRESS]
                if stage_candidates:
                    front = min(stage_candidates, key=lambda p: _dist(rear.x, rear.y, p.x, p.y))
                else:
                    obj = min(frontier_targets, key=lambda t: _dist(rear.x, rear.y, t.x, t.y))
                    rem = [p for p in safe_fronts if p.id != rear.id]
                    if not rem:
                        continue
                    front = min(rem, key=lambda p: _dist(p.x, p.y, obj.x, obj.y))
                if front.id == rear.id:
                    continue
                send = int(atk_left(rear.id) * send_ratio)
                if send < REAR_SEND_MIN_SHIPS:
                    continue
                aim = world.plan_shot(rear.id, front.id, send)
                if aim is None:
                    continue
                if aim[1] > REAR_MAX_TRAVEL_TURNS:
                    continue
                append_move(rear.id, aim[0], send)

    # Final deduplication
    final_moves = []
    used_final = defaultdict(int)
    for src_id, angle, ships in moves:
        source = world.planet_by_id[src_id]
        max_allowed = int(source.ships) - used_final[src_id]
        send = min(int(ships), max_allowed)
        if send >= 1:
            final_moves.append([src_id, float(angle), int(send)])
            used_final[src_id] += send
    return final_moves


# ---------------------------------------------------------------------------
# NumpyEvaluator — kept for future REINFORCE training
# ---------------------------------------------------------------------------

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
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2,
                                self.W3.ravel(), self.b3])

    def set_params(self, params):
        params = params.astype(np.float32)
        i = 0
        for attr, shape in [("W1", (24, 32)), ("b1", (32,)), ("W2", (32, 16)),
                             ("b2", (16,)), ("W3", (16, 1)), ("b3", (1,))]:
            size = int(np.prod(shape))
            setattr(self, attr, params[i:i + size].reshape(shape))
            i += size


# ---------------------------------------------------------------------------
# Obs parsing
# ---------------------------------------------------------------------------

def _get(o, k, d=None):
    return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)


def _build_world(obs):
    player = int(_get(obs, "player", 0) or 0)
    step = int(_get(obs, "step", 0) or 0)
    raw_planets = list(_get(obs, "planets", []) or [])
    raw_fleets = list(_get(obs, "fleets", []) or [])
    raw_init = list(_get(obs, "initial_planets", []) or [])
    raw_comets = list(_get(obs, "comets", []) or [])
    comet_ids = set(int(x) for x in (_get(obs, "comet_planet_ids", []) or []))
    ang_vel = float(_get(obs, "angular_velocity", 0.0) or 0.0)

    planets = [Planet(*[float(v) if i >= 1 else int(v) for i, v in enumerate(p)]) for p in raw_planets]
    # Fix types: id int, owner int, x/y/r/ships/prod float
    planets = [Planet(int(p[0]), int(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6]))
               for p in raw_planets] if raw_planets else []
    fleets = [Fleet(int(f[0]), int(f[1]), float(f[2]), float(f[3]), float(f[4]), int(f[5]), float(f[6]))
              for f in raw_fleets] if raw_fleets else []
    init_planets = [Planet(int(p[0]), int(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6]))
                    for p in raw_init] if raw_init else planets
    initial_by_id = {p.id: p for p in init_planets}

    comets = []
    for g in raw_comets:
        pids = [int(x) for x in (_get(g, "planet_ids", []) or [])]
        paths = _get(g, "paths", []) or []
        idx = int(_get(g, "path_index", 0) or 0)
        comets.append({"planet_ids": pids, "paths": paths, "path_index": idx})

    return WorldModel(player=player, step=step, planets=planets, fleets=fleets,
                      initial_by_id=initial_by_id, ang_vel=ang_vel, comets=comets, comet_ids=comet_ids)


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

def agent(obs, config=None):
    try:
        world = _build_world(obs)
        if not world.my_planets:
            return []
        return plan_moves(world)
    except Exception:
        return []
