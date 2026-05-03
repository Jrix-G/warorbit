"""Orbit Wars submission — standalone V8.5.

Generated from bot_v7.py + bot_v8_5.py so the Kaggle submission entry point
matches bot_v8_5.agent while remaining single-file/self-contained.
"""

import math
import time
import base64
import io
import collections
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from pathlib import Path

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
SAFE_OPENING_PROD_THRESHOLD = 2
SAFE_OPENING_TURN_LIMIT = 12
ROTATING_OPENING_MAX_TURNS = 15
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
REAR_SOURCE_MIN_SHIPS = 18
REAR_DISTANCE_RATIO = 1.25
REAR_STAGE_PROGRESS = 0.78
REAR_SEND_RATIO_TWO_PLAYER = 0.58
REAR_SEND_RATIO_FOUR_PLAYER = 0.7
REAR_SEND_MIN_SHIPS = 10
REAR_MAX_TRAVEL_TURNS = 55

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
TWO_PLAYER_HOSTILE_AGGRESSION_BOOST = 1.50   # attack opponent harder
TWO_PLAYER_NEUTRAL_MARGIN_BASE = 0           # lower garrison margin on neutrals
TWO_PLAYER_NEUTRAL_MARGIN_PROD_WEIGHT = 1    # keep 2p neutral sends thin
TWO_PLAYER_NEUTRAL_MARGIN_CAP = 5
TWO_PLAYER_OPENING_TURN_LIMIT = 60           # exit cautious opening mode faster
TWO_PLAYER_NEUTRAL_VALUE_MULT = 1.45         # neutrals more valuable (faster scaling)
TWO_PLAYER_SAFE_NEUTRAL_BOOST = 1.40         # chase safe neutrals aggressively

# --- Art of War rules (derived from 95-game leader analysis) ---
# Rule 1: opening — opening filter already exits at t60 in 2p; tighten early prod filter
AOW_OPENING_MIN_PROD = 1              # was 2 — kovi (top1) takes prod=1 at ~20% in opening; filtering them costs us territory + denial
AOW_EARLY_FLOOD_TURNS = 25           # before this step: minimal margins, capture everything
AOW_EARLY_FLOOD_MARGIN = 0           # flood uses the bare minimum needed to capture
# Rule 2: alarm if < 8 planets at t035-t050 in 2p
AOW_ALARM_START = 35
AOW_ALARM_END = 50
AOW_ALARM_PLANET_THRESHOLD = 8
AOW_ALARM_NEUTRAL_BOOST = 1.55        # panic expansion multiplier
AOW_ALARM_MARGIN_BASE = 0             # drop garrison requirement completely
# Rule 3: conversion push — neutral bonus while below 15 planets and step < 80
AOW_CONVERSION_THRESHOLD = 15
AOW_CONVERSION_TURN_LIMIT = 80
AOW_CONVERSION_NEUTRAL_BOOST = 1.45
# Rule 4: anti-hoarding — force staging from idle planets
AOW_HOARD_PROD_RATIO = 18             # ships > prod * this → considered hoarding
AOW_HOARD_SEND_RATIO = 0.50           # fraction to push toward front
# Rule 5: production priority — penalize capturing prod=1 during opening/early
AOW_LOW_PROD_THRESHOLD = 2
AOW_LOW_PROD_PENALTY = 1.00           # was 0.84 — top1 captures prod<=2 at the same rate as prod>=3 ; the penalty was empirically wrong
# Rule 6: abandon peripheral defense when badly behind
AOW_RETREAT_PLANET_DEFICIT = 3        # enemy_planets - my_planets to trigger
AOW_RETREAT_MAX_PROD = 2              # only skip defense of prod ≤ this
# Rule 7: 4p single-enemy focus
AOW_4P_WEAKEST_FOCUS_BOOST = 2.10    # stronger focus on weakest enemy
AOW_4P_NON_WEAKEST_PENALTY = 0.62    # penalise attacking non-weakest in 4p
# Rule 8: decisive window t040-t080 when ahead on planets
AOW_DECISIVE_START = 40
AOW_DECISIVE_END = 80
AOW_DECISIVE_LEAD = 2                 # planet lead required
AOW_DECISIVE_NEUTRAL_BOOST = 1.30    # push expansion harder during this window


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

    n_my = len(world.my_planets)
    n_enemy = len(world.enemy_planets)

    # Rule 2: panic expansion if dangerously few planets in 2p mid-opening
    is_alarm = (
        not world.is_four_player
        and AOW_ALARM_START < world.step < AOW_ALARM_END
        and n_my < AOW_ALARM_PLANET_THRESHOLD
    )
    # Rule 3: push neutral expansion until conversion threshold is crossed
    is_conversion_push = (
        world.step < AOW_CONVERSION_TURN_LIMIT
        and n_my < AOW_CONVERSION_THRESHOLD
    )
    # Rule 6: abandon soft defense of low-prod planets when badly behind
    planet_deficit = n_enemy - n_my
    is_retreat_mode = (
        not world.is_four_player
        and planet_deficit >= AOW_RETREAT_PLANET_DEFICIT
        and world.step < 200
    )
    # Rule 8: decisive window — we're ahead, finish it
    is_decisive_window = (
        not world.is_four_player
        and AOW_DECISIVE_START < world.step < AOW_DECISIVE_END
        and (n_my - n_enemy) >= AOW_DECISIVE_LEAD
    )

    return {
        "domination": domination,
        "is_behind": is_behind,
        "is_ahead": is_ahead,
        "is_dominating": is_dominating,
        "is_finishing": is_finishing,
        "attack_margin_mult": attack_margin_mult,
        "is_alarm": is_alarm,
        "is_conversion_push": is_conversion_push,
        "is_retreat_mode": is_retreat_mode,
        "is_decisive_window": is_decisive_window,
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
    # Early flood: for first AOW_EARLY_FLOOD_TURNS, don't filter — capture everything reachable
    if world.step < AOW_EARLY_FLOOD_TURNS:
        return arrival_turns > ROTATING_OPENING_MAX_TURNS
    # Rule 1: skip prod=1 neutrals during opening in 2p
    if not world.is_four_player and target.production < AOW_OPENING_MIN_PROD:
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
        if world.is_early and not (
            not world.is_four_player and world.step < AOW_EARLY_FLOOD_TURNS
        ):
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

    # --- Art of War rules ---
    if target.owner == -1:
        # Rule 2: alarm — panic expansion, neutrals become urgent
        if modes.get("is_alarm"):
            value *= AOW_ALARM_NEUTRAL_BOOST
        # Rule 3: conversion push — stay below 15 planets = expand at all costs
        if modes.get("is_conversion_push"):
            value *= AOW_CONVERSION_NEUTRAL_BOOST
        # Rule 5: penalise very-low-prod neutrals during opening/early
        if world.is_early and target.production <= AOW_LOW_PROD_THRESHOLD:
            value *= AOW_LOW_PROD_PENALTY
        # Rule 8: decisive window — we're ahead, keep expanding to seal the game
        if modes.get("is_decisive_window"):
            value *= AOW_DECISIVE_NEUTRAL_BOOST
    # Rule 6: retreat mode — skip soft defense of low-prod own planets
    if (mission == "reinforce"
            and modes.get("is_retreat_mode")
            and target.owner == world.player
            and target.production <= AOW_RETREAT_MAX_PROD):
        return 0.0
    # Rule 7: 4p single-enemy focus — amplify weakest, penalise others harder
    if world.is_four_player and target.owner not in (-1, world.player):
        if target.owner == world.weakest_enemy_id:
            value *= AOW_4P_WEAKEST_FOCUS_BOOST
        elif not world.is_late:
            value *= AOW_4P_NON_WEAKEST_PENALTY

    if _SCORER is not None and value > 0:
        if _LOG_PLAYER < 0 or world.player == _LOG_PLAYER:
            feat = _build_mission_features(target, arrival_turns, mission, world, modes)
            noise = float(np.random.randn()) * _SCORER_NOISE_STD
            log_mult = float(_SCORER(feat)) + noise
            log_mult = max(-3.0, min(3.0, log_mult))
            _EPISODE_LOG.append((feat, noise))
            value *= math.exp(log_mult)

    return value


def _preferred_send(target, base_needed, arrival_turns, src_available, world, modes):
    if target.owner == -1 and not world.is_four_player and world.step < AOW_EARLY_FLOOD_TURNS:
        send = base_needed
    else:
        send = max(base_needed, int(math.ceil(base_needed * modes["attack_margin_mult"])))
    if target.owner == -1:
        # Early flood: minimal margin to enable parallel multi-planet capture
        if not world.is_four_player and world.step < AOW_EARLY_FLOOD_TURNS:
            margin = AOW_EARLY_FLOOD_MARGIN
            if _is_contested_neutral(target, world):
                margin = 1
            elif not _is_safe_neutral(target, world) and arrival_turns > max(6, ROTATING_OPENING_MAX_TURNS - 2):
                margin = 1
        elif modes.get("is_alarm"):
            margin_base = AOW_ALARM_MARGIN_BASE
            margin = min(NEUTRAL_MARGIN_CAP, margin_base + target.production * NEUTRAL_MARGIN_PROD_WEIGHT)
        elif not world.is_four_player:
            margin_base = TWO_PLAYER_NEUTRAL_MARGIN_BASE
            margin = min(
                TWO_PLAYER_NEUTRAL_MARGIN_CAP,
                margin_base + target.production * TWO_PLAYER_NEUTRAL_MARGIN_PROD_WEIGHT,
            )
            if _is_safe_neutral(target, world):
                margin = max(0, margin - 1)
            elif _is_contested_neutral(target, world):
                margin = min(TWO_PLAYER_NEUTRAL_MARGIN_CAP, margin + 1)
        else:
            margin_base = NEUTRAL_MARGIN_BASE
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

    # Rule 4: anti-hoarding — push ships from idle fat planets toward the front
    if (world.enemy_planets or world.neutral_planets) and len(world.my_planets) > 1 and not world.is_late:
        frontier_targets_ah = world.enemy_planets or world.static_neutral_planets or world.neutral_planets
        if frontier_targets_ah:
            front_ah = min(world.my_planets,
                           key=lambda p: min(_dist(p.x, p.y, t.x, t.y) for t in frontier_targets_ah))
            for planet in world.my_planets:
                if planet.id == front_ah.id or planet.id in world.doomed_candidates:
                    continue
                if planet.id in world.threatened_candidates:
                    continue
                hoard_limit = int(planet.production * AOW_HOARD_PROD_RATIO)
                avail = atk_left(planet.id)
                if avail < hoard_limit:
                    continue
                send = int(avail * AOW_HOARD_SEND_RATIO)
                if send < REAR_SEND_MIN_SHIPS:
                    continue
                aim = world.plan_shot(planet.id, front_ah.id, send)
                if aim is None:
                    continue
                if aim[1] > REAR_MAX_TRAVEL_TURNS:
                    continue
                append_move(planet.id, aim[0], send)

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
# REINFORCE injection (training only — no effect in competition)
# ---------------------------------------------------------------------------

MISSION_FEATURE_DIM = 15

_SCORER = None          # callable(np.ndarray) -> float, or None
_SCORER_NOISE_STD = 0.0
_EPISODE_LOG = []       # list of (features_array, noise_float)
_LOG_PLAYER = -1        # -1 = log all players, >=0 = only log that player


def set_scorer(scorer_fn, noise_std=0.0, log_player=-1):
    global _SCORER, _SCORER_NOISE_STD, _LOG_PLAYER
    _SCORER = scorer_fn
    _SCORER_NOISE_STD = float(noise_std)
    _LOG_PLAYER = int(log_player)


def reset_episode_log():
    global _EPISODE_LOG
    _EPISODE_LOG = []


def pop_episode_log():
    global _EPISODE_LOG
    log = _EPISODE_LOG[:]
    _EPISODE_LOG = []
    return log


# Tunable heuristic constants exposed for ES training.
# Reading globals() at function-call time (not at module-import) so writing to
# the module attribute *does* take effect on subsequent agent() calls.
_TUNABLE_KEYS = (
    "HOSTILE_MARGIN_BASE",
    "HOSTILE_MARGIN_CAP",
    "HOSTILE_MARGIN_PROD_WEIGHT",
    "NEUTRAL_MARGIN_BASE",
    "NEUTRAL_MARGIN_CAP",
    "NEUTRAL_MARGIN_PROD_WEIGHT",
    "TWO_PLAYER_HOSTILE_AGGRESSION_BOOST",
    "TWO_PLAYER_NEUTRAL_MARGIN_BASE",
    "ATTACK_COST_TURN_WEIGHT",
    "SNIPE_COST_TURN_WEIGHT",
    "STATIC_TARGET_MARGIN",
    "CONTESTED_TARGET_MARGIN",
    "SAFE_NEUTRAL_MARGIN",
    "CONTESTED_NEUTRAL_MARGIN",
)
_DEFAULT_PARAMS = {k: globals()[k] for k in _TUNABLE_KEYS if k in globals()}
_BASE_HEURISTIC_DEFAULTS = dict(_DEFAULT_PARAMS)
_DEFAULT_TRAINING_CHECKPOINT = Path(__file__).resolve().parent / "evaluations" / "scorer_v7_kaggle.npz"
_DEFAULT_CHECKPOINT_HEURISTIC_SPECS = (
    ("HOSTILE_MARGIN_BASE", 3.0, 1.5),
    ("HOSTILE_MARGIN_CAP", 12.0, 4.0),
    ("HOSTILE_MARGIN_PROD_WEIGHT", 2.0, 1.0),
    ("NEUTRAL_MARGIN_BASE", 2.0, 1.0),
    ("TWO_PLAYER_HOSTILE_AGGRESSION_BOOST", 1.35, 0.30),
    ("ATTACK_COST_TURN_WEIGHT", 0.55, 0.20),
    ("STATIC_TARGET_MARGIN", 4.0, 1.5),
)
_DEFAULT_SCORER_WEIGHTS = None


def set_heuristic_params(params):
    """Override tunable heuristic constants (used for ES training)."""
    g = globals()
    for k, v in params.items():
        if k in _DEFAULT_PARAMS:
            g[k] = v


def reset_heuristic_params():
    """Restore heuristic constants to their original module-load values."""
    g = globals()
    for k, v in _BASE_HEURISTIC_DEFAULTS.items():
        g[k] = v


def _decode_training_checkpoint_params(params):
    params = np.asarray(params, dtype=np.float32).ravel()
    if params.size < MISSION_FEATURE_DIM + len(_DEFAULT_CHECKPOINT_HEURISTIC_SPECS):
        return None
    scorer_w = params[:MISSION_FEATURE_DIM].astype(np.float64) * 0.5
    heur = {}
    for i, (name, base, scale) in enumerate(_DEFAULT_CHECKPOINT_HEURISTIC_SPECS):
        p = float(params[MISSION_FEATURE_DIM + i])
        p = max(-2.0, min(2.0, p))
        heur[name] = max(0.01, base + p * scale)
    return scorer_w, heur


def _load_default_training_checkpoint():
    global _DEFAULT_SCORER_WEIGHTS
    if not _DEFAULT_TRAINING_CHECKPOINT.exists():
        return False
    try:
        ckpt = np.load(_DEFAULT_TRAINING_CHECKPOINT, allow_pickle=False)
        decoded = _decode_training_checkpoint_params(ckpt["params"])
        if decoded is None:
            return False
        scorer_w, heur = decoded
        generation = int(ckpt["generation"]) if "generation" in ckpt else -1
        wr = float(ckpt["wr"]) if "wr" in ckpt else float("nan")
        reset_heuristic_params()
        set_heuristic_params(heur)
        _DEFAULT_SCORER_WEIGHTS = scorer_w
        set_scorer(lambda feat, w=scorer_w: float(w @ feat.astype(np.float64)),
                   noise_std=0.0, log_player=-1)
        print(
            f"[bot_v7] loaded checkpoint 53% from { _DEFAULT_TRAINING_CHECKPOINT.name } "
            f"(gen={generation}, wr={wr:.3f})"
        )
        return True
    except Exception:
        return False


_load_default_training_checkpoint()


def _build_mission_features(target, arrival_turns, mission, world, modes):
    total_prod = world.my_prod + world.enemy_prod + 1e-6
    domination = (world.my_total - world.max_enemy_strength) / max(
        1, world.my_total + world.max_enemy_strength
    )
    indirect = world.indirect_wealth_map.get(target.id, 0)
    return np.array([
        min(target.production / 5.0, 2.0),
        min(arrival_turns / 100.0, 2.0),
        world.remaining_steps / 500.0,
        float(not world.is_four_player),
        max(-1.0, min(1.0, domination)),
        float(target.owner not in (-1, world.player)),
        float(world.is_static(target.id)),
        min(indirect / (100.0 * world.remaining_steps + 1.0), 2.0),
        world.my_prod / total_prod,
        float(world.is_early),
        float(world.is_late),
        float(mission == "capture"),
        float(mission == "snipe"),
        float(mission == "swarm"),
        float(mission == "reinforce"),
    ], dtype=np.float32)


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


# ---------------------------------------------------------------------------
# V8.5 standalone compatibility layer
# ---------------------------------------------------------------------------

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_V7_AGENT = agent

class _V7Fallback:
    pass

v7 = _V7Fallback()
v7.agent = _V7_AGENT

# ---------------------------------------------------------------------------
# V8.5 constants (Fix #0..#5, 4p plans)
# ---------------------------------------------------------------------------

# Fix #0 — already-in-flight thresholds
ENEMY_PARTIAL_COMMITMENT_RATIO = 0.5

# Fix #1 — garrison floor multipliers (ships per production unit)
GARRISON_FLOOR_NEAR_DIST = 20.0
GARRISON_FLOOR_MID_DIST = 35.0
GARRISON_FLOOR_NEAR_MULT = 8
GARRISON_FLOOR_MID_MULT = 5
GARRISON_FLOOR_FAR_MULT = 3
GARRISON_FLOOR_MIN_ABSOLUTE = 5
GARRISON_FLOOR_4P_EARLY_RATIO = 0.55  # halve floor before t60 in 4p so expansion isn't strangled

# Fix #2 — garrison ratio cap
GARRISON_FLOOR_RATIO = 0.32        # at least 32% of total ships on planets

# Fix #3 — threat model
THREAT_LATENT_HORIZON = 12         # turns ahead to estimate latent enemy strikes
THREAT_LATENT_DISCOUNT = 0.30      # weight applied to top enemy reachable
THREAT_LATENT_TOP_K = 1            # only the strongest reachable enemy contributes

# Fix #4 — 4p early expansion overrides
FOUR_P_EARLY_TURN_LIMIT = 60
FOUR_P_EARLY_NEUTRAL_MARGIN = 1
FOUR_P_REAR_STAGING_MIN_TURN = 60

# Fix #5 — 4p offense throttle
ACTIVE_FRONT_THRESHOLD = 2
THROTTLE_GARRISON_RATIO = 0.28
THROTTLE_TOP2_PRESSURE_RATIO = 0.75
THROTTLE_RELEASE_TURNS = 8         # consecutive low-pressure turns to exit throttle

# 4p plan triggers
LATE_BLITZ_TURN = 280
LATE_BLITZ_EARLY_TURN = 90       # expose the finisher inside 120-turn ES games
LATE_BLITZ_MIN_GARRISON_RATIO = 0.36
LATE_BLITZ_MAX_ACTIVE_FRONTS = 1
LATE_BLITZ_MIN_SHIP_LEAD = 1.05
LATE_BLITZ_MIN_PROD_LEAD = 1.05
ELIMINATE_WEAKEST_FRACTION = 0.40  # weakest enemy < this fraction of my ships
OPPORTUNISTIC_DEPLETED_PROD_MULT = 2.25  # target.ships < production * this

# Ranker dimensions
N_STATE_FEATURES = 24
N_CANDIDATE_FEATURES = 14
N_PLANS_MAX = 11                   # baseline + attack + expand + defense + reserve + transfer + 5 4p variants
RANKER_TEMPERATURE = 1.0
EARLY_4P_EXPAND_PLANET_TARGET = 6
EARLY_4P_EXPAND_OVERRIDE_TURN = 70
EARLY_4P_EXPAND_MIN_GARRISON_RATIO = 0.28
CONVERSION_PUSH_MIN_TURN = 60
CONVERSION_PUSH_MIN_GARRISON_RATIO = 0.30
CONVERSION_PUSH_MAX_ACTIVE_FRONTS = 2
CONVERSION_PUSH_MIN_SHIP_LEAD = 1.03
CONVERSION_PUSH_MIN_PROD_LEAD = 1.03

# Friendly staging transfers. Kovi-style strong bots use friendly moves heavily
# to consolidate rear production into a small number of launch platforms.
TRANSFER_SOURCE_MIN_SHIPS = 14
TRANSFER_DEST_MIN_PRODUCTION = 1
TRANSFER_MAX_TRAVEL_TURNS = 42
TRANSFER_FRONT_PROGRESS = 0.88
TRANSFER_REAR_DISTANCE_RATIO = 1.12
TRANSFER_BASE_SEND_RATIO = 0.55
TRANSFER_PUSH_SEND_RATIO = 0.78
TRANSFER_MIN_SEND_SHIPS = 8
TRANSFER_FRONT_STOCK_PROD_MULT = 15
TRANSFER_FRONT_STOCK_ABS = 35
TRANSFER_ACTIVE_FRONT_BONUS = 1.35
TRANSFER_PLAN_BONUS = 1.45
TRANSFER_OVERRIDE_MIN_STEP = 25
TRANSFER_OVERRIDE_MIN_FRAC = 0.08
TRANSFER_OVERRIDE_MAX_ACTIVE_FRONTS = 3


# ---------------------------------------------------------------------------
# Module-level mutable state for ranker
# ---------------------------------------------------------------------------

_STATE_W = np.zeros((N_PLANS_MAX, N_STATE_FEATURES), dtype=np.float32)
_CAND_W = np.zeros(N_CANDIDATE_FEATURES, dtype=np.float32)
_VALUE_W = np.zeros(N_STATE_FEATURES, dtype=np.float32)
_VALUE_B = np.float32(0.0)
_CANDIDATE_LOG_CB: Optional[Callable[[dict], None]] = None
_THROTTLE_RELEASE_COUNTER = 0


# Plan id -> human label (also drives the ranker rows)
PLAN_NAMES = (
    "v7_baseline",          # 0
    "expand_focus",         # 1
    "attack_focus",         # 2
    "defense_focus",        # 3
    "reserve_hold",         # 4
    "transfer_push",        # 5  (friendly staging emphasis)
    "4p_opportunistic",     # 6  (4p only)
    "4p_eliminate_weakest", # 7  (4p only)
    "4p_conservation",      # 8  (4p only)
    "4p_late_blitz",        # 9  (4p only, t > LATE_BLITZ_TURN)
    "4p_conversion_push",   # 10 (4p only, favorable midgame conversion)
)


# ---------------------------------------------------------------------------
# Public training hooks
# ---------------------------------------------------------------------------

def set_ranker_weights(state_w=None, candidate_w=None, value_w=None, value_b=None):
    """Override ranker weights. Pass None to leave unchanged."""
    global _STATE_W, _CAND_W, _VALUE_W, _VALUE_B
    if state_w is not None:
        arr = np.asarray(state_w, dtype=np.float32)
        if arr.shape != (N_PLANS_MAX, N_STATE_FEATURES):
            raise ValueError(f"state_w shape {arr.shape} != {(N_PLANS_MAX, N_STATE_FEATURES)}")
        _STATE_W = arr.copy()
    if candidate_w is not None:
        arr = np.asarray(candidate_w, dtype=np.float32).ravel()
        if arr.size != N_CANDIDATE_FEATURES:
            raise ValueError(f"candidate_w size {arr.size} != {N_CANDIDATE_FEATURES}")
        _CAND_W = arr.copy()
    if value_w is not None:
        arr = np.asarray(value_w, dtype=np.float32).ravel()
        if arr.size != N_STATE_FEATURES:
            raise ValueError(f"value_w size {arr.size} != {N_STATE_FEATURES}")
        _VALUE_W = arr.copy()
    if value_b is not None:
        _VALUE_B = np.float32(value_b)


def get_ranker_weights():
    return (_STATE_W.copy(), _CAND_W.copy(), _VALUE_W.copy(), float(_VALUE_B))


def set_candidate_log_callback(cb):
    global _CANDIDATE_LOG_CB
    _CANDIDATE_LOG_CB = cb


def reset_throttle_state():
    global _THROTTLE_RELEASE_COUNTER
    _THROTTLE_RELEASE_COUNTER = 0


# ---------------------------------------------------------------------------
# Fix #0 — already-in-flight check
# ---------------------------------------------------------------------------

def my_ships_en_route_to(target_id: int, world: WorldModel) -> int:
    """Sum of own ships already in flight toward this target."""
    total = 0
    for entry in world.arrivals_by_planet.get(target_id, []):
        eta, owner, ships = entry
        if owner == world.player and ships > 0:
            total += int(ships)
    return total


def already_committed_enough(target_id: int, eta: int, world: WorldModel,
                              planned_commitments) -> bool:
    """True if current in-flight (+ planned) is enough — caller should skip new commit."""
    target = world.planet_by_id[target_id]
    in_flight = my_ships_en_route_to(target_id, world)
    if in_flight <= 0:
        return False
    needed = world.ships_needed_to_capture(target_id, eta, planned_commitments)
    if needed <= 0:
        return True
    if in_flight >= needed:
        return True
    if target.owner not in (-1, world.player) and in_flight > needed * ENEMY_PARTIAL_COMMITMENT_RATIO:
        # Enemy target with substantial commitment already — wait for resolution
        return True
    return False


# ---------------------------------------------------------------------------
# Fix #1 — garrison floor
# ---------------------------------------------------------------------------

def nearest_enemy_dist(world: WorldModel, planet) -> float:
    if not world.enemy_planets:
        return 1e6
    return min(_dist(planet.x, planet.y, e.x, e.y) for e in world.enemy_planets)


def garrison_floor(planet, world: WorldModel) -> int:
    prod = max(1, int(planet.production))
    d = nearest_enemy_dist(world, planet)
    if d < GARRISON_FLOOR_NEAR_DIST:
        base = prod * GARRISON_FLOOR_NEAR_MULT
    elif d < GARRISON_FLOOR_MID_DIST:
        base = prod * GARRISON_FLOOR_MID_MULT
    else:
        base = prod * GARRISON_FLOOR_FAR_MULT
    if world.is_four_player and world.step < FOUR_P_EARLY_TURN_LIMIT:
        base = int(base * GARRISON_FLOOR_4P_EARLY_RATIO)
    return max(GARRISON_FLOOR_MIN_ABSOLUTE, base)


# ---------------------------------------------------------------------------
# Fix #3 — latent threat per planet
# ---------------------------------------------------------------------------

def latent_threat(planet, world: WorldModel) -> int:
    """Estimated extra enemy ships that could reach this planet within horizon.

    Uses only the top-K reachable enemy planet(s) instead of summing the whole
    enemy roster — otherwise in 4p we add three rosters and lock garrisons so
    high that expansion stalls.
    """
    if not world.enemy_planets:
        return 0
    candidates = []
    for e in world.enemy_planets:
        if e.ships <= 0:
            continue
        eta = _travel_time(e.x, e.y, e.radius, planet.x, planet.y, planet.radius, max(1, int(e.ships)))
        if eta > THREAT_LATENT_HORIZON:
            continue
        candidates.append(int(e.ships))
    if not candidates:
        return 0
    candidates.sort(reverse=True)
    top = candidates[:max(1, THREAT_LATENT_TOP_K)]
    return int(sum(top) * THREAT_LATENT_DISCOUNT)


# ---------------------------------------------------------------------------
# Active fronts & garrison ratio (state-level signals)
# ---------------------------------------------------------------------------

def compute_active_fronts(world: WorldModel) -> Dict[int, int]:
    """{enemy_id: count_of_my_planets_targeted} for enemies currently inbound."""
    fronts: Dict[int, set] = defaultdict(set)
    for planet in world.my_planets:
        for entry in world.arrivals_by_planet.get(planet.id, []):
            eta, owner, ships = entry
            if owner not in (-1, world.player) and ships > 0:
                fronts[owner].add(planet.id)
    return {oid: len(s) for oid, s in fronts.items()}


def garrison_ratio_now(world: WorldModel) -> float:
    if world.my_total <= 0:
        return 1.0
    on_planets = sum(int(p.ships) for p in world.my_planets)
    return on_planets / max(1, world.my_total)


# ---------------------------------------------------------------------------
# State features (24-D) — used by ranker + value head
# ---------------------------------------------------------------------------

def build_state_features(world: WorldModel) -> np.ndarray:
    n_my = max(1, len(world.my_planets))
    n_enemy = max(1, len(world.enemy_planets))
    total_planets = max(1, n_my + n_enemy + len(world.neutral_planets))
    total_ships = max(1, world.my_total + world.enemy_total)
    total_prod = max(1, world.my_prod + world.enemy_prod)

    fronts = compute_active_fronts(world)
    n_active_fronts = len(fronts)
    n_contested = sum(1 for s in fronts.values() if s > 0)

    weakest_strength = 0
    strongest_strength = 0
    if world.enemy_planets:
        strengths = [s for o, s in world.owner_strength.items() if o not in (-1, world.player)]
        if strengths:
            weakest_strength = min(strengths)
            strongest_strength = max(strengths)

    weakest_frac = weakest_strength / max(1, world.my_total)
    strongest_frac = strongest_strength / max(1, world.my_total)

    g_ratio = garrison_ratio_now(world)
    avg_prod = world.my_prod / n_my
    if world.my_planets:
        min_garr = min(int(p.ships) for p in world.my_planets)
        min_garr_norm = min_garr / max(1, avg_prod)
    else:
        min_garr_norm = 0.0

    inter_enemy = 0
    for f in world.fleets:
        if f.owner in (-1, world.player):
            continue
        # Targeting any non-player non-neutral planet?
        for p in world.planets:
            if p.owner not in (-1, world.player, f.owner):
                inter_enemy += int(f.ships)
                break

    feat = np.array([
        world.my_total / total_ships,                              # 0
        world.enemy_total / total_ships,                           # 1
        world.my_prod / total_prod,                                 # 2
        world.enemy_prod / total_prod,                              # 3
        n_my / total_planets,                                       # 4
        n_enemy / total_planets,                                    # 5
        len(world.neutral_planets) / total_planets,                 # 6
        world.step / 500.0,                                         # 7
        world.remaining_steps / 500.0,                              # 8
        1.0 if world.is_four_player else 0.0,                       # 9
        1.0 if world.is_opening else 0.0,                           # 10
        1.0 if world.is_late else 0.0,                              # 11
        n_active_fronts / 3.0,                                      # 12
        weakest_frac,                                               # 13
        strongest_frac,                                             # 14
        n_contested / max(1, n_my),                                 # 15
        inter_enemy / max(1, total_ships),                          # 16
        g_ratio,                                                    # 17
        min(min_garr_norm / 10.0, 2.0),                             # 18
        min(world.max_enemy_strength / max(1, world.my_total), 3.0),# 19
        len(world.doomed_candidates) / max(1, n_my),                # 20
        len(world.threatened_candidates) / max(1, n_my),            # 21
        min(len(world.fleets) / 30.0, 2.0),                         # 22
        1.0,                                                        # 23 bias
    ], dtype=np.float32)
    return feat


# ---------------------------------------------------------------------------
# V7-equivalent plan generator (used as candidate 0 + as workhorse for variants)
# ---------------------------------------------------------------------------

# We re-implement plan_moves locally so we can inject:
#   - Fix #0: skip targets already in-flight
#   - Fix #1+#3: keep_needed = max(timeline, garrison_floor, latent_threat)
#   - Fix #2: garrison_ratio cap before final emit
#   - Fix #4: 4p early expansion overrides
#   - Fix #5: 4p offense throttle
#   - Variant biases (attack/expand/defense/reserve, 4p_*)
# Code mirrors v7.plan_moves but with hooks. Kept compact.

@dataclass
class PlanContext:
    variant: str = "v7_baseline"
    attack_boost: float = 1.0
    expand_boost: float = 1.0
    defense_boost: float = 1.0
    only_targets_owner: Optional[int] = None      # restrict offensive targets to this owner
    only_depleted_targets: bool = False           # opportunistic
    block_offense: bool = False                   # conservation
    suppress_garrison_floor: bool = False         # late_blitz
    suppress_garrison_ratio_cap: bool = False
    suppress_rear_staging: bool = False
    transfer_boost: float = 1.0
    force_transfer_staging: bool = False


def _adjust_keep_needed(world: WorldModel, planet, ctx: PlanContext, base_keep: int) -> int:
    if ctx.suppress_garrison_floor:
        return base_keep
    floor = garrison_floor(planet, world)
    latent = latent_threat(planet, world)
    return max(base_keep, floor, latent)


def _allow_4p_late_blitz(world: WorldModel, n_fronts: int, g_ratio: float) -> bool:
    """Late blitz stays late by default, but can activate earlier in a clearly favorable 4p endgame."""
    if world.step > LATE_BLITZ_TURN:
        return True
    if world.step < LATE_BLITZ_EARLY_TURN:
        return False
    if n_fronts > LATE_BLITZ_MAX_ACTIVE_FRONTS:
        return False
    if g_ratio < LATE_BLITZ_MIN_GARRISON_RATIO:
        return False
    strongest_enemy = max(
        (s for owner, s in world.owner_strength.items() if owner not in (-1, world.player)),
        default=0,
    )
    strongest_enemy_prod = max(
        (p for owner, p in world.owner_production.items() if owner not in (-1, world.player)),
        default=0,
    )
    ship_lead = world.my_total / max(1, strongest_enemy)
    prod_lead = world.my_prod / max(1, strongest_enemy_prod)
    return ship_lead >= LATE_BLITZ_MIN_SHIP_LEAD or prod_lead >= LATE_BLITZ_MIN_PROD_LEAD


def _conversion_push_owner(world: WorldModel, n_fronts: Optional[int] = None,
                           g_ratio: Optional[float] = None) -> Optional[int]:
    """Return the enemy owner to pressure when a stable 4p lead should convert."""
    if not world.is_four_player or world.step < CONVERSION_PUSH_MIN_TURN:
        return None
    if n_fronts is None:
        n_fronts = len(compute_active_fronts(world))
    if g_ratio is None:
        g_ratio = garrison_ratio_now(world)
    if n_fronts > CONVERSION_PUSH_MAX_ACTIVE_FRONTS or g_ratio < CONVERSION_PUSH_MIN_GARRISON_RATIO:
        return None

    enemy_strengths = [
        (owner, strength)
        for owner, strength in world.owner_strength.items()
        if owner not in (-1, world.player)
    ]
    if not enemy_strengths:
        return None
    strongest_owner, strongest_strength = max(enemy_strengths, key=lambda item: item[1])
    strongest_prod = max(
        (prod for owner, prod in world.owner_production.items() if owner not in (-1, world.player)),
        default=0,
    )
    ship_lead = world.my_total / max(1, strongest_strength)
    prod_lead = world.my_prod / max(1, strongest_prod)
    max_enemy_planets = max(
        (sum(1 for p in world.planets if p.owner == owner) for owner, _ in enemy_strengths),
        default=0,
    )
    planet_lead = len(world.my_planets) - max_enemy_planets
    if (ship_lead >= CONVERSION_PUSH_MIN_SHIP_LEAD
            or prod_lead >= CONVERSION_PUSH_MIN_PROD_LEAD
            or planet_lead >= 2):
        return int(strongest_owner)
    return None


def _has_4p_opportunistic_target(world: WorldModel) -> bool:
    """Only expose opportunistic mode when there is an actual depleted contested target."""
    for target in world.planets:
        if target.owner == world.player:
            continue
        is_depleted = target.ships < target.production * OPPORTUNISTIC_DEPLETED_PROD_MULT
        if not is_depleted:
            continue
        has_enemy_inbound = any(
            owner not in (-1, world.player)
            for _, owner, ships in world.arrivals_by_planet.get(target.id, [])
            if ships > 0
        )
        if target.owner not in (-1, world.player) or has_enemy_inbound:
            return True
    return False


def _transfer_frontier_targets(world: WorldModel) -> List[Planet]:
    return world.enemy_planets or world.static_neutral_planets or world.neutral_planets


def _frontier_distance_map(world: WorldModel, targets: Sequence[Planet]) -> Dict[int, float]:
    if not targets:
        return {p.id: 1e9 for p in world.my_planets}
    return {
        p.id: min(_dist(p.x, p.y, t.x, t.y) for t in targets)
        for p in world.my_planets
    }


def _emit_transfer_staging(world: WorldModel, ctx: PlanContext, atk_left, append_move) -> None:
    """Move safe rear ships to friendly launch platforms near the action."""
    if ctx.suppress_rear_staging or world.is_late or len(world.my_planets) < 2:
        return
    if world.is_four_player and world.step < FOUR_P_REAR_STAGING_MIN_TURN and not ctx.force_transfer_staging:
        return

    frontier_targets = _transfer_frontier_targets(world)
    if not frontier_targets:
        return

    frontier_distance = _frontier_distance_map(world, frontier_targets)
    safe_planets = [p for p in world.my_planets if p.id not in world.doomed_candidates]
    if len(safe_planets) < 2:
        return

    active_front_ids = {p.id for p in compute_active_fronts(world)}
    destinations = [
        p for p in safe_planets
        if p.production >= TRANSFER_DEST_MIN_PRODUCTION
    ]
    if not destinations:
        return

    send_ratio = TRANSFER_PUSH_SEND_RATIO if ctx.force_transfer_staging else TRANSFER_BASE_SEND_RATIO
    send_ratio = min(0.92, send_ratio * max(0.2, ctx.transfer_boost))

    for src in sorted(safe_planets, key=lambda p: -frontier_distance[p.id]):
        available = atk_left(src.id)
        if available < TRANSFER_SOURCE_MIN_SHIPS:
            continue

        src_frontier_dist = frontier_distance[src.id]
        best = None
        for dst in destinations:
            if dst.id == src.id:
                continue
            dst_frontier_dist = frontier_distance[dst.id]
            if dst_frontier_dist >= src_frontier_dist * TRANSFER_FRONT_PROGRESS:
                continue
            if src_frontier_dist < dst_frontier_dist * TRANSFER_REAR_DISTANCE_RATIO:
                continue

            stock_limit = max(TRANSFER_FRONT_STOCK_ABS, int(dst.production * TRANSFER_FRONT_STOCK_PROD_MULT))
            if int(dst.ships) >= stock_limit and dst.id not in active_front_ids and not ctx.force_transfer_staging:
                continue

            probe_send = max(TRANSFER_MIN_SEND_SHIPS, int(available * send_ratio))
            aim = world.plan_shot(src.id, dst.id, probe_send)
            if aim is None:
                continue
            angle, turns, _, _ = aim
            if turns > TRANSFER_MAX_TRAVEL_TURNS:
                continue

            progress = max(0.0, src_frontier_dist - dst_frontier_dist)
            stock_need = max(0, stock_limit - int(dst.ships))
            active_bonus = TRANSFER_ACTIVE_FRONT_BONUS if dst.id in active_front_ids else 1.0
            score = (progress / max(1.0, turns)) * active_bonus
            score += min(1.5, stock_need / max(1.0, stock_limit))
            if ctx.force_transfer_staging:
                score *= TRANSFER_PLAN_BONUS
            if best is None or score > best[0]:
                best = (score, dst, angle, turns)

        if best is None:
            continue
        _, dst, angle, _turns = best
        stock_limit = max(TRANSFER_FRONT_STOCK_ABS, int(dst.production * TRANSFER_FRONT_STOCK_PROD_MULT))
        stock_need = max(TRANSFER_MIN_SEND_SHIPS, stock_limit - int(dst.ships))
        stock_send = min(stock_need, int(available * 0.85))
        send = min(available, max(TRANSFER_MIN_SEND_SHIPS, int(available * send_ratio), stock_send))
        if send < TRANSFER_MIN_SEND_SHIPS:
            continue
        append_move(src.id, angle, send)


def _v8_5_plan(world: WorldModel, ctx: PlanContext) -> Tuple[List[List], List[int]]:
    """Returns (moves, sources_ships_sent). Pure heuristic plan with V8.5 fixes + variant bias."""
    modes = _build_modes(world)

    # Fix #4: 4p early expansion override (mutates modes via attack margin trick + adjusted preferred_send)
    is_4p_early = world.is_four_player and world.step < FOUR_P_EARLY_TURN_LIMIT

    # Fix #5: throttle decision
    fronts = compute_active_fronts(world)
    n_fronts = len(fronts)
    enemy_strengths = sorted((s for o, s in world.owner_strength.items() if o not in (-1, world.player)), reverse=True)
    top2_pressure = sum(enemy_strengths[:2])
    g_ratio = garrison_ratio_now(world)
    throttle = (
        world.is_four_player
        and (
            (n_fronts >= ACTIVE_FRONT_THRESHOLD and g_ratio < THROTTLE_GARRISON_RATIO)
            or (n_fronts >= ACTIVE_FRONT_THRESHOLD and top2_pressure > world.my_total * THROTTLE_TOP2_PRESSURE_RATIO)
        )
    )
    if ctx.block_offense:
        throttle = True

    planned_commitments = defaultdict(list)
    source_options_by_target = defaultdict(list)
    missions: List[Mission] = []
    moves: List[List] = []
    spent_total: Dict[int, int] = defaultdict(int)

    # V8.5 reserves: rebuild with adjusted keep_needed (Fix #1+#3).
    adjusted_reserve = {}
    for p in world.my_planets:
        base_keep = world.reserve.get(p.id, 0)
        adjusted_reserve[p.id] = min(int(p.ships), _adjust_keep_needed(world, p, ctx, base_keep))

    def inv_left(sid: int) -> int:
        src = world.planet_by_id[sid]
        return max(0, int(src.ships) - spent_total[sid])

    def atk_left(sid: int) -> int:
        return max(0, inv_left(sid) - adjusted_reserve.get(sid, 0))

    def append_move(sid: int, angle: float, ships: int) -> int:
        send = min(int(ships), inv_left(sid))
        if send < 1:
            return 0
        moves.append([sid, float(angle), int(send)])
        spent_total[sid] += send
        return send

    # Reinforcements (always run — defense baseline)
    missions.extend(_build_reinforcement_missions(world, planned_commitments, modes, inv_left))

    # Offensive candidates (skip if throttled)
    if not throttle:
        for src in world.my_planets:
            src_available = atk_left(src.id)
            if src_available <= 0:
                continue
            for target in world.planets:
                if target.id == src.id or target.owner == world.player:
                    continue

                # Variant filters
                if ctx.only_targets_owner is not None:
                    # eliminate_weakest mode: only attack the weakest enemy (and neutrals when nothing close)
                    if target.owner not in (-1, ctx.only_targets_owner):
                        continue
                if ctx.only_depleted_targets:
                    # opportunistic: only truly depleted planets that are contested
                    # or already changed hands. This avoids chasing soft neutrals.
                    is_depleted = target.ships < target.production * OPPORTUNISTIC_DEPLETED_PROD_MULT
                    has_enemy_inbound = any(o not in (-1, world.player)
                                            for _, o, _ in world.arrivals_by_planet.get(target.id, []))
                    recently_fought = target.owner not in (-1, world.player) or has_enemy_inbound
                    if not (is_depleted and recently_fought):
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

                # Fix #0 — skip if already enough in flight from previous turns
                if already_committed_enough(target.id, rough_turns, world, planned_commitments):
                    continue

                rough_needed = world.ships_needed_to_capture(target.id, rough_turns, planned_commitments)
                if rough_needed <= 0:
                    continue
                if _opening_filter(target, rough_turns, rough_needed, src_available, world):
                    continue

                # Fix #4 — tighter neutral margin in 4p early game
                send_guess = _preferred_send(target, rough_needed, rough_turns, src_available, world, modes)
                if is_4p_early and target.owner == -1:
                    send_guess = max(rough_needed + FOUR_P_EARLY_NEUTRAL_MARGIN, rough_needed)
                    send_guess = min(send_guess, src_available)

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

                if already_committed_enough(target.id, turns, world, planned_commitments):
                    continue

                needed = world.ships_needed_to_capture(target.id, turns, planned_commitments)
                if needed <= 0:
                    continue
                if _opening_filter(target, turns, needed, src_available, world):
                    continue

                send_cap = min(src_available, _preferred_send(target, needed, turns, src_available, world, modes))
                if is_4p_early and target.owner == -1:
                    send_cap = min(src_available, needed + FOUR_P_EARLY_NEUTRAL_MARGIN)
                if send_cap < 1:
                    continue
                if send_cap < needed and send_cap < PARTIAL_SOURCE_MIN_SHIPS:
                    continue
                value = _target_value(target, turns, "capture", world, modes)
                if value <= 0:
                    continue
                # Variant biases
                if target.owner not in (-1, world.player):
                    value *= ctx.attack_boost
                elif target.owner == -1:
                    value *= ctx.expand_boost
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

        # Multi-source swarms (re-uses V7 logic)
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
                    if hostile_target:
                        value *= ctx.attack_boost
                    elif target.owner == -1:
                        value *= ctx.expand_boost
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
                            if any(trio[a].send_cap + trio[b].send_cap >= need for a in range(3) for b in range(a + 1, 3)):
                                continue
                            value = _target_value(target, jt, "swarm", world, modes)
                            if value <= 0:
                                continue
                            value *= ctx.attack_boost
                            ts = _apply_score_mods(value / (need + jt * ATTACK_COST_TURN_WEIGHT + 1.0),
                                                   target, "swarm", world) * THREE_SOURCE_PLAN_PENALTY
                            missions.append(Mission(kind="swarm", score=ts, target_id=target_id, turns=jt, options=trio))

    missions.sort(key=lambda m: -m.score)

    # Apply boost to defense missions (reinforce kind)
    if ctx.defense_boost != 1.0:
        for m in missions:
            if m.kind == "reinforce":
                m.score *= ctx.defense_boost
        missions.sort(key=lambda m: -m.score)

    # Mission emission (mirrors V7)
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

        # Swarm
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
            remaining_other = sum(lim for _, lim in ordered[idx + 1:])
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

    # Follow-up pass
    if not throttle and not world.is_very_late:
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
                if ctx.only_targets_owner is not None and target.owner not in (-1, ctx.only_targets_owner):
                    continue
                rough_ships = max(1, min(src_left, max(PARTIAL_SOURCE_MIN_SHIPS, int(target.ships) + 1)))
                rough_aim = world.plan_shot(src.id, target.id, rough_ships)
                if rough_aim is None:
                    continue
                est_turns = rough_aim[1]
                if world.is_late and est_turns > world.remaining_steps - LATE_CAPTURE_BUFFER:
                    continue
                if already_committed_enough(target.id, est_turns, world, planned_commitments):
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
                if target.owner not in (-1, world.player):
                    value *= ctx.attack_boost
                else:
                    value *= ctx.expand_boost
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
            if already_committed_enough(target.id, turns, world, planned_commitments):
                continue
            src_left = atk_left(src.id)
            send = min(src_left, max(missing, _preferred_send(target, missing, turns, src_left, world, modes)))
            if send < missing:
                continue
            actual = append_move(src.id, angle, send)
            if actual >= missing:
                planned_commitments[target.id].append((turns, world.player, int(actual)))

    # Doomed evac (always — never throttle survival)
    if world.doomed_candidates:
        frontier_targets = world.enemy_planets or world.static_neutral_planets or world.neutral_planets
        if frontier_targets:
            frontier_distance = {p.id: min(_dist(p.x, p.y, t.x, t.y) for t in frontier_targets) for p in world.my_planets}
        else:
            frontier_distance = {p.id: 1e9 for p in world.my_planets}
        for planet in world.my_planets:
            if planet.id not in world.doomed_candidates:
                continue
            available_now = inv_left(planet.id)
            if available_now < adjusted_reserve.get(planet.id, 0):
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
            retreat = min(safe_allies, key=lambda a: (frontier_distance.get(a.id, 1e9), _dist(planet.x, planet.y, a.x, a.y)))
            aim = world.plan_shot(planet.id, retreat.id, available_now)
            if aim is None:
                continue
            append_move(planet.id, aim[0], available_now)

    _emit_transfer_staging(world, ctx, atk_left, append_move)

    # Rear-to-front staging — Fix #4 disables it during 4p early
    rear_staging_active = (
        not throttle
        and not ctx.suppress_rear_staging
        and (world.enemy_planets or world.neutral_planets)
        and len(world.my_planets) > 1
        and not world.is_late
        and not (world.is_four_player and world.step < FOUR_P_REAR_STAGING_MIN_TURN)
    )
    if rear_staging_active:
        frontier_targets = world.enemy_planets or world.static_neutral_planets or world.neutral_planets
        frontier_distance = {p.id: min(_dist(p.x, p.y, t.x, t.y) for t in frontier_targets) for p in world.my_planets}
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

    # Anti-hoarding (skip when throttled)
    if (not throttle and not ctx.suppress_rear_staging
            and (world.enemy_planets or world.neutral_planets)
            and len(world.my_planets) > 1 and not world.is_late):
        frontier_targets_ah = world.enemy_planets or world.static_neutral_planets or world.neutral_planets
        if frontier_targets_ah:
            front_ah = min(world.my_planets,
                           key=lambda p: min(_dist(p.x, p.y, t.x, t.y) for t in frontier_targets_ah))
            for planet in world.my_planets:
                if planet.id == front_ah.id or planet.id in world.doomed_candidates:
                    continue
                if planet.id in world.threatened_candidates:
                    continue
                hoard_limit = int(planet.production * AOW_HOARD_PROD_RATIO)
                avail = atk_left(planet.id)
                if avail < hoard_limit:
                    continue
                send = int(avail * AOW_HOARD_SEND_RATIO)
                if send < REAR_SEND_MIN_SHIPS:
                    continue
                aim = world.plan_shot(planet.id, front_ah.id, send)
                if aim is None:
                    continue
                if aim[1] > REAR_MAX_TRAVEL_TURNS:
                    continue
                append_move(planet.id, aim[0], send)

    # Fix #2 — garrison ratio cap. Trim moves that would push g_ratio too low.
    if not ctx.suppress_garrison_ratio_cap and moves and world.my_total > 0:
        total_send = sum(m[2] for m in moves)
        on_planets_after = sum(int(p.ships) for p in world.my_planets) - total_send
        ratio_after = on_planets_after / max(1, world.my_total)
        if ratio_after < GARRISON_FLOOR_RATIO:
            target_on_planets = world.my_total * GARRISON_FLOOR_RATIO
            current_on_planets = sum(int(p.ships) for p in world.my_planets)
            max_total_send = max(0, current_on_planets - int(target_on_planets))
            if total_send > max_total_send:
                # Trim least valuable moves (last-emitted = followup/staging) first
                trim = total_send - max_total_send
                trimmed = []
                for sid, ang, ships in reversed(moves):
                    if trim <= 0:
                        trimmed.append([sid, ang, ships])
                        continue
                    if ships <= trim:
                        trim -= ships
                        # drop entirely
                    else:
                        trimmed.append([sid, ang, ships - trim])
                        trim = 0
                trimmed.reverse()
                moves = trimmed

    # Final dedup
    final_moves = []
    used_final: Dict[int, int] = defaultdict(int)
    for sid, angle, ships in moves:
        source = world.planet_by_id[sid]
        max_allowed = int(source.ships) - used_final[sid]
        send = min(int(ships), max_allowed)
        if send >= 1:
            final_moves.append([sid, float(angle), int(send)])
            used_final[sid] += send

    return final_moves, [used_final[sid] for sid in used_final]


# ---------------------------------------------------------------------------
# Candidate plans — produce up to N_PLANS_MAX (some only in 4p)
# ---------------------------------------------------------------------------

def _generate_candidates(world: WorldModel) -> List[Tuple[str, List[List]]]:
    plans: List[Tuple[str, List[List]]] = []
    is_4p = world.is_four_player

    # 0. v7_baseline (V7 grammar + full V8.5 safety fixes)
    plans.append(("v7_baseline", _v8_5_plan(world, PlanContext("v7_baseline"))[0]))
    # 1. expand_focus
    plans.append(("expand_focus",
                  _v8_5_plan(world, PlanContext("expand_focus", expand_boost=1.45,
                                                attack_boost=0.85))[0]))
    # 2. attack_focus
    plans.append(("attack_focus",
                  _v8_5_plan(world, PlanContext("attack_focus", attack_boost=1.55,
                                                expand_boost=0.85))[0]))
    # 3. defense_focus
    plans.append(("defense_focus",
                  _v8_5_plan(world, PlanContext("defense_focus", defense_boost=1.50,
                                                attack_boost=0.85, expand_boost=0.85))[0]))
    # 4. reserve_hold — emit no offense, only reinforcement & evac
    plans.append(("reserve_hold",
                  _v8_5_plan(world, PlanContext("reserve_hold", block_offense=True,
                                                defense_boost=1.30,
                                                suppress_rear_staging=True))[0]))
    # 5. transfer_push — explicitly consolidates rear production into forward launch platforms.
    plans.append(("transfer_push",
                  _v8_5_plan(world, PlanContext("transfer_push",
                                                attack_boost=0.92,
                                                expand_boost=0.95,
                                                transfer_boost=1.30,
                                                force_transfer_staging=True))[0]))

    if is_4p:
        # 6. 4p_opportunistic. When no scavenging target exists, keep this as
        # a guarded tempo plan so old rankers do not collapse into passivity.
        if _has_4p_opportunistic_target(world):
            plans.append(("4p_opportunistic",
                          _v8_5_plan(world, PlanContext("4p_opportunistic",
                                                        only_depleted_targets=True,
                                                        attack_boost=1.10,
                                                        expand_boost=1.10))[0]))
        else:
            plans.append(("4p_opportunistic",
                          _v8_5_plan(world, PlanContext("4p_opportunistic",
                                                        attack_boost=0.95,
                                                        expand_boost=1.25,
                                                        defense_boost=1.05))[0]))
        # 7. 4p_eliminate_weakest — only if a player is weak enough to focus down.
        weakest = world.weakest_enemy_id
        weakest_strength = world.owner_strength.get(weakest, 0) if weakest is not None else 0
        if weakest is not None and weakest_strength <= world.my_total * ELIMINATE_WEAKEST_FRACTION:
            plans.append(("4p_eliminate_weakest",
                          _v8_5_plan(world, PlanContext("4p_eliminate_weakest",
                                                         only_targets_owner=weakest,
                                                         attack_boost=1.80,
                                                         expand_boost=0.70))[0]))
        # 8. 4p_conservation — explicit no-offense plan, but still allows rear-to-front redistribution
        conservation_boost = 1.85 if len(compute_active_fronts(world)) < 3 else 2.05
        plans.append(("4p_conservation",
                      _v8_5_plan(world, PlanContext("4p_conservation", block_offense=True,
                                                    defense_boost=conservation_boost,
                                                    suppress_rear_staging=False))[0]))
        # 9. 4p_late_blitz
        if _allow_4p_late_blitz(world, len(compute_active_fronts(world)), garrison_ratio_now(world)):
            plans.append(("4p_late_blitz",
                          _v8_5_plan(world, PlanContext("4p_late_blitz",
                                                        attack_boost=1.80,
                                                        suppress_garrison_floor=True,
                                                        suppress_garrison_ratio_cap=True))[0]))
        # 10. 4p_conversion_push — convert a stable lead before a weak enemy/blitz exists.
        conversion_owner = _conversion_push_owner(world, len(compute_active_fronts(world)), garrison_ratio_now(world))
        if conversion_owner is not None:
            plans.append(("4p_conversion_push",
                          _v8_5_plan(world, PlanContext("4p_conversion_push",
                                                        only_targets_owner=conversion_owner,
                                                        attack_boost=2.05,
                                                        expand_boost=1.05,
                                                        defense_boost=0.95))[0]))

    return plans


# ---------------------------------------------------------------------------
# Candidate features (14-D per plan)
# ---------------------------------------------------------------------------

def build_candidate_features(plan_name: str, moves: List[List], world: WorldModel) -> np.ndarray:
    n_moves = len(moves)
    total_send = sum(m[2] for m in moves) if moves else 0
    my_total = max(1, world.my_total)
    ship_frac = total_send / my_total

    # Targets summary — match each move to the closest planet along its launch
    # angle. The previous tolerance of 0.25 rad was too tight: orbit-prediction
    # angles often deviate by 0.4-0.6 rad from a straight-line shot, so most
    # moves were unmatched and the categorical features collapsed to zero.
    targets_attack = 0
    targets_expand = 0
    targets_defense = 0
    transfer_ships = 0
    avg_eta = 0.0
    n_eta = 0
    distinct_targets = set()
    for sid, angle, ships in moves:
        src = world.planet_by_id.get(sid)
        if src is None:
            continue
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        best = None
        best_score = 1e18
        for p in world.planets:
            if p.id == sid:
                continue
            dx = p.x - src.x
            dy = p.y - src.y
            d = math.hypot(dx, dy)
            if d <= 1e-6:
                continue
            # Project onto launch direction; require positive forward distance
            proj = dx * cos_a + dy * sin_a
            if proj <= 0:
                continue
            # Perpendicular gap shrinks as planets line up with the angle
            perp = abs(dx * sin_a - dy * cos_a)
            if perp > p.radius + 6.0:
                continue
            score = perp - 0.05 * proj  # prefer closer alignment, ties broken toward nearer
            if score < best_score:
                best = p
                best_score = score
        if best is None:
            continue
        distinct_targets.add(best.id)
        if best.owner == world.player:
            targets_defense += 1
            transfer_ships += int(ships)
        elif best.owner == -1:
            targets_expand += 1
        else:
            targets_attack += 1
        avg_eta += math.hypot(best.x - src.x, best.y - src.y)
        n_eta += 1
    avg_eta = (avg_eta / max(1, n_eta)) / 100.0

    transfer_frac = transfer_ships / my_total

    # Plan-id one-hot collapsed: keep plan_idx normalized + dedicated flags
    plan_idx = PLAN_NAMES.index(plan_name) if plan_name in PLAN_NAMES else 0

    feat = np.array([
        min(n_moves / 20.0, 2.0),                                    # 0
        min(ship_frac, 2.0),                                         # 1
        targets_attack / max(1, n_moves),                            # 2
        targets_expand / max(1, n_moves),                            # 3
        targets_defense / max(1, n_moves),                           # 4
        avg_eta,                                                     # 5
        plan_idx / float(N_PLANS_MAX),                               # 6
        1.0 if plan_name == "v7_baseline" else 0.0,                  # 7
        1.0 if plan_name.startswith("4p_") else 0.0,                 # 8
        1.0 if plan_name in ("defense_focus", "reserve_hold", "4p_conservation") else 0.0,  # 9
        1.0 if plan_name in ("attack_focus", "4p_eliminate_weakest", "4p_late_blitz", "4p_conversion_push") else 0.0,  # 10
        min(transfer_frac, 2.0),                                      # 11
        1.0 if plan_name == "transfer_push" else 0.0,                # 12
        1.0,                                                         # 13 bias
    ], dtype=np.float32)
    return feat


# ---------------------------------------------------------------------------
# Ranker scoring
# ---------------------------------------------------------------------------

def score_candidates(state_feat: np.ndarray, candidates: List[Tuple[str, List[List]]],
                     world: WorldModel) -> Tuple[List[float], List[np.ndarray]]:
    """Return (scores per candidate, candidate features per candidate)."""
    cand_feats: List[np.ndarray] = []
    scores: List[float] = []
    for plan_name, moves in candidates:
        cf = build_candidate_features(plan_name, moves, world)
        plan_idx = PLAN_NAMES.index(plan_name)
        # Score = state_w[plan_idx] · state_feat + cand_w · cand_feat
        s = float(_STATE_W[plan_idx] @ state_feat) + float(_CAND_W @ cf)
        scores.append(s)
        cand_feats.append(cf)
    return scores, cand_feats


def value_estimate(state_feat: np.ndarray) -> float:
    return float(_VALUE_W @ state_feat) + float(_VALUE_B)


def _select_candidate_index(candidates: List[Tuple[str, List[List]]], scores: List[float],
                            cand_feats: List[np.ndarray], world: WorldModel) -> int:
    if (world.is_four_player
            and world.step < EARLY_4P_EXPAND_OVERRIDE_TURN
            and len(world.my_planets) < EARLY_4P_EXPAND_PLANET_TARGET
            and not compute_active_fronts(world)
            and garrison_ratio_now(world) >= EARLY_4P_EXPAND_MIN_GARRISON_RATIO):
        for i, (plan_name, moves) in enumerate(candidates):
            if plan_name == "expand_focus" and moves:
                return i

    best_idx = 0
    best_score = scores[0]
    for i in range(1, len(scores)):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
    if (world.step >= TRANSFER_OVERRIDE_MIN_STEP
            and not world.is_late
            and len(compute_active_fronts(world)) <= TRANSFER_OVERRIDE_MAX_ACTIVE_FRONTS):
        chosen_transfer_frac = float(cand_feats[best_idx][11])
        for i, (plan_name, moves) in enumerate(candidates):
            if plan_name != "transfer_push" or not moves:
                continue
            transfer_frac = float(cand_feats[i][11])
            if transfer_frac >= TRANSFER_OVERRIDE_MIN_FRAC and transfer_frac > chosen_transfer_frac + 0.04:
                return i
    if _conversion_push_owner(world) is not None:
        chosen_name, chosen_moves = candidates[best_idx]
        chosen_attack_ratio = float(cand_feats[best_idx][2])
        chosen_is_generic = chosen_name in ("v7_baseline", "attack_focus", "4p_opportunistic")
        chosen_is_passive = chosen_name in ("defense_focus", "reserve_hold", "4p_conservation")
        chosen_lacks_attack = bool(chosen_moves) and chosen_attack_ratio <= 0.05
        for i, (plan_name, moves) in enumerate(candidates):
            if plan_name == "4p_conversion_push" and moves and float(cand_feats[i][2]) > 0.05:
                if chosen_is_generic or chosen_is_passive or chosen_lacks_attack or not chosen_moves:
                    return i
    return best_idx


# Public deterministic planner useful for tests and ablations. It returns the
# current ranker-selected V8.5 candidate plan for an already-built WorldModel.
def plan_moves_v8_5(world: WorldModel) -> List[List]:
    candidates = _generate_candidates(world)
    if not candidates:
        return []
    state_feat = build_state_features(world)
    scores, cand_feats = score_candidates(state_feat, candidates, world)
    best_idx = _select_candidate_index(candidates, scores, cand_feats, world)
    return candidates[best_idx][1]


# ---------------------------------------------------------------------------
# Agent entry
# ---------------------------------------------------------------------------

def agent(obs, config=None):
    try:
        world = _build_world(obs)
        if not world.my_planets:
            return []
        candidates = _generate_candidates(world)
        if not candidates:
            return []
        # Single candidate? Return immediately.
        if len(candidates) == 1:
            return candidates[0][1]
        state_feat = build_state_features(world)
        scores, cand_feats = score_candidates(state_feat, candidates, world)
        best_idx = _select_candidate_index(candidates, scores, cand_feats, world)
        if _CANDIDATE_LOG_CB is not None:
            try:
                _CANDIDATE_LOG_CB({
                    "step": world.step,
                    "player": world.player,
                    "state_feat": state_feat,
                    "candidate_names": [c[0] for c in candidates],
                    "candidate_feats": cand_feats,
                    "candidate_moves": [c[1] for c in candidates],
                    "scores": scores,
                    "chosen": best_idx,
                    "value": value_estimate(state_feat),
                })
            except Exception:
                pass
        return candidates[best_idx][1]
    except Exception:
        # Hard fallback to V7 grammar — never forfeit a turn silently.
        try:
            return v7.agent(obs, config)
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "evaluations" / "v8_5_ranker.npz"


def save_checkpoint(path: Optional[str] = None, meta: Optional[dict] = None) -> str:
    p = Path(path) if path else DEFAULT_CHECKPOINT
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_w": _STATE_W,
        "candidate_w": _CAND_W,
        "value_w": _VALUE_W,
        "value_b": np.array([_VALUE_B], dtype=np.float32),
    }
    if meta:
        for k, v in meta.items():
            payload[f"meta_{k}"] = np.asarray(v)
    np.savez_compressed(str(p), **payload)
    return str(p)


def load_checkpoint(path: Optional[str] = None) -> bool:
    p = Path(path) if path else DEFAULT_CHECKPOINT
    if not p.exists():
        return False
    try:
        data = np.load(str(p))
        state_w = np.asarray(data["state_w"], dtype=np.float32)
        if state_w.shape != (N_PLANS_MAX, N_STATE_FEATURES):
            migrated = np.zeros((N_PLANS_MAX, N_STATE_FEATURES), dtype=np.float32)
            rows = min(state_w.shape[0], N_PLANS_MAX)
            cols = min(state_w.shape[1], N_STATE_FEATURES)
            migrated[:rows, :cols] = state_w[:rows, :cols]
            state_w = migrated
        candidate_w = np.asarray(data["candidate_w"], dtype=np.float32).ravel()
        if candidate_w.size != N_CANDIDATE_FEATURES:
            migrated_candidate = np.zeros(N_CANDIDATE_FEATURES, dtype=np.float32)
            cols = min(candidate_w.size, N_CANDIDATE_FEATURES)
            migrated_candidate[:cols] = candidate_w[:cols]
            candidate_w = migrated_candidate
        set_ranker_weights(
            state_w=state_w,
            candidate_w=candidate_w,
            value_w=data["value_w"],
            value_b=float(data["value_b"][0]),
        )
        return True
    except Exception:
        return False


# Auto-load on import if checkpoint exists
if os.environ.get("BOT_V8_5_NO_AUTOLOAD") != "1":
    load_checkpoint()


__all__ = [
    "agent",
    "set_ranker_weights",
    "get_ranker_weights",
    "set_candidate_log_callback",
    "save_checkpoint",
    "load_checkpoint",
    "build_state_features",
    "build_candidate_features",
    "score_candidates",
    "value_estimate",
    "plan_moves_v8_5",
    "PLAN_NAMES",
    "N_STATE_FEATURES",
    "N_CANDIDATE_FEATURES",
    "N_PLANS_MAX",
    "DEFAULT_CHECKPOINT",
]
