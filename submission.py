"""Standalone Orbit Wars V9 submission.

This file is self-contained for Kaggle upload:
- no imports from the local repo
- V9 weights embedded from v9_guardian_8h_vps_policy.npz, exported from the promoted best checkpoint
- V9-style state / plan features and policy scoring
- compact candidate generation for multiple plan families
"""

from __future__ import annotations

import base64
import io
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


P_ID, P_OWNER, P_X, P_Y, P_R, P_SHIPS, P_PROD = range(7)
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = range(7)

TOTAL_STEPS = 500
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_R = 10.0
SUN_SHOT_MARGIN = 4.0
LAUNCH_CLEARANCE = 0.1
MAX_SPEED = 6.0
AIM_ITERATIONS = 10
INTERCEPT_TOLERANCE = 1
SHOT_HORIZON = 90
SUN_GUARD_RAY_DISTANCE = 150.0

STATE_FEATURE_NAMES = (
    "my_ship_share",
    "enemy_ship_share",
    "strongest_enemy_ship_share",
    "my_prod_share",
    "enemy_prod_share",
    "strongest_enemy_prod_share",
    "my_planet_share",
    "enemy_planet_share",
    "neutral_planet_share",
    "neutral_prod_share",
    "neutral_softness",
    "step_fraction",
    "remaining_fraction",
    "is_4p",
    "is_opening",
    "is_late",
    "is_very_late",
    "active_front_ratio",
    "threatened_ratio",
    "doomed_ratio",
    "garrison_ratio",
    "fleet_density",
    "weakest_enemy_fraction",
    "ship_lead_vs_strongest",
    "prod_lead_vs_strongest",
    "finish_pressure",
    "comeback_pressure",
    "owned_production_density",
    "enemy_production_density",
    "center_control",
    "frontier_spread",
    "bias",
)

PLAN_TYPES = (
    "balanced",
    "aggressive_expansion",
    "delayed_strike",
    "multi_step_trap",
    "resource_denial",
    "endgame_finisher",
    "defensive_consolidation",
    "staging_transfer",
    "opportunistic_snipe",
    "probe",
    "reserve_hold",
)

PLAN_FEATURE_NAMES = (
    "move_count",
    "ship_commitment",
    "attack_move_frac",
    "expand_move_frac",
    "defense_move_frac",
    "transfer_ship_frac",
    "avg_eta",
    "target_prod_gain",
    "target_ship_cost",
    "distinct_target_frac",
    "overcommit_risk",
    "undercommit_risk",
    "finisher_flag",
    "denial_flag",
    "trap_flag",
    "probe_flag",
    "opening_flag",
    "late_flag",
    "four_player_flag",
    "frontier_pressure",
    "garrison_after_commit",
    "weak_enemy_focus",
    "high_prod_focus",
    "neutral_focus",
    "enemy_focus",
    "plan_type_norm",
    "base_score",
    "bias",
)

STATE_FEATURE_INDEX = {name: i for i, name in enumerate(STATE_FEATURE_NAMES)}
PLAN_FEATURE_INDEX = {name: i for i, name in enumerate(PLAN_FEATURE_NAMES)}
PLAN_TYPE_TO_INDEX = {name: i for i, name in enumerate(PLAN_TYPES)}


@dataclass
class Planet:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: float
    production: float

    @property
    def r(self) -> float:
        return self.radius

    @property
    def prod(self) -> float:
        return self.production


@dataclass
class Fleet:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: float

    @property
    def from_id(self) -> int:
        return self.from_planet_id


@dataclass
class PlanCandidate:
    name: str
    moves: List[List]
    plan_type: str
    base_score: float = 0.0
    metadata: Dict[str, float] = field(default_factory=dict)

    def clipped(self, max_moves: int) -> "PlanCandidate":
        if len(self.moves) <= max_moves:
            return self
        clone = PlanCandidate(
            name=self.name,
            moves=self.moves[:max_moves],
            plan_type=self.plan_type,
            base_score=self.base_score,
            metadata=dict(self.metadata),
        )
        clone.metadata["clipped"] = 1.0
        return clone


@dataclass
class World:
    player: int
    step: int
    remaining_steps: int
    planets: List[Planet]
    fleets: List[Fleet]
    planet_by_id: Dict[int, Planet]
    my_planets: List[Planet]
    enemy_planets: List[Planet]
    neutral_planets: List[Planet]
    initial_by_id: Dict[int, Planet]
    angular_velocity: float
    comets: List[dict]
    comet_ids: set
    owner_strength: Dict[int, float]
    owner_production: Dict[int, float]
    my_total: float
    enemy_total: float
    my_prod: float
    enemy_prod: float
    is_four_player: bool
    is_opening: bool
    is_late: bool
    is_very_late: bool
    threatened_candidates: set
    doomed_candidates: set
    incoming_enemy: Dict[int, float]
    incoming_friendly: Dict[int, float]
    incoming_enemy_eta: Dict[int, int]
    weakest_enemy_id: Optional[int]
    strongest_enemy_id: Optional[int]


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _dist(a, b) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def _angle(a, b) -> float:
    return math.atan2(float(b.y) - float(a.y), float(b.x) - float(a.x))


def _angle_delta(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(d)


def _fleet_speed(ships: float) -> float:
    if ships <= 1:
        return 1.0
    ratio = math.log(max(float(ships), 1.0)) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


def _point_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    lsq = dx * dx + dy * dy
    if lsq <= 1e-9:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / lsq))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _path_sun_clearance(src: Planet, angle: float, distance: float) -> float:
    ca = math.cos(angle)
    sa = math.sin(angle)
    clearance = float(src.radius) + LAUNCH_CLEARANCE
    x1 = float(src.x) + ca * clearance
    y1 = float(src.y) + sa * clearance
    x2 = x1 + ca * max(0.0, float(distance))
    y2 = y1 + sa * max(0.0, float(distance))
    return _point_segment_distance(CENTER_X, CENTER_Y, x1, y1, x2, y2)


def _path_too_close_to_sun(src: Planet, angle: float, distance: float) -> bool:
    guard_distance = max(float(distance), SUN_GUARD_RAY_DISTANCE)
    return _path_sun_clearance(src, angle, guard_distance) < SUN_R + SUN_SHOT_MARGIN


def _parse_planet(raw) -> Planet:
    return Planet(
        int(raw[P_ID]),
        int(raw[P_OWNER]),
        float(raw[P_X]),
        float(raw[P_Y]),
        float(raw[P_R]),
        float(raw[P_SHIPS]),
        float(raw[P_PROD]),
    )


def _parse_fleet(raw) -> Fleet:
    return Fleet(
        int(raw[F_ID]),
        int(raw[F_OWNER]),
        float(raw[F_X]),
        float(raw[F_Y]),
        float(raw[F_ANGLE]),
        int(raw[F_FROM]),
        float(raw[F_SHIPS]),
    )


def _predict_planet_pos(planet: Planet, world: World, turns: int) -> Tuple[float, float]:
    init = world.initial_by_id.get(int(planet.id))
    if init is None:
        return float(planet.x), float(planet.y)
    radius_from_center = math.hypot(float(init.x) - CENTER_X, float(init.y) - CENTER_Y)
    if radius_from_center + float(init.radius) >= 50.0:
        return float(planet.x), float(planet.y)
    cur_ang = math.atan2(float(planet.y) - CENTER_Y, float(planet.x) - CENTER_X)
    new_ang = cur_ang + float(world.angular_velocity) * int(turns)
    return CENTER_X + radius_from_center * math.cos(new_ang), CENTER_Y + radius_from_center * math.sin(new_ang)


def _predict_comet_pos(planet_id: int, world: World, turns: int) -> Optional[Tuple[float, float]]:
    for group in world.comets:
        pids = group.get("planet_ids", []) or []
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", []) or []
        path_index = int(group.get("path_index", 0) or 0)
        if idx >= len(paths):
            return None
        path = paths[idx]
        fi = path_index + int(turns)
        if 0 <= fi < len(path):
            return float(path[fi][0]), float(path[fi][1])
        return None
    return None


def _comet_remaining_life(planet_id: int, world: World) -> int:
    for group in world.comets:
        pids = group.get("planet_ids", []) or []
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", []) or []
        path_index = int(group.get("path_index", 0) or 0)
        if idx < len(paths):
            return max(0, len(paths[idx]) - path_index)
    return 0


def _predict_target_pos(target: Planet, world: World, turns: int) -> Optional[Tuple[float, float]]:
    if int(target.id) in world.comet_ids:
        return _predict_comet_pos(int(target.id), world, turns)
    return _predict_planet_pos(target, world, turns)


def _estimate_arrival(src: Planet, target: Planet, ships: int, tx: float, ty: float) -> Optional[Tuple[float, int, float]]:
    angle = math.atan2(float(ty) - float(src.y), float(tx) - float(src.x))
    raw_dist = math.hypot(float(tx) - float(src.x), float(ty) - float(src.y))
    travel_dist = max(0.0, raw_dist - float(src.radius) - LAUNCH_CLEARANCE - float(target.radius))
    if _path_too_close_to_sun(src, angle, travel_dist):
        return None
    turns = max(1, int(math.ceil(travel_dist / _fleet_speed(max(1, int(ships))))))
    return angle, turns, travel_dist


def _safe_plan_shot(world: World, src: Planet, target: Planet, ships: int) -> Optional[Tuple[float, int]]:
    direct = _estimate_arrival(src, target, ships, float(target.x), float(target.y))
    if direct is None:
        return _search_safe_intercept(world, src, target, ships)

    tx = float(target.x)
    ty = float(target.y)
    est = direct
    for _ in range(AIM_ITERATIONS):
        _angle_est, turns, _dist_est = est
        pos = _predict_target_pos(target, world, turns)
        if pos is None:
            return None
        ntx, nty = pos
        next_est = _estimate_arrival(src, target, ships, ntx, nty)
        if next_est is None:
            return _search_safe_intercept(world, src, target, ships)
        if abs(ntx - tx) < 0.25 and abs(nty - ty) < 0.25 and abs(int(next_est[1]) - int(turns)) <= INTERCEPT_TOLERANCE:
            return float(next_est[0]), int(next_est[1])
        tx, ty = ntx, nty
        est = next_est
    return float(est[0]), int(est[1])


def _search_safe_intercept(world: World, src: Planet, target: Planet, ships: int) -> Optional[Tuple[float, int]]:
    max_turns = SHOT_HORIZON
    if int(target.id) in world.comet_ids:
        max_turns = min(max_turns, max(0, _comet_remaining_life(int(target.id), world) - 1))
    best = None
    best_score = None
    for cand_turn in range(1, max_turns + 1):
        pos = _predict_target_pos(target, world, cand_turn)
        if pos is None:
            continue
        est = _estimate_arrival(src, target, ships, pos[0], pos[1])
        if est is None:
            continue
        angle, turns, _dist_est = est
        if abs(int(turns) - int(cand_turn)) > INTERCEPT_TOLERANCE:
            continue
        score = (abs(int(turns) - int(cand_turn)), int(turns), int(cand_turn))
        if best is None or score < best_score:
            best = (float(angle), int(turns))
            best_score = score
    return best


def _fleet_target_planet(fleet: Fleet, planets: Sequence[Planet]) -> Tuple[Optional[Planet], Optional[int]]:
    best_planet = None
    best_time = 1e9
    dir_x = math.cos(float(fleet.angle))
    dir_y = math.sin(float(fleet.angle))
    speed = _fleet_speed(float(fleet.ships))
    for planet in planets:
        dx = float(planet.x) - float(fleet.x)
        dy = float(planet.y) - float(fleet.y)
        proj = dx * dir_x + dy * dir_y
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        r2 = float(planet.radius) * float(planet.radius)
        if perp_sq >= r2:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, r2 - perp_sq)))
        turns = hit_d / max(1e-6, speed)
        if turns <= SHOT_HORIZON and turns < best_time:
            best_time = turns
            best_planet = planet
    if best_planet is None:
        return None, None
    return best_planet, int(math.ceil(best_time))


def _count_players(planets: Sequence[Planet], fleets: Sequence[Fleet], player: int) -> int:
    owners = {player}
    for p in planets:
        if p.owner >= 0:
            owners.add(int(p.owner))
    for f in fleets:
        if f.owner >= 0:
            owners.add(int(f.owner))
    return max(2, max(owners) + 1 if owners else 2)


def _build_world(obs) -> World:
    player = int(_get(obs, "player", 0) or 0)
    step = int(_get(obs, "step", 0) or 0)
    raw_planets = list(_get(obs, "planets", []) or [])
    raw_fleets = list(_get(obs, "fleets", []) or [])
    raw_initial = list(_get(obs, "initial_planets", []) or [])
    raw_comets = list(_get(obs, "comets", []) or [])
    comet_ids = set(int(x) for x in (_get(obs, "comet_planet_ids", []) or []))
    angular_velocity = float(_get(obs, "angular_velocity", 0.0) or 0.0)
    planets = [_parse_planet(p) for p in raw_planets]
    fleets = [_parse_fleet(f) for f in raw_fleets]
    initial_planets = [_parse_planet(p) for p in raw_initial] if raw_initial else list(planets)
    initial_by_id = {p.id: p for p in initial_planets}
    comets = []
    for group in raw_comets:
        comets.append({
            "planet_ids": [int(x) for x in (_get(group, "planet_ids", []) or [])],
            "paths": _get(group, "paths", []) or [],
            "path_index": int(_get(group, "path_index", 0) or 0),
        })
    planet_by_id = {p.id: p for p in planets}

    my_planets = [p for p in planets if p.owner == player]
    enemy_planets = [p for p in planets if p.owner >= 0 and p.owner != player]
    neutral_planets = [p for p in planets if p.owner == -1]

    owner_strength: Dict[int, float] = defaultdict(float)
    owner_production: Dict[int, float] = defaultdict(float)
    for p in planets:
        if p.owner >= 0:
            owner_strength[int(p.owner)] += float(p.ships)
            owner_production[int(p.owner)] += float(p.production)
    for f in fleets:
        if f.owner >= 0:
            owner_strength[int(f.owner)] += float(f.ships)

    my_total = float(owner_strength.get(player, 0.0))
    my_prod = float(owner_production.get(player, 0.0))
    enemy_total = sum(v for o, v in owner_strength.items() if o not in (-1, player))
    enemy_prod = sum(v for o, v in owner_production.items() if o not in (-1, player))

    explicit_players = _get(obs, "n_players", None) or _get(obs, "players", None)
    try:
        n_players = max(2, int(explicit_players)) if explicit_players is not None else _count_players(planets, fleets, player)
    except Exception:
        n_players = _count_players(planets, fleets, player)
    is_four_player = n_players >= 4
    remaining_steps = max(1, TOTAL_STEPS - step)
    opening_limit = 80 if is_four_player else 60
    is_opening = step < opening_limit
    is_late = remaining_steps < 60
    is_very_late = remaining_steps < 25

    incoming_enemy: Dict[int, float] = defaultdict(float)
    incoming_friendly: Dict[int, float] = defaultdict(float)
    incoming_enemy_eta: Dict[int, int] = {}
    for fleet in fleets:
        target, eta = _fleet_target_planet(fleet, planets)
        if target is None or eta is None:
            continue
        if int(fleet.owner) == player:
            incoming_friendly[int(target.id)] += float(fleet.ships)
        elif int(fleet.owner) >= 0:
            incoming_enemy[int(target.id)] += float(fleet.ships)
            pid = int(target.id)
            incoming_enemy_eta[pid] = min(int(eta), incoming_enemy_eta.get(pid, int(eta)))

    threatened: set[int] = set()
    doomed: set[int] = set()
    enemy_assets = [p for p in enemy_planets]
    for mine in my_planets:
        if not enemy_assets:
            nearest = None
        else:
            nearest = min(enemy_assets, key=lambda p: _dist(mine, p))
        if nearest is not None:
            d = _dist(mine, nearest)
            ship_ratio = float(nearest.ships) / max(1.0, float(mine.ships))
            pressure = d / max(1.0, float(mine.radius + nearest.radius))
            if pressure < 1.8 and ship_ratio > 0.45:
                threatened.add(mine.id)
            if pressure < 1.25 and ship_ratio > 0.85:
                doomed.add(mine.id)
        enemy_in = float(incoming_enemy.get(int(mine.id), 0.0))
        if enemy_in > 0.0:
            eta = int(incoming_enemy_eta.get(int(mine.id), 1))
            friendly_in = float(incoming_friendly.get(int(mine.id), 0.0))
            projected = float(mine.ships) + max(0, eta) * float(mine.production) + friendly_in
            if enemy_in >= max(4.0, 0.30 * projected):
                threatened.add(mine.id)
            if enemy_in + 2.0 >= projected:
                doomed.add(mine.id)

    enemy_strengths = {o: s for o, s in owner_strength.items() if o not in (-1, player)}
    weakest_enemy_id = min(enemy_strengths, key=enemy_strengths.get) if enemy_strengths else None
    strongest_enemy_id = max(enemy_strengths, key=enemy_strengths.get) if enemy_strengths else None

    return World(
        player=player,
        step=step,
        remaining_steps=remaining_steps,
        planets=planets,
        fleets=fleets,
        planet_by_id=planet_by_id,
        my_planets=my_planets,
        enemy_planets=enemy_planets,
        neutral_planets=neutral_planets,
        initial_by_id=initial_by_id,
        angular_velocity=angular_velocity,
        comets=comets,
        comet_ids=comet_ids,
        owner_strength=dict(owner_strength),
        owner_production=dict(owner_production),
        my_total=my_total,
        enemy_total=enemy_total,
        my_prod=my_prod,
        enemy_prod=enemy_prod,
        is_four_player=is_four_player,
        is_opening=is_opening,
        is_late=is_late,
        is_very_late=is_very_late,
        threatened_candidates=threatened,
        doomed_candidates=doomed,
        incoming_enemy=dict(incoming_enemy),
        incoming_friendly=dict(incoming_friendly),
        incoming_enemy_eta=dict(incoming_enemy_eta),
        weakest_enemy_id=weakest_enemy_id,
        strongest_enemy_id=strongest_enemy_id,
    )


def _front_count(world: World) -> int:
    count = 0
    for mine in world.my_planets:
        if any(_dist(mine, enemy) <= 34.0 + mine.radius + enemy.radius for enemy in world.enemy_planets):
            count += 1
    return count


def _center_control(planets: Iterable[Planet], owner: int) -> float:
    value = 0.0
    total = 0.0
    for p in planets:
        weight = max(0.0, 1.0 - math.hypot(p.x - 50.0, p.y - 50.0) / 70.0) * max(1.0, p.production)
        total += weight
        if p.owner == owner:
            value += weight
    return value / max(1e-6, total)


def _frontier_spread(world: World) -> float:
    if len(world.my_planets) <= 1:
        return 0.0
    dists = []
    for p in world.my_planets:
        nearest = min((_dist(p, q) for q in world.my_planets if q.id != p.id), default=0.0)
        dists.append(nearest)
    return min(1.0, float(np.mean(dists)) / 55.0)


def extract_state_features(world: World) -> np.ndarray:
    total_ships = sum(max(0.0, float(v)) for o, v in world.owner_strength.items() if o != -1)
    total_prod = sum(max(0.0, float(v)) for o, v in world.owner_production.items() if o != -1)
    total_planets = max(1, len(world.planets))
    neutral_prod = sum(float(p.production) for p in world.neutral_planets)
    neutral_ships = sum(float(p.ships) for p in world.neutral_planets)

    enemy_strengths = [s for o, s in world.owner_strength.items() if o not in (-1, world.player)]
    enemy_prods = [p for o, p in world.owner_production.items() if o not in (-1, world.player)]
    strongest_enemy = max(enemy_strengths, default=0.0)
    strongest_enemy_prod = max(enemy_prods, default=0.0)
    weakest_enemy = min(enemy_strengths, default=0.0)

    active_fronts = _front_count(world)
    own_planet_ships = sum(float(p.ships) for p in world.my_planets)
    fleet_count = len(world.fleets)
    soft_neutral = neutral_prod / max(1.0, neutral_ships + 4.0 * len(world.neutral_planets))

    ship_lead = world.my_total / max(1.0, float(strongest_enemy))
    prod_lead = world.my_prod / max(1.0, float(strongest_enemy_prod))
    finish_pressure = 0.0
    if world.enemy_planets:
        finish_pressure = min(1.5, 0.35 * ship_lead + 0.35 * prod_lead + (0.30 if world.is_late else 0.0))
    comeback = min(1.5, max(0.0, float(strongest_enemy) - world.my_total) / max(1.0, world.my_total + strongest_enemy))

    features = np.array(
        [
            world.my_total / max(1.0, float(total_ships)),
            float(world.enemy_total) / max(1.0, float(total_ships)),
            float(strongest_enemy) / max(1.0, float(total_ships)),
            world.my_prod / max(1.0, float(total_prod)),
            float(world.enemy_prod) / max(1.0, float(total_prod)),
            float(strongest_enemy_prod) / max(1.0, float(total_prod)),
            len(world.my_planets) / float(total_planets),
            len(world.enemy_planets) / float(total_planets),
            len(world.neutral_planets) / float(total_planets),
            neutral_prod / max(1.0, float(total_prod) + neutral_prod),
            min(2.0, soft_neutral),
            min(1.0, world.step / 500.0),
            min(1.0, world.remaining_steps / 500.0),
            1.0 if world.is_four_player else 0.0,
            1.0 if world.is_opening else 0.0,
            1.0 if world.is_late else 0.0,
            1.0 if world.is_very_late else 0.0,
            active_fronts / max(1.0, float(len(world.my_planets))),
            len(world.threatened_candidates) / max(1.0, float(len(world.my_planets))),
            len(world.doomed_candidates) / max(1.0, float(len(world.my_planets))),
            own_planet_ships / max(1.0, world.my_total),
            min(2.0, fleet_count / max(1.0, len(world.planets))),
            float(weakest_enemy) / max(1.0, world.my_total),
            min(2.0, ship_lead / 1.5),
            min(2.0, prod_lead / 1.5),
            finish_pressure,
            comeback,
            world.my_prod / max(1.0, len(world.my_planets)),
            float(world.enemy_prod) / max(1.0, len(world.enemy_planets) or 1),
            _center_control(world.planets, world.player),
            _frontier_spread(world),
            1.0,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, copy=False)


def match_move_target(move: List, world: World) -> Optional[Tuple[Planet, float]]:
    if len(move) != 3:
        return None
    sid, angle, _ships = int(move[0]), float(move[1]), int(move[2])
    src = world.planet_by_id.get(sid)
    if src is None:
        return None
    best = None
    best_score = 1e18
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for p in world.planets:
        if p.id == sid:
            continue
        dx = float(p.x - src.x)
        dy = float(p.y - src.y)
        dist = math.hypot(dx, dy)
        if dist <= 1e-6:
            continue
        proj = dx * cos_a + dy * sin_a
        if proj <= 0:
            continue
        target_angle = math.atan2(dy, dx)
        angle_gap = _angle_delta(angle, target_angle)
        perp = abs(dx * sin_a - dy * cos_a)
        if perp > float(p.radius) + 8.0 and angle_gap > 0.34:
            continue
        score = angle_gap * 50.0 + perp - 0.03 * proj
        if score < best_score:
            best = p
            best_score = score
    if best is None:
        return None
    eta = math.hypot(float(best.x - src.x), float(best.y - src.y)) / 4.0
    return best, eta


def extract_plan_features(candidate: PlanCandidate, world: World) -> np.ndarray:
    moves = candidate.moves or []
    my_total = max(1.0, float(world.my_total))
    total_send = float(sum(max(0, int(m[2])) for m in moves if len(m) == 3))
    attack = expand = defense = 0
    transfer_ships = 0.0
    eta_sum = 0.0
    eta_n = 0
    target_prod = 0.0
    target_cost = 0.0
    targets = set()
    weakest_enemy = world.weakest_enemy_id
    weak_focus = 0
    high_prod_focus = 0
    neutral_focus = 0
    enemy_focus = 0

    for move in moves:
        match = match_move_target(move, world)
        if match is None:
            continue
        target, eta = match
        targets.add(int(target.id))
        eta_sum += float(eta)
        eta_n += 1
        target_prod += max(0.0, float(target.production))
        target_cost += max(0.0, float(target.ships))
        if target.owner == world.player:
            defense += 1
            transfer_ships += max(0, int(move[2]))
        elif target.owner == -1:
            expand += 1
            neutral_focus += 1
        else:
            attack += 1
            enemy_focus += 1
            if target.owner == weakest_enemy:
                weak_focus += 1
            if float(target.production) >= 3.0:
                high_prod_focus += 1

    own_planet_ships = sum(float(p.ships) for p in world.my_planets)
    garrison_after = max(0.0, own_planet_ships - total_send) / my_total
    overcommit = max(0.0, total_send / my_total - (0.74 if world.is_late else 0.58))
    undercommit = 0.0
    if world.is_late and world.enemy_planets:
        undercommit = max(0.0, 0.18 - total_send / my_total)

    active_front_ratio = 0.0
    if world.my_planets and world.enemy_planets:
        fronts = 0
        for mine in world.my_planets:
            if any(math.hypot(mine.x - enemy.x, mine.y - enemy.y) <= 34.0 for enemy in world.enemy_planets):
                fronts += 1
        active_front_ratio = fronts / max(1.0, len(world.my_planets))

    n_moves = max(1, len(moves))
    ptype_idx = PLAN_TYPE_TO_INDEX.get(candidate.plan_type, 0)
    feat = np.array(
        [
            min(2.0, len(moves) / 14.0),
            min(2.0, total_send / my_total),
            attack / n_moves,
            expand / n_moves,
            defense / n_moves,
            min(2.0, transfer_ships / my_total),
            min(2.0, (eta_sum / max(1, eta_n)) / 35.0),
            min(2.0, target_prod / max(1.0, float(world.my_prod) + 1.0)),
            min(2.0, target_cost / my_total),
            min(1.0, len(targets) / max(1.0, len(world.planets))),
            min(2.0, overcommit),
            min(2.0, undercommit),
            1.0 if candidate.plan_type == "endgame_finisher" else 0.0,
            1.0 if candidate.plan_type == "resource_denial" else 0.0,
            1.0 if candidate.plan_type == "multi_step_trap" else 0.0,
            1.0 if candidate.plan_type == "probe" else 0.0,
            1.0 if world.is_opening else 0.0,
            1.0 if world.is_late or world.is_very_late else 0.0,
            1.0 if world.is_four_player else 0.0,
            active_front_ratio,
            min(2.0, garrison_after),
            weak_focus / n_moves,
            high_prod_focus / n_moves,
            neutral_focus / n_moves,
            enemy_focus / n_moves,
            ptype_idx / max(1.0, len(PLAN_TYPES) - 1.0),
            float(candidate.base_score),
            1.0,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(feat, copy=False)


EMBEDDED_V9_GUARDIAN_8H_VPS_BEST_POLICY_B64 = """
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
c29uLm5weVBLBQYAAAAABgAGAGEBAAC+EwAAAAA=
"""


@dataclass
class V9Weights:
    state_plan_w: np.ndarray
    plan_w: np.ndarray
    plan_bias: np.ndarray
    interaction_w: np.ndarray

    @classmethod
    def from_npz(cls, npz) -> "V9Weights":
        return cls(
            np.asarray(npz["state_plan_w"], dtype=np.float32),
            np.asarray(npz["plan_w"], dtype=np.float32),
            np.asarray(npz["plan_bias"], dtype=np.float32),
            np.asarray(npz["interaction_w"], dtype=np.float32),
        )

    @classmethod
    def defaults(cls) -> "V9Weights":
        state_plan_w = np.zeros((len(PLAN_TYPES), len(STATE_FEATURE_NAMES)), dtype=np.float32)
        plan_w = np.zeros(len(PLAN_FEATURE_NAMES), dtype=np.float32)
        plan_bias = np.zeros(len(PLAN_TYPES), dtype=np.float32)
        interaction_w = np.array([0.55, 0.44, 0.50, 0.35, -0.62, -0.35, 0.38, 0.30], dtype=np.float32)
        return cls(state_plan_w, plan_w, plan_bias, interaction_w)


def _load_weights() -> V9Weights:
    try:
        blob = base64.b64decode(EMBEDDED_V9_GUARDIAN_8H_VPS_BEST_POLICY_B64.encode("ascii"))
        with np.load(io.BytesIO(blob), allow_pickle=False) as npz:
            return V9Weights.from_npz(npz)
    except Exception:
        return V9Weights.defaults()


def _sim_score(estimate, rollout_weight: float, uncertainty_penalty: float) -> float:
    if estimate is None:
        return 0.0
    useful = (
        getattr(estimate, "mean_delta", 0.0)
        + 0.35 * getattr(estimate, "worst_delta", 0.0)
        + 0.58 * getattr(estimate, "margin_delta", 0.0)
        + 0.12 * getattr(estimate, "production_delta", 0.0)
        + 0.08 * getattr(estimate, "planet_delta", 0.0)
        + 1.25 * getattr(estimate, "finish_bonus", 0.0)
    )
    return rollout_weight * useful - uncertainty_penalty * getattr(estimate, "uncertainty", 0.0)


class V9Policy:
    def __init__(self, weights: Optional[V9Weights] = None):
        self.weights = weights if weights is not None else _load_weights()

    def score_candidates(
        self,
        world: World,
        candidates: Iterable[PlanCandidate],
        *,
        estimates: Optional[Dict[str, object]] = None,
        rollout_weight: float = 0.42,
        uncertainty_penalty: float = 0.20,
        injected_plan_bias: Optional[Dict[str, float]] = None,
        front_pressure_plan_bias: float = 0.12,
        front_pressure_attack_penalty: float = 0.12,
    ) -> List[Tuple[PlanCandidate, float, np.ndarray]]:
        state_feat = extract_state_features(world)
        scored: List[Tuple[PlanCandidate, float, np.ndarray]] = []
        injected_plan_bias = injected_plan_bias or {}
        for candidate in candidates:
            plan_feat = extract_plan_features(candidate, world)
            pidx = PLAN_TYPE_TO_INDEX.get(candidate.plan_type, 0)
            linear = float(self.weights.state_plan_w[pidx] @ state_feat)
            linear += float(self.weights.plan_w @ plan_feat)
            linear += float(self.weights.plan_bias[pidx])
            linear += float(injected_plan_bias.get(candidate.plan_type, 0.0))

            opening = state_feat[STATE_FEATURE_INDEX["is_opening"]]
            late = max(state_feat[STATE_FEATURE_INDEX["is_late"]], state_feat[STATE_FEATURE_INDEX["is_very_late"]])
            finish_pressure = state_feat[STATE_FEATURE_INDEX["finish_pressure"]]
            comeback = state_feat[STATE_FEATURE_INDEX["comeback_pressure"]]
            threatened = state_feat[STATE_FEATURE_INDEX["threatened_ratio"]]
            fronts = state_feat[STATE_FEATURE_INDEX["active_front_ratio"]]
            four_p = state_feat[STATE_FEATURE_INDEX["is_4p"]]

            attack = plan_feat[PLAN_FEATURE_INDEX["attack_move_frac"]]
            expand = plan_feat[PLAN_FEATURE_INDEX["expand_move_frac"]]
            defense = plan_feat[PLAN_FEATURE_INDEX["defense_move_frac"]]
            transfer = plan_feat[PLAN_FEATURE_INDEX["transfer_ship_frac"]]
            overcommit = plan_feat[PLAN_FEATURE_INDEX["overcommit_risk"]]
            undercommit = plan_feat[PLAN_FEATURE_INDEX["undercommit_risk"]]
            weak_focus = plan_feat[PLAN_FEATURE_INDEX["weak_enemy_focus"]]
            high_prod = plan_feat[PLAN_FEATURE_INDEX["high_prod_focus"]]

            iw = self.weights.interaction_w
            nonlinear = 0.0
            nonlinear += float(iw[0]) * math.tanh(float(finish_pressure * attack + late * weak_focus))
            nonlinear += float(iw[1]) * math.tanh(float(opening * expand + plan_feat[PLAN_FEATURE_INDEX["neutral_focus"]]))
            nonlinear += float(iw[2]) * math.tanh(float(four_p * (transfer + high_prod)))
            nonlinear += float(iw[3]) * math.tanh(float((threatened + fronts) * defense))
            nonlinear += float(iw[4]) * float(overcommit) * (1.0 + float(threatened + fronts))
            nonlinear += float(iw[5]) * float(undercommit) * (1.0 + float(late))
            nonlinear += float(iw[6]) * math.tanh(float(comeback * (defense + expand)))
            nonlinear += float(iw[7]) * math.tanh(float(high_prod + weak_focus))

            safety = 0.0
            metadata_bonus = 0.0
            if four_p > 0.5:
                metadata = candidate.metadata or {}
                backbone = float(metadata.get("backbone", 0.0))
                front_lock = float(metadata.get("front_lock", 0.0))
                consolidation_threshold = float(metadata.get("consolidation_threshold", 0.0))
                staged_finisher = float(metadata.get("staged_finisher", 0.0))
                metadata_bonus += 0.36 * backbone
                metadata_bonus += (0.12 + 0.14 * float(fronts)) * front_lock
                metadata_bonus += 0.20 * consolidation_threshold
                if candidate.plan_type == "staging_transfer":
                    metadata_bonus += 0.10 + 0.16 * backbone + 0.08 * front_lock
                if backbone > 0.0 and transfer >= 0.30 and attack < 0.35:
                    metadata_bonus += 0.08
                metadata_bonus += (0.12 + 0.30 * float(finish_pressure)) * staged_finisher
                if 6 <= len(world.my_planets) < 15 and not world.is_late:
                    metadata_bonus += 0.18 * float(transfer + defense)
                    if attack > 0.45 and transfer < 0.12:
                        metadata_bonus -= 0.12
                if fronts > 0.36 and not world.is_late:
                    front_pressure = float((fronts - 0.36) / 0.64)
                    if candidate.plan_type in ("defensive_consolidation", "staging_transfer", "reserve_hold"):
                        metadata_bonus += float(front_pressure_plan_bias) * (0.75 + front_pressure)
                    if backbone > 0.0 or front_lock > 0.0 or consolidation_threshold > 0.0:
                        metadata_bonus += 0.08 * front_pressure
                    finisher_ready = candidate.plan_type == "endgame_finisher" and (staged_finisher > 0.0 or finish_pressure > 1.0)
                    if not finisher_ready and candidate.plan_type in (
                        "resource_denial",
                        "delayed_strike",
                        "multi_step_trap",
                        "opportunistic_snipe",
                        "aggressive_expansion",
                    ):
                        metadata_bonus -= float(front_pressure_attack_penalty) * (0.60 + front_pressure)
                    if attack > 0.35 and transfer < 0.18 and not finisher_ready:
                        metadata_bonus -= 0.10 + 0.10 * front_pressure
                if fronts > 0.42 and candidate.plan_type not in ("defensive_consolidation", "staging_transfer", "reserve_hold"):
                    metadata_bonus -= 0.10 * float(fronts)
                focus = metadata.get("focus_enemy_id")
                if focus is not None and world.weakest_enemy_id is not None and int(focus) == int(world.weakest_enemy_id):
                    metadata_bonus += 0.05
            if candidate.plan_type == "reserve_hold" and not world.threatened_candidates and not world.doomed_candidates:
                safety -= 0.22
            if candidate.plan_type == "probe" and (world.is_late or finish_pressure > 0.8):
                safety -= 0.20
            if not candidate.moves and candidate.plan_type != "reserve_hold":
                safety -= 0.55

            estimate = estimates.get(candidate.name) if estimates else None
            score = linear + nonlinear + safety + metadata_bonus + _sim_score(estimate, rollout_weight, uncertainty_penalty)
            scored.append((candidate, float(score), plan_feat))
        return scored

    def choose(
        self,
        world: World,
        candidates: List[PlanCandidate],
        *,
        estimates: Optional[Dict[str, object]] = None,
        exploration_rate: float = 0.0,
        rollout_weight: float = 0.42,
        uncertainty_penalty: float = 0.20,
        injected_plan_bias: Optional[Dict[str, float]] = None,
        front_pressure_plan_bias: float = 0.12,
        front_pressure_attack_penalty: float = 0.12,
        rng: Optional[random.Random] = None,
    ) -> Tuple[PlanCandidate, List[Tuple[PlanCandidate, float, np.ndarray]]]:
        scored = self.score_candidates(
            world,
            candidates,
            estimates=estimates,
            rollout_weight=rollout_weight,
            uncertainty_penalty=uncertainty_penalty,
            injected_plan_bias=injected_plan_bias,
            front_pressure_plan_bias=front_pressure_plan_bias,
            front_pressure_attack_penalty=front_pressure_attack_penalty,
        )
        if not scored:
            return PlanCandidate("empty", [], "reserve_hold"), []
        scored.sort(key=lambda item: item[1], reverse=True)
        best = scored[0][0] if scored[0][0].moves else next((c for c, _s, _f in scored if c.moves), scored[0][0])
        rng = rng or random.Random()
        if exploration_rate > 0 and len(scored) > 1 and rng.random() < exploration_rate:
            pool = scored[: min(5, len(scored))]
            vals = np.array([s for _c, s, _f in pool], dtype=np.float32)
            vals = vals - float(np.max(vals))
            probs = np.exp(vals / 0.28)
            probs = probs / max(1e-8, float(np.sum(probs)))
            pick = int(rng.choices(range(len(pool)), weights=probs.tolist(), k=1)[0])
            best = pool[pick][0]
        return best, scored


def _planet_value(src: Planet, target: Planet, focus_enemy: Optional[int], world: World) -> float:
    dist = _dist(src, target)
    value = 20.0 * float(target.production) - 0.32 * float(target.ships) - 0.28 * dist
    if target.owner == -1:
        value += 8.0
    elif target.owner != world.player:
        value += 15.0
    if focus_enemy is not None and int(target.owner) == int(focus_enemy):
        value += 10.0
    elif world.is_four_player and target.owner not in (-1, world.player):
        value -= 6.0
    return value


def _active_front_count(world: World, focus_enemy: Optional[int]) -> int:
    enemies = [p for p in world.enemy_planets if focus_enemy is None or int(p.owner) == int(focus_enemy)]
    if not enemies:
        enemies = list(world.enemy_planets)
    count = 0
    for mine in world.my_planets:
        if any(_dist(mine, enemy) <= 36.0 + float(mine.radius) + float(enemy.radius) for enemy in enemies):
            count += 1
    return count


def _select_focus_enemy(world: World, current: Optional[int]) -> Optional[int]:
    if current is not None and any(int(p.owner) == int(current) for p in world.enemy_planets):
        return int(current)
    if world.weakest_enemy_id is not None:
        return int(world.weakest_enemy_id)
    owners = [
        int(o)
        for o in world.owner_strength
        if o not in (-1, world.player) and any(int(p.owner) == int(o) for p in world.enemy_planets)
    ]
    return min(owners, key=lambda o: float(world.owner_strength.get(o, 0.0)) + 18.0 * float(world.owner_production.get(o, 0.0))) if owners else None


def _front_anchor(world: World, focus_enemy: Optional[int]) -> Optional[Planet]:
    if not world.my_planets:
        return None
    targets = [p for p in world.enemy_planets if focus_enemy is None or int(p.owner) == int(focus_enemy)]
    if not targets:
        targets = list(world.enemy_planets or world.neutral_planets)
    if not targets:
        return max(world.my_planets, key=lambda p: p.ships)
    safe = [p for p in world.my_planets if p.id not in world.doomed_candidates] or list(world.my_planets)
    return min(
        safe,
        key=lambda p: (
            min(_dist(p, t) for t in targets),
            p.id in world.threatened_candidates,
            -float(p.production),
            -float(p.ships),
        ),
    )


def _available_ships(world: World, planet: Planet, front_pressure: int) -> int:
    reserve = 0.42 if planet.id in world.threatened_candidates else 0.58
    if world.is_four_player and front_pressure > 0:
        reserve = min(reserve, 0.48)
    reserve_ships = float(planet.ships) * reserve
    enemy_in = float(world.incoming_enemy.get(int(planet.id), 0.0))
    friendly_in = float(world.incoming_friendly.get(int(planet.id), 0.0))
    if enemy_in > 0.0:
        eta = int(world.incoming_enemy_eta.get(int(planet.id), 1))
        expected_growth = max(0, eta) * float(planet.production)
        reserve_ships = max(reserve_ships, enemy_in - friendly_in - expected_growth + 4.0)
    return max(0, int(float(planet.ships) - max(0.0, reserve_ships)))


def _capture_need(target: Planet, world: World, *, family: str) -> int:
    if target.owner == world.player:
        enemy_in = float(world.incoming_enemy.get(int(target.id), 0.0))
        friendly_in = float(world.incoming_friendly.get(int(target.id), 0.0))
        return max(0, int(math.ceil(enemy_in - friendly_in + 4.0)))
    if target.owner == -1:
        return int(math.ceil(float(target.ships) + 1.0 * float(target.production) + 1.0))
    pressure = 1.2 if family != "opportunistic_snipe" else 0.8
    return int(math.ceil(float(target.ships) + pressure * float(target.production) + 2.0))


def _add_move(moves: List[List], world: World, src: Planet, target: Planet, ships: int, *, bias: float = 0.0) -> bool:
    ships = int(ships)
    if ships <= 0:
        return False
    shot = _safe_plan_shot(world, src, target, ships)
    if shot is None:
        return False
    angle, turns = shot
    if world.is_very_late and turns > world.remaining_steps - 1:
        return False
    if world.is_late and turns > world.remaining_steps - 4:
        return False
    moves.append([int(src.id), float(angle + bias), int(ships)])
    return True


def _pick_targets(world: World, focus_enemy: Optional[int], *, family: str) -> List[Planet]:
    targets = list(world.neutral_planets) + list(world.enemy_planets)
    if family == "resource_denial":
        targets = sorted(targets, key=lambda t: (t.owner == world.player, -(t.production if t.owner != -1 else 0.0), float(t.ships)))
    elif family == "endgame_finisher":
        targets = sorted(targets, key=lambda t: (t.owner != world.weakest_enemy_id, float(t.ships), -float(t.production)))
    elif family == "opportunistic_snipe":
        targets = sorted(targets, key=lambda t: (float(t.ships), -float(t.production), _planet_value(world.my_planets[0], t, focus_enemy, world)))
    elif family == "probe":
        targets = sorted(targets, key=lambda t: (-float(t.production), float(t.ships)))
    else:
        targets = sorted(targets, key=lambda t: (-_planet_value(world.my_planets[0], t, focus_enemy, world), -float(t.production), float(t.ships)))
    return targets


def _single_source_attack(world: World, focus_enemy: Optional[int], *, family: str, max_moves: int = 3, min_send_scale: float = 1.0) -> List[List]:
    moves: List[List] = []
    if not world.my_planets:
        return moves
    sources = sorted(world.my_planets, key=lambda p: float(p.ships), reverse=True)
    targets = _pick_targets(world, focus_enemy, family=family)
    for src in sources[:4]:
        left = _available_ships(world, src, 1 if family in ("resource_denial", "endgame_finisher") else 0)
        if left < 5:
            continue
        for target in targets[:10]:
            if target.owner == world.player:
                continue
            needed = _capture_need(target, world, family=family)
            if world.is_four_player and focus_enemy is not None and target.owner not in (-1, world.player) and int(target.owner) != int(focus_enemy):
                needed = int(needed * 1.15)
            if family != "probe" and left < needed:
                continue
            send = min(left, max(4, int(needed * min_send_scale)))
            if send <= 0:
                continue
            if _add_move(moves, world, src, target, send):
                break
        if len(moves) >= max_moves:
            break
    return moves


def _reinforce_threats(world: World, focus_enemy: Optional[int]) -> List[List]:
    moves: List[List] = []
    if not world.threatened_candidates and not world.doomed_candidates:
        return moves
    anchor = _front_anchor(world, focus_enemy)
    if anchor is None:
        return moves
    threatened = [p for p in world.my_planets if p.id in world.threatened_candidates or p.id in world.doomed_candidates]
    sources = sorted(
        [p for p in world.my_planets if p.id not in world.doomed_candidates],
        key=lambda p: (p.id in world.threatened_candidates, -float(p.ships), _dist(p, anchor)),
        reverse=True,
    )
    for target in threatened[:3]:
        need = int(float(target.ships) * 0.60) + 6
        for src in sources:
            if src.id == target.id:
                continue
            left = _available_ships(world, src, 1)
            if left < 4:
                continue
            send = min(left, max(4, need))
            if _add_move(moves, world, src, target, send):
                need -= send
                if need <= 0:
                    break
    return moves


def _staging_moves(world: World, focus_enemy: Optional[int], front_pressure: int) -> List[List]:
    moves: List[List] = []
    anchor = _front_anchor(world, focus_enemy)
    if anchor is None:
        return moves
    sources = sorted(
        [p for p in world.my_planets if p.id not in world.doomed_candidates],
        key=lambda p: (_dist(p, anchor), float(p.ships)),
        reverse=True,
    )
    transfer_cap = 2 if front_pressure <= 0 else 3 if front_pressure == 1 else 4
    for src in sources:
        if len(moves) >= transfer_cap:
            break
        left = _available_ships(world, src, front_pressure)
        if left < 8:
            continue
        if _dist(src, anchor) < 6.0:
            continue
        send = int(left * (0.42 if front_pressure else 0.34))
        if send >= 6:
            _add_move(moves, world, src, anchor, send)
    return moves


def _consolidate(world: World, focus_enemy: Optional[int]) -> List[List]:
    moves: List[List] = []
    anchor = _front_anchor(world, focus_enemy)
    if anchor is None:
        return moves
    interior = sorted(
        [p for p in world.my_planets if p.id not in world.threatened_candidates and p.id not in world.doomed_candidates],
        key=lambda p: (_dist(p, anchor), -float(p.production), -float(p.ships)),
        reverse=True,
    )
    for src in interior[:3]:
        left = _available_ships(world, src, 0)
        if left < 10:
            continue
        send = int(left * 0.30)
        if send >= 6:
            _add_move(moves, world, src, anchor, send)
    return moves


def _build_candidates(world: World, focus_enemy: Optional[int]) -> List[PlanCandidate]:
    front_count = _active_front_count(world, focus_enemy)
    front_pressure = 0 if len(world.my_planets) == 0 else int(round(front_count / max(1.0, len(world.my_planets)) * 3))
    anchor = _front_anchor(world, focus_enemy)
    anchor_id = int(anchor.id) if anchor is not None else None

    balanced = _single_source_attack(world, focus_enemy, family="balanced", max_moves=2, min_send_scale=1.00)
    if anchor is not None and world.my_planets:
        for src in sorted(world.my_planets, key=lambda p: float(p.ships), reverse=True)[:2]:
            if src.id != anchor.id and _available_ships(world, src, front_pressure) >= 8:
                _add_move(balanced, world, src, anchor, max(6, int(_available_ships(world, src, front_pressure) * 0.22)))

    aggressive = _single_source_attack(world, focus_enemy, family="aggressive_expansion", max_moves=3, min_send_scale=0.92)
    delayed = _single_source_attack(world, focus_enemy, family="delayed_strike", max_moves=2, min_send_scale=1.08)
    denial = _single_source_attack(world, focus_enemy, family="resource_denial", max_moves=3, min_send_scale=1.05)
    strongest_ships = max((s for o, s in world.owner_strength.items() if o not in (-1, world.player)), default=0.0)
    strongest_prod = max((p for o, p in world.owner_production.items() if o not in (-1, world.player)), default=0.0)
    finisher_ready = (
        world.is_late
        or world.is_very_late
        or world.my_total >= max(1.0, strongest_ships) * 1.20
        or world.my_prod >= max(1.0, strongest_prod) * 1.20
    )
    finisher = (
        _single_source_attack(world, world.weakest_enemy_id or focus_enemy, family="endgame_finisher", max_moves=3, min_send_scale=1.12)
        if finisher_ready
        else []
    )
    opportunistic = _single_source_attack(world, focus_enemy, family="opportunistic_snipe", max_moves=3, min_send_scale=0.90)
    probe = _single_source_attack(world, focus_enemy, family="probe", max_moves=2, min_send_scale=0.70)
    defensive = _reinforce_threats(world, focus_enemy) + _consolidate(world, focus_enemy)
    staging = _staging_moves(world, focus_enemy, front_pressure)
    trap = _staging_moves(world, focus_enemy, front_pressure) + _single_source_attack(world, focus_enemy, family="multi_step_trap", max_moves=1, min_send_scale=0.98)
    reserve = _reinforce_threats(world, focus_enemy)

    candidates = [
        PlanCandidate(
            "balanced",
            balanced[:12],
            "balanced",
            0.05,
            {"active_fronts": float(front_count), "focus_enemy_id": float(focus_enemy or -1), "front_anchor_id": float(anchor_id or -1)},
        ),
        PlanCandidate(
            "aggressive_expansion",
            aggressive[:12],
            "aggressive_expansion",
            0.12,
            {"active_fronts": float(front_count), "focus_enemy_id": float(focus_enemy or -1), "front_anchor_id": float(anchor_id or -1)},
        ),
        PlanCandidate(
            "delayed_strike",
            delayed[:12],
            "delayed_strike",
            0.08,
            {"active_fronts": float(front_count), "focus_enemy_id": float(focus_enemy or -1), "front_anchor_id": float(anchor_id or -1)},
        ),
        PlanCandidate(
            "multi_step_trap",
            trap[:12],
            "multi_step_trap",
            0.06,
            {"active_fronts": float(front_count), "focus_enemy_id": float(focus_enemy or -1), "front_anchor_id": float(anchor_id or -1), "trap": 1.0},
        ),
        PlanCandidate(
            "resource_denial",
            denial[:12],
            "resource_denial",
            0.11,
            {"active_fronts": float(front_count), "focus_enemy_id": float(focus_enemy or -1), "front_anchor_id": float(anchor_id or -1), "denial": 1.0},
        ),
        PlanCandidate(
            "endgame_finisher",
            finisher[:12],
            "endgame_finisher",
            0.04,
            {
                "active_fronts": float(front_count),
                "focus_enemy_id": float(world.weakest_enemy_id or focus_enemy or -1),
                "front_anchor_id": float(anchor_id or -1),
                "staged_finisher": 1.0 if finisher_ready else 0.0,
            },
        ),
        PlanCandidate(
            "defensive_consolidation",
            defensive[:12],
            "defensive_consolidation",
            0.09,
            {"active_fronts": float(front_count), "focus_enemy_id": float(focus_enemy or -1), "front_anchor_id": float(anchor_id or -1), "front_lock": 1.0, "consolidation_threshold": 1.0},
        ),
        PlanCandidate(
            "staging_transfer",
            staging[:12],
            "staging_transfer",
            0.20,
            {"active_fronts": float(front_count), "focus_enemy_id": float(focus_enemy or -1), "front_anchor_id": float(anchor_id or -1), "backbone": 1.0, "front_lock": 1.0},
        ),
        PlanCandidate(
            "opportunistic_snipe",
            opportunistic[:12],
            "opportunistic_snipe",
            0.08,
            {"active_fronts": float(front_count), "focus_enemy_id": float(focus_enemy or -1), "front_anchor_id": float(anchor_id or -1)},
        ),
        PlanCandidate(
            "probe",
            probe[:12],
            "probe",
            -0.02,
            {"active_fronts": float(front_count), "focus_enemy_id": float(focus_enemy or -1), "front_anchor_id": float(anchor_id or -1), "probe": 1.0},
        ),
        PlanCandidate(
            "reserve_hold",
            reserve[:12],
            "reserve_hold",
            -0.10,
            {"active_fronts": float(front_count), "focus_enemy_id": float(focus_enemy or -1), "front_anchor_id": float(anchor_id or -1), "front_lock": 0.5},
        ),
    ]

    if not candidates[-1].moves and not defensive:
        candidates[-1].moves = []
    return candidates


def plan_moves(world: World) -> List[List]:
    if not world.my_planets:
        return []
    focus_enemy = _select_focus_enemy(world, None)
    candidates = _build_candidates(world, focus_enemy)
    policy = _GLOBAL_POLICY
    best, scored = policy.choose(
        world,
        candidates,
        rollout_weight=0.0,
        uncertainty_penalty=0.0,
        front_pressure_plan_bias=0.12,
        front_pressure_attack_penalty=0.12,
    )
    if not best.moves:
        for cand, _score, _feat in scored:
            if cand.moves:
                best = cand
                break
    return best.moves[:12]


_GLOBAL_POLICY = V9Policy()


def agent(obs, config=None):
    try:
        world = _build_world(obs)
        if not world.my_planets:
            return []
        return plan_moves(world)
    except Exception:
        return []
