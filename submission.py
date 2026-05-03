"""Standalone Orbit Wars V9 submission.

This file is self-contained for Kaggle upload:
- no imports from the local repo
- V9 weights embedded from the best trained checkpoint we found locally
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


def _count_players(planets: Sequence[Planet], fleets: Sequence[Fleet], player: int) -> int:
    owners = {player}
    for p in planets:
        if p.owner >= 0:
            owners.add(int(p.owner))
    for f in fleets:
        if f.owner >= 0:
            owners.add(int(f.owner))
    return max(2, len(owners))


def _build_world(obs) -> World:
    player = int(_get(obs, "player", 0) or 0)
    step = int(_get(obs, "step", 0) or 0)
    raw_planets = list(_get(obs, "planets", []) or [])
    raw_fleets = list(_get(obs, "fleets", []) or [])
    planets = [_parse_planet(p) for p in raw_planets]
    fleets = [_parse_fleet(f) for f in raw_fleets]
    planet_by_id = {p.id: p for p in planets}

    my_planets = [p for p in planets if p.owner == player]
    enemy_planets = [p for p in planets if p.owner not in (-1, player)]
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

    n_players = _count_players(planets, fleets, player)
    is_four_player = n_players >= 4
    remaining_steps = max(1, TOTAL_STEPS - step)
    opening_limit = 80 if is_four_player else 60
    is_opening = step < opening_limit
    is_late = remaining_steps < 60
    is_very_late = remaining_steps < 25

    threatened: set[int] = set()
    doomed: set[int] = set()
    enemy_assets = [p for p in enemy_planets]
    for mine in my_planets:
        if not enemy_assets:
            continue
        nearest = min(enemy_assets, key=lambda p: _dist(mine, p))
        d = _dist(mine, nearest)
        ship_ratio = float(nearest.ships) / max(1.0, float(mine.ships))
        pressure = d / max(1.0, float(mine.radius + nearest.radius))
        if pressure < 1.8 and ship_ratio > 0.45:
            threatened.add(mine.id)
        if pressure < 1.25 and ship_ratio > 0.85:
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


EMBEDDED_V9_BACKBONE_60M_POLICY_B64 = """
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
AAB9EgAAAAA=
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
        blob = base64.b64decode(EMBEDDED_V9_BACKBONE_60M_POLICY_B64.encode("ascii"))
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
    return max(0, int(float(planet.ships) * (1.0 - reserve)))


def _add_move(moves: List[List], src: Planet, target: Planet, ships: int, *, bias: float = 0.0) -> None:
    ships = int(ships)
    if ships <= 0:
        return
    moves.append([int(src.id), float(_angle(src, target) + bias), int(ships)])


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
        for target in targets[:5]:
            if target.owner == world.player:
                continue
            needed = int(float(target.ships) + 1.2 * float(target.production) + 2)
            if target.owner == -1:
                needed = int(float(target.ships) + 1.0 * float(target.production) + 1)
            if world.is_four_player and focus_enemy is not None and target.owner not in (-1, world.player) and int(target.owner) != int(focus_enemy):
                needed = int(needed * 1.15)
            send = min(left, max(4, int(needed * min_send_scale)))
            if send <= 0:
                continue
            _add_move(moves, src, target, send)
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
            _add_move(moves, src, target, send)
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
            _add_move(moves, src, anchor, send)
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
            _add_move(moves, src, anchor, send)
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
                _add_move(balanced, src, anchor, max(6, int(_available_ships(world, src, front_pressure) * 0.22)))

    aggressive = _single_source_attack(world, focus_enemy, family="aggressive_expansion", max_moves=3, min_send_scale=0.92)
    delayed = _single_source_attack(world, focus_enemy, family="delayed_strike", max_moves=2, min_send_scale=1.08)
    denial = _single_source_attack(world, focus_enemy, family="resource_denial", max_moves=3, min_send_scale=1.05)
    finisher = _single_source_attack(world, world.weakest_enemy_id or focus_enemy, family="endgame_finisher", max_moves=3, min_send_scale=1.12)
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
            {"active_fronts": float(front_count), "focus_enemy_id": float(world.weakest_enemy_id or focus_enemy or -1), "front_anchor_id": float(anchor_id or -1), "staged_finisher": 1.0},
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
