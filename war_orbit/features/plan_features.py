"""Plan-level features and candidate container for V9."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


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

PLAN_TYPE_TO_INDEX = {name: i for i, name in enumerate(PLAN_TYPES)}

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

FEATURE_DIM = len(PLAN_FEATURE_NAMES)


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


def _angle_delta(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(d)


def match_move_target(move: List, world) -> Optional[Tuple[object, float]]:
    """Match a launch to the most likely target planet by ray alignment."""

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
    src = world.planet_by_id[sid]
    eta = math.hypot(float(best.x - src.x), float(best.y - src.y)) / 4.0
    return best, eta


def extract_plan_features(candidate: PlanCandidate, world) -> np.ndarray:
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
    close_neutral_focus = 0
    opening_capture_mass = 0
    long_opening_attack = 0
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
            if float(eta) <= 11.0:
                close_neutral_focus += 1
            if world.is_opening and int(move[2]) >= (16 if world.is_four_player else 14):
                opening_capture_mass += 1
            if world.is_opening and world.is_four_player and float(eta) > 14.0 and float(target.production) < 5.0:
                long_opening_attack += 1
        else:
            attack += 1
            enemy_focus += 1
            if world.is_opening and int(move[2]) >= (16 if world.is_four_player else 14):
                opening_capture_mass += 1
            if world.is_opening and world.is_four_player and float(eta) > 14.0 and float(target.production) < 5.0:
                long_opening_attack += 1
            if target.owner == weakest_enemy:
                weak_focus += 1
            if float(target.production) >= 3.0:
                high_prod_focus += 1

    own_planet_ships = sum(float(p.ships) for p in world.my_planets)
    garrison_after = max(0.0, own_planet_ships - total_send) / my_total
    n_moves = max(1, len(moves))
    robust_opening_expansion = (
        world.is_opening
        and neutral_focus / n_moves >= 0.50
        and opening_capture_mass / n_moves >= 0.45
        and close_neutral_focus / max(1, neutral_focus) >= (0.30 if world.is_four_player else 0.20)
    )
    if world.is_late:
        overcommit_threshold = 0.74
    elif robust_opening_expansion:
        overcommit_threshold = 0.76 if world.is_four_player else 0.72
    else:
        overcommit_threshold = 0.58
    overcommit = max(0.0, total_send / my_total - overcommit_threshold)
    if world.is_opening and world.is_four_player and long_opening_attack:
        overcommit += 0.05 * min(3.0, float(long_opening_attack))
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

    ptype_idx = PLAN_TYPE_TO_INDEX.get(candidate.plan_type, 0)
    feat = np.array([
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
    ], dtype=np.float32)
    return np.nan_to_num(feat, copy=False)
