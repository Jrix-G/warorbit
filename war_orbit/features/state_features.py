"""State-level features for V9 scoring.

The features are normalized and intentionally inexpensive. They describe phase,
resource balance, pressure, and finishing opportunity; the policy learns which
plan family should dominate in each region.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


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

FEATURE_DIM = len(STATE_FEATURE_NAMES)


def _dist(a, b) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def _front_count(world) -> int:
    count = 0
    for mine in world.my_planets:
        if any(_dist(mine, enemy) <= 34.0 + mine.radius + enemy.radius for enemy in world.enemy_planets):
            count += 1
    return count


def _center_control(planets: Iterable, owner: int) -> float:
    value = 0.0
    total = 0.0
    for p in planets:
        weight = max(0.0, 1.0 - math.hypot(p.x - 50.0, p.y - 50.0) / 70.0) * max(1.0, p.production)
        total += weight
        if p.owner == owner:
            value += weight
    return value / max(1e-6, total)


def _frontier_spread(world) -> float:
    if len(world.my_planets) <= 1:
        return 0.0
    dists = []
    for p in world.my_planets:
        nearest = min((_dist(p, q) for q in world.my_planets if q.id != p.id), default=0.0)
        dists.append(nearest)
    return min(1.0, float(np.mean(dists)) / 55.0)


def extract_state_features(world) -> np.ndarray:
    owners = [o for o in world.owner_strength if o != -1]
    total_ships = sum(max(0, int(v)) for o, v in world.owner_strength.items() if o != -1)
    total_prod = sum(max(0, int(v)) for o, v in world.owner_production.items() if o != -1)
    total_planets = max(1, len(world.planets))
    neutral_prod = sum(float(p.production) for p in world.neutral_planets)
    neutral_ships = sum(float(p.ships) for p in world.neutral_planets)

    enemy_strengths = [s for o, s in world.owner_strength.items() if o not in (-1, world.player)]
    enemy_prods = [p for o, p in world.owner_production.items() if o not in (-1, world.player)]
    strongest_enemy = max(enemy_strengths, default=0)
    strongest_enemy_prod = max(enemy_prods, default=0)
    weakest_enemy = min(enemy_strengths, default=0)

    active_fronts = _front_count(world)
    own_planet_ships = sum(float(p.ships) for p in world.my_planets)
    fleet_count = len(world.fleets)
    soft_neutral = neutral_prod / max(1.0, neutral_ships + 4.0 * len(world.neutral_planets))

    my_total = float(world.my_total)
    ship_lead = my_total / max(1.0, float(strongest_enemy))
    prod_lead = float(world.my_prod) / max(1.0, float(strongest_enemy_prod))
    finish_pressure = 0.0
    if world.enemy_planets:
        finish_pressure = min(1.5, 0.35 * ship_lead + 0.35 * prod_lead + (0.30 if world.is_late else 0.0))
    comeback = min(1.5, max(0.0, float(strongest_enemy) - my_total) / max(1.0, my_total + strongest_enemy))

    features = np.array([
        my_total / max(1.0, float(total_ships)),
        float(world.enemy_total) / max(1.0, float(total_ships)),
        float(strongest_enemy) / max(1.0, float(total_ships)),
        float(world.my_prod) / max(1.0, float(total_prod)),
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
        own_planet_ships / max(1.0, my_total),
        min(2.0, fleet_count / max(1.0, len(world.planets))),
        float(weakest_enemy) / max(1.0, my_total),
        min(2.0, ship_lead / 1.5),
        min(2.0, prod_lead / 1.5),
        finish_pressure,
        comeback,
        float(world.my_prod) / max(1.0, len(world.my_planets)),
        float(world.enemy_prod) / max(1.0, len(world.enemy_planets)),
        _center_control(world.planets, world.player),
        _frontier_spread(world),
        1.0,
    ], dtype=np.float32)
    return np.nan_to_num(features, copy=False)
