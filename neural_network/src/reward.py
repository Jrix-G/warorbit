from __future__ import annotations

from typing import Any, Dict


def compute_reward(prev_state: Dict[str, Any], action: Dict[str, Any], next_state: Dict[str, Any], terminal: bool = False) -> float:
    prev_planets = prev_state.get("planets", [])
    next_planets = next_state.get("planets", [])
    my_id = prev_state.get("my_id", 0)

    prev_my = sum(1 for p in prev_planets if p["owner"] == my_id)
    next_my = sum(1 for p in next_planets if p["owner"] == my_id)
    prev_prod = sum(p["production"] for p in prev_planets if p["owner"] == my_id)
    next_prod = sum(p["production"] for p in next_planets if p["owner"] == my_id)
    prev_ships = sum(p["ships"] for p in prev_planets if p["owner"] == my_id)
    next_ships = sum(p["ships"] for p in next_planets if p["owner"] == my_id)
    prev_enemy_planets = sum(1 for p in prev_planets if p["owner"] not in (-1, my_id))
    next_enemy_planets = sum(1 for p in next_planets if p["owner"] not in (-1, my_id))

    reward = 0.0
    reward += 1.5 * (next_my - prev_my)
    reward += 0.45 * (next_prod - prev_prod)
    reward += 0.05 * (next_ships - prev_ships)
    reward += 1.25 * (prev_enemy_planets - next_enemy_planets)
    ships_sent = float(action.get("ships", 0)) if action else 0.0
    reward += 0.02 * ships_sent
    if ships_sent <= 0:
        reward -= 0.05
    if next_my < prev_my:
        reward -= 2.0

    if terminal:
        if next_state.get("winner") == my_id:
            reward += 25.0
        elif next_state.get("winner") is not None:
            reward -= 25.0

    if next_state.get("is_four_player", False):
        reward *= 1.0
    return float(reward)
