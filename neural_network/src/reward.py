from __future__ import annotations
from typing import Any, Dict


def compute_dense_reward(
    prev_state: Dict[str, Any],
    next_state: Dict[str, Any],
    current_step: int,
    curriculum_steps: int = 50_000,
) -> float:
    """
    Curriculum shaping : reward dense annealée progressivement vers 0.
    Valide en multi-agent car non potential-based (pas de γΦ(s')-Φ(s)).
    Uniquement actif pendant la phase d'exploration initiale.
    """
    annealing = max(0.0, 1.0 - current_step / curriculum_steps)
    if annealing == 0.0:
        return 0.0

    my_id = prev_state.get("my_id", 0)
    prev_planets = prev_state.get("planets", [])
    next_planets = next_state.get("planets", [])

    prev_my = sum(1 for p in prev_planets if p["owner"] == my_id)
    next_my = sum(1 for p in next_planets if p["owner"] == my_id)
    delta = next_my - prev_my  # >0 si capture, <0 si perte

    return float(0.03 * delta * annealing)


def compute_reward(
    prev_state: Dict[str, Any],
    action: Dict[str, Any],
    next_state: Dict[str, Any],
    terminal: bool = False,
) -> float:
    reward = compute_dense_reward(prev_state, next_state, current_step=0)
    if terminal:
        my_id = int(prev_state.get("my_id", next_state.get("my_id", 0)))
        winner = next_state.get("winner")
        if winner == my_id:
            reward += 20.0
        elif winner is not None and int(winner) >= 0:
            reward -= 20.0
    ships = float(action.get("ships", 0.0)) if isinstance(action, dict) else 0.0
    if ships > 0:
        reward += min(0.05, ships / 1000.0)
    return float(reward)
