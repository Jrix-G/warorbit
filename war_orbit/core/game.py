"""Adapters from Kaggle-style observations to the local fast simulator.

V9 keeps the game engine in one place. The existing `SimGame.py` runner is the
fast local engine; this module only normalizes observations and exposes compact
state scoring utilities used by search and training.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

import bot_v7
from SimGame import F_OWNER, F_SHIPS, FastState, P_OWNER, P_PROD, P_SHIPS, SimGame


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def build_world(obs):
    """Build the stable V7 WorldModel from an observation."""

    return bot_v7._build_world(obs)


def _array(rows: Iterable, width: int) -> np.ndarray:
    rows = list(rows or [])
    if not rows:
        return np.zeros((0, width), dtype=np.float32)
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f"Expected array with width {width}, got {arr.shape}")
    return arr


def infer_player_count(obs) -> int:
    """Infer active player count from planets/fleets, preserving 2p minimum."""

    explicit = _get(obs, "n_players", None) or _get(obs, "players", None)
    if explicit:
        try:
            return max(2, int(explicit))
        except Exception:
            pass

    owners = set()
    for p in _get(obs, "planets", []) or []:
        owner = int(p[P_OWNER])
        if owner >= 0:
            owners.add(owner)
    for f in _get(obs, "fleets", []) or []:
        owner = int(f[F_OWNER])
        if owner >= 0:
            owners.add(owner)
    player = int(_get(obs, "player", 0) or 0)
    owners.add(player)
    return max(2, max(owners) + 1 if owners else 2)


def observation_to_fast_state(obs, max_steps: Optional[int] = None) -> FastState:
    """Convert an observation into a SimGame FastState copy."""

    planets = _array(_get(obs, "planets", []), 7)
    fleets = _array(_get(obs, "fleets", []), 7)
    initial = _array(_get(obs, "initial_planets", []), 7)
    if len(initial) == 0:
        initial = planets.copy()
    step = int(_get(obs, "step", 0) or 0)
    next_fleet_id = int(_get(obs, "next_fleet_id", 0) or 0)
    if next_fleet_id <= 0 and len(fleets):
        next_fleet_id = int(np.max(fleets[:, 0])) + 1
    horizon = int(max_steps) if max_steps is not None else max(500, step + 32)
    return FastState(
        planets=planets.copy(),
        fleets=fleets.copy(),
        initial_planets=initial.copy(),
        angular_velocity=float(_get(obs, "angular_velocity", 0.0) or 0.0),
        step=step,
        next_fleet_id=next_fleet_id,
        max_steps=horizon,
    )


def make_game_from_observation(obs, max_steps: Optional[int] = None) -> SimGame:
    state = observation_to_fast_state(obs, max_steps=max_steps)
    return SimGame(state, n_players=infer_player_count(obs))


def score_state(state: FastState, n_players: int, player: int) -> Dict[str, float]:
    """Return a compact strategic score snapshot for one player."""

    planets = state.planets
    fleets = state.fleets
    my_planets = planets[planets[:, P_OWNER] == player] if len(planets) else np.zeros((0, 7), dtype=np.float32)
    enemy_planets = planets[(planets[:, P_OWNER] >= 0) & (planets[:, P_OWNER] != player)] if len(planets) else np.zeros((0, 7), dtype=np.float32)
    neutral_planets = planets[planets[:, P_OWNER] < 0] if len(planets) else np.zeros((0, 7), dtype=np.float32)

    owner_scores = []
    for owner in range(n_players):
        p_mask = planets[:, P_OWNER] == owner if len(planets) else np.zeros(0, dtype=bool)
        f_mask = fleets[:, F_OWNER] == owner if len(fleets) else np.zeros(0, dtype=bool)
        ships = float(planets[p_mask, P_SHIPS].sum()) if len(planets) and np.any(p_mask) else 0.0
        ships += float(fleets[f_mask, F_SHIPS].sum()) if len(fleets) and np.any(f_mask) else 0.0
        prod = float(planets[p_mask, P_PROD].sum()) if len(planets) and np.any(p_mask) else 0.0
        owned = float(np.sum(p_mask)) if len(planets) else 0.0
        owner_scores.append(ships + 18.0 * prod + 7.0 * owned)

    my_score = owner_scores[player] if player < len(owner_scores) else 0.0
    top_other = max((s for i, s in enumerate(owner_scores) if i != player), default=0.0)
    my_fleets = fleets[fleets[:, F_OWNER] == player] if len(fleets) else np.zeros((0, 7), dtype=np.float32)

    return {
        "score": my_score,
        "margin": my_score - top_other,
        "ships": float(my_planets[:, P_SHIPS].sum()) + float(my_fleets[:, F_SHIPS].sum()) if len(my_planets) or len(my_fleets) else 0.0,
        "production": float(my_planets[:, P_PROD].sum()) if len(my_planets) else 0.0,
        "planets": float(len(my_planets)),
        "enemy_planets": float(len(enemy_planets)),
        "neutral_planets": float(len(neutral_planets)),
        "top_other": top_other,
    }
