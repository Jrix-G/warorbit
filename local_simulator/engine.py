"""Local two-player Orbit Wars simulator helpers.

This module deliberately lives in local_simulator/ and only imports the
project's existing Kaggle-faithful simulator and current submission bot.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sim  # noqa: E402
import submission  # noqa: E402
from SimGame import SimGame  # noqa: E402


HUMAN_PLAYER = 0
BOT_PLAYER = 1
MAX_HUMAN_MOVES = 12


@dataclass
class TurnResult:
    human_actions: List[List[float]]
    bot_actions: List[List[float]]
    step: int
    winner: int
    scores: List[int]


def new_state(seed: Optional[int] = None, neutral_pairs: int = 8) -> sim.GameState:
    """Create a realistic 2p map and convert it to sim.py's state format."""
    fast = SimGame.random_state(seed=seed, n_players=2, neutral_pairs=neutral_pairs, max_steps=sim.TOTAL_TURNS)
    planets = fast.planets.astype(np.float32).copy()
    initial = fast.initial_planets.astype(np.float32).copy()
    return sim.GameState(
        planets=planets,
        fleets=np.zeros((0, 7), dtype=np.float32),
        init_positions=initial[:, [sim.P_X, sim.P_Y]].astype(np.float32).copy(),
        init_planet_ids=initial[:, sim.P_ID].astype(np.int32).copy(),
        angular_velocity=float(fast.angular_velocity),
        comet_ids=set(),
        comet_groups=[],
        step=0,
        player=HUMAN_PLAYER,
        next_fleet_id=0,
    )


def observation(state: sim.GameState, player: int) -> dict:
    """Build a Kaggle-style observation for a player."""
    init_by_id = {
        int(state.init_planet_ids[i]): state.init_positions[i]
        for i in range(len(state.init_planet_ids))
    }
    initial_planets = []
    for planet in state.planets:
        pid = int(planet[sim.P_ID])
        pos = init_by_id.get(pid, planet[[sim.P_X, sim.P_Y]])
        initial_planets.append(
            [
                pid,
                int(planet[sim.P_OWNER]),
                float(pos[0]),
                float(pos[1]),
                float(planet[sim.P_R]),
                int(planet[sim.P_SHIPS]),
                int(planet[sim.P_PROD]),
            ]
        )

    return {
        "player": int(player),
        "step": int(state.step),
        "planets": state.planets.astype(float).tolist(),
        "fleets": state.fleets.astype(float).tolist(),
        "initial_planets": initial_planets,
        "angular_velocity": float(state.angular_velocity),
        "next_fleet_id": int(state.next_fleet_id),
        "remainingOverageTime": 60.0,
        "comets": [
            {
                "planet_ids": list(group.get("planet_ids", [])),
                "paths": group.get("paths", []),
                "path_index": int(group.get("path_index", -1)),
            }
            for group in state.comet_groups
        ],
        "comet_planet_ids": sorted(int(pid) for pid in state.comet_ids),
    }


def bot_actions(state: sim.GameState) -> List[List[float]]:
    """Ask submission.py's V9 best bot for its current move."""
    try:
        actions = submission.agent(observation(state, BOT_PLAYER), None)
    except TypeError:
        actions = submission.agent(observation(state, BOT_PLAYER))
    if not isinstance(actions, list):
        return []
    return sanitize_actions(state, BOT_PLAYER, actions)


def sanitize_actions(state: sim.GameState, player: int, actions: Sequence[Sequence[float]]) -> List[List[float]]:
    """Keep only legal-ish actions before passing them into the simulator."""
    planets_by_id = {int(p[sim.P_ID]): p for p in state.planets}
    remaining = {
        int(p[sim.P_ID]): int(p[sim.P_SHIPS])
        for p in state.planets
        if int(p[sim.P_OWNER]) == int(player)
    }
    clean: List[List[float]] = []
    for action in actions:
        if len(action) != 3:
            continue
        src_id = int(action[0])
        src = planets_by_id.get(src_id)
        if src is None or int(src[sim.P_OWNER]) != int(player):
            continue
        ships = int(action[2])
        if ships <= 0 or remaining.get(src_id, 0) < ships:
            continue
        angle = float(action[1])
        if not math.isfinite(angle):
            continue
        remaining[src_id] -= ships
        clean.append([src_id, angle, ships])
        if len(clean) >= MAX_HUMAN_MOVES:
            break
    return clean


def make_human_action(state: sim.GameState, src_id: int, target_id: int, fraction: float) -> Optional[List[float]]:
    """Create one attack/transfer action aimed at a target planet."""
    src = planet_by_id(state, src_id)
    target = planet_by_id(state, target_id)
    if src is None or target is None:
        return None
    if int(src[sim.P_OWNER]) != HUMAN_PLAYER or int(src_id) == int(target_id):
        return None
    ships = int(float(src[sim.P_SHIPS]) * max(0.05, min(1.0, fraction)))
    ships = min(ships, int(src[sim.P_SHIPS]))
    if ships <= 0:
        return None
    angle = math.atan2(float(target[sim.P_Y] - src[sim.P_Y]), float(target[sim.P_X] - src[sim.P_X]))
    return [int(src_id), float(angle), int(ships)]


def advance_turn(state: sim.GameState, human_actions: Sequence[Sequence[float]]) -> Tuple[sim.GameState, TurnResult]:
    """Run one simultaneous human+bot turn."""
    human = sanitize_actions(state, HUMAN_PLAYER, human_actions)
    bot = bot_actions(state)
    next_state = sim.step(state, {HUMAN_PLAYER: human, BOT_PLAYER: bot})
    return next_state, TurnResult(
        human_actions=human,
        bot_actions=bot,
        step=int(next_state.step),
        winner=winner(next_state),
        scores=scores(next_state),
    )


def planet_by_id(state: sim.GameState, planet_id: int):
    for planet in state.planets:
        if int(planet[sim.P_ID]) == int(planet_id):
            return planet
    return None


def scores(state: sim.GameState) -> List[int]:
    return [sim.player_total_ships(state, 0), sim.player_total_ships(state, 1)]


def winner(state: sim.GameState) -> int:
    return sim.winner(state, n_players=2) if sim.is_terminal(state) else -2


def is_terminal(state: sim.GameState) -> bool:
    return sim.is_terminal(state)

