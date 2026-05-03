"""Short-horizon simulation evaluator for V9 candidate plans."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from SimGame import P_ID, P_OWNER, P_PROD, P_SHIPS, P_X, P_Y, SimGame

from ...core.game import infer_player_count, make_game_from_observation, score_state
from ...features.plan_features import PlanCandidate


@dataclass
class SimulationEstimate:
    mean_delta: float = 0.0
    worst_delta: float = 0.0
    margin_delta: float = 0.0
    production_delta: float = 0.0
    planet_delta: float = 0.0
    finish_bonus: float = 0.0
    uncertainty: float = 0.0
    rollouts: int = 0


def _planet_rows(game: SimGame, owner: Optional[int] = None):
    planets = game.state.planets
    if len(planets) == 0:
        return planets
    if owner is None:
        return planets
    return planets[planets[:, P_OWNER] == owner]


def _angle(src, target) -> float:
    return math.atan2(float(target[P_Y] - src[P_Y]), float(target[P_X] - src[P_X]))


def _nearest_targets(planets, src, player: int, prefer: str):
    others = planets[planets[:, P_OWNER] != player]
    if len(others) == 0:
        return []
    if prefer == "expansion":
        pool = others[others[:, P_OWNER] < 0]
        if len(pool) == 0:
            pool = others
        order = sorted(pool, key=lambda t: (-float(t[P_PROD]), float(t[P_SHIPS]), np.hypot(t[P_X] - src[P_X], t[P_Y] - src[P_Y])))
    elif prefer == "finisher":
        enemies = others[others[:, P_OWNER] >= 0]
        pool = enemies if len(enemies) else others
        order = sorted(pool, key=lambda t: (float(t[P_SHIPS]), -float(t[P_PROD]), np.hypot(t[P_X] - src[P_X], t[P_Y] - src[P_Y])))
    elif prefer == "denial":
        enemies = others[others[:, P_OWNER] >= 0]
        pool = enemies if len(enemies) else others
        order = sorted(pool, key=lambda t: (-float(t[P_PROD]), float(t[P_SHIPS])))
    else:
        order = sorted(others, key=lambda t: np.hypot(t[P_X] - src[P_X], t[P_Y] - src[P_Y]))
    return order


def opponent_policy(game: SimGame, player: int, style: str, rng: random.Random) -> List[List]:
    """Cheap state-policy opponent used inside rollouts."""

    planets = game.state.planets
    if len(planets) == 0:
        return []
    my = _planet_rows(game, player)
    if len(my) == 0:
        return []
    moves: List[List] = []
    if style == "passive":
        return moves

    for src in sorted(my, key=lambda p: -float(p[P_SHIPS])):
        ships = int(src[P_SHIPS])
        if ships < 9:
            continue
        if style == "random":
            if rng.random() < 0.45:
                moves.append([int(src[P_ID]), rng.random() * 2.0 * math.pi, max(1, ships // 3)])
            continue
        prefer = "balanced"
        if style in ("expansion", "greedy"):
            prefer = "expansion"
        elif style == "finisher":
            prefer = "finisher"
        elif style == "denial":
            prefer = "denial"
        elif style == "probe":
            prefer = "finisher"
        targets = _nearest_targets(planets, src, player, prefer)
        if not targets:
            continue
        target = targets[0]
        if style == "conservative":
            send = max(0, (ships - 18) // 2)
        elif style == "turtle":
            send = max(0, (ships - 28) // 3)
        elif style == "finisher":
            send = int(ships * 0.72)
        elif style == "denial":
            send = int(ships * 0.58)
        elif style == "probe":
            send = min(12, max(0, ships // 5))
        else:
            send = int(ships * 0.50)
        if send >= 4:
            moves.append([int(src[P_ID]), _angle(src, target), send])
    return moves


class V9Evaluator:
    """Evaluate candidate plans with short stochastic rollouts."""

    def __init__(self, *, depth: int = 3, rollouts: int = 2, opponent_samples: int = 2, seed: int = 9009):
        self.depth = int(depth)
        self.rollouts = int(rollouts)
        self.opponent_samples = int(opponent_samples)
        self.seed = int(seed)

    def evaluate(self, obs, candidates: Iterable[PlanCandidate]) -> Dict[str, SimulationEstimate]:
        candidates = list(candidates)
        if self.depth <= 0 or self.rollouts <= 0 or not candidates:
            return {c.name: SimulationEstimate(rollouts=0) for c in candidates}
        player = int(obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0))
        n_players = infer_player_count(obs)
        out: Dict[str, SimulationEstimate] = {}
        for idx, candidate in enumerate(candidates):
            out[candidate.name] = self._evaluate_one(obs, candidate, player, n_players, idx)
        return out

    def _evaluate_one(self, obs, candidate: PlanCandidate, player: int, n_players: int, idx: int) -> SimulationEstimate:
        deltas = []
        margins = []
        prods = []
        planets = []
        finishes = []
        styles = ["greedy", "expansion", "denial", "conservative", "finisher", "random", "probe", "turtle"]
        for r in range(self.rollouts):
            rng = random.Random(self.seed + 1009 * idx + 9173 * r + int(obs.get("step", 0) if isinstance(obs, dict) else 0))
            game = make_game_from_observation(obs, max_steps=max(500, int(obs.get("step", 0) if isinstance(obs, dict) else 0) + self.depth + 2))
            before = score_state(game.state, n_players, player)
            first_actions = {player: [list(m) for m in candidate.moves]}
            for opp in range(n_players):
                if opp == player:
                    continue
                style = styles[(opp + r + idx) % min(len(styles), max(1, self.opponent_samples + 2))]
                first_actions[opp] = opponent_policy(game, opp, style, rng)
            game.step(first_actions)
            for t in range(max(0, self.depth - 1)):
                if game.is_terminal():
                    break
                actions = {}
                for slot in range(n_players):
                    if slot == player:
                        own_style = "finisher" if candidate.plan_type == "endgame_finisher" else "denial" if candidate.plan_type == "resource_denial" else "greedy"
                        actions[slot] = opponent_policy(game, slot, own_style, rng)
                    else:
                        style = styles[(slot + r + t) % len(styles)]
                        actions[slot] = opponent_policy(game, slot, style, rng)
                game.step(actions)
            after = score_state(game.state, n_players, player)
            deltas.append((after["score"] - before["score"]) / 100.0)
            margins.append((after["margin"] - before["margin"]) / 100.0)
            prods.append(after["production"] - before["production"])
            planets.append(after["planets"] - before["planets"])
            alive = game.alive_players()
            finishes.append(1.0 if alive == [player] else 0.0)

        arr = np.asarray(deltas, dtype=np.float32)
        return SimulationEstimate(
            mean_delta=float(np.mean(arr)) if len(arr) else 0.0,
            worst_delta=float(np.min(arr)) if len(arr) else 0.0,
            margin_delta=float(np.mean(margins)) if margins else 0.0,
            production_delta=float(np.mean(prods)) if prods else 0.0,
            planet_delta=float(np.mean(planets)) if planets else 0.0,
            finish_bonus=float(np.mean(finishes)) if finishes else 0.0,
            uncertainty=float(np.std(arr)) if len(arr) > 1 else 0.0,
            rollouts=len(deltas),
        )
