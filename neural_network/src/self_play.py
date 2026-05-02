from __future__ import annotations

from typing import Any, Dict, List
import math
import numpy as np

from .encoder import encode_game_state
from .policy import choose_action, reconstruct_action
from .reward import compute_reward


def make_synthetic_game(seed: int = 0, four_player: bool = False) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    planets = []
    if four_player:
        base_positions = [(18, 30), (30, 78), (68, 22), (82, 70)]
        owners = [0, 1, 2, 3]
        player_ids = [0, 1, 2, 3]
    else:
        base_positions = [(20, 50), (35, 60), (65, 40), (80, 55)]
        owners = [0, 0, 1, 1]
        player_ids = [0, 1]
    for i, ((x, y), owner) in enumerate(zip(base_positions, owners)):
        planets.append({
            "id": i,
            "owner": owner,
            "x": float(x + rng.uniform(-3, 3)),
            "y": float(y + rng.uniform(-3, 3)),
            "radius": 1.0,
            "production": 3.0,
            "ships": 35.0 + float(rng.integers(0, 10)),
        })
    for i in range(4, 8):
        planets.append({
            "id": i,
            "owner": -1,
            "x": float(rng.uniform(15, 85)),
            "y": float(rng.uniform(15, 85)),
            "radius": 1.0,
            "production": float(rng.integers(1, 4)),
            "ships": float(rng.integers(8, 20)),
        })
    return {
        "my_id": 0,
        "player_ids": player_ids,
        "turn": 0,
        "planets": planets,
        "fleets": [],
        "is_four_player": four_player,
        "winner": None,
        "terminal": False,
    }


def _clone(game: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **game,
        "planets": [dict(p) for p in game.get("planets", [])],
        "fleets": [dict(f) for f in game.get("fleets", [])],
        "player_ids": list(game.get("player_ids", [])),
    }


def _advance_game(game: Dict[str, Any], action: tuple[int, int, int], turn: int) -> Dict[str, Any]:
    next_game = _clone(game)
    next_game["turn"] = turn + 1
    for planet in next_game["planets"]:
        if planet["owner"] != -1:
            planet["ships"] += float(planet.get("production", 0.0))
    if action[0] >= 0 and action[2] > 0:
        src = next(p for p in next_game["planets"] if p["id"] == action[0])
        tgt = next(p for p in next_game["planets"] if p["id"] == action[1])
        moved = min(int(action[2]), max(0, int(src["ships"]) - 1))
        src["ships"] -= moved
        dist = math.hypot(src["x"] - tgt["x"], src["y"] - tgt["y"])
        flight_turns = max(1, int(round(dist / 12.0)))
        next_game["fleets"].append({"owner": game["my_id"], "source_id": src["id"], "target_id": tgt["id"], "ships": float(moved), "eta": flight_turns, "x": src["x"], "y": src["y"]})
    for fleet in next_game["fleets"]:
        fleet["eta"] -= 1
    arrivals = [f for f in next_game["fleets"] if f["eta"] <= 0]
    next_game["fleets"] = [f for f in next_game["fleets"] if f["eta"] > 0]
    for fleet in arrivals:
        tgt = next(p for p in next_game["planets"] if p["id"] == fleet["target_id"])
        if tgt["owner"] in (-1, fleet["owner"]):
            if fleet["ships"] >= tgt["ships"]:
                tgt["owner"] = fleet["owner"]
                tgt["ships"] = fleet["ships"] - tgt["ships"] + 1.0
            else:
                tgt["ships"] -= fleet["ships"]
        else:
            tgt["ships"] -= fleet["ships"]
            if tgt["ships"] <= 0:
                tgt["owner"] = fleet["owner"]
                tgt["ships"] = abs(tgt["ships"]) + 1.0
    players = list(game.get("player_ids", []))
    alive = []
    for pid in players:
        has_planet = any(p["owner"] == pid for p in next_game["planets"])
        has_fleet = any(f["owner"] == pid for f in next_game["fleets"])
        if has_planet or has_fleet:
            alive.append(pid)
    next_game["winner"] = alive[0] if len(alive) == 1 else None
    next_game["terminal"] = len(alive) == 1
    return next_game


def _linear_schedule(start: float, end: float, frac: float) -> float:
    return float(start + (end - start) * min(1.0, max(0.0, frac)))


def play_episode(model, config: Dict[str, Any], seed: int = 0, progress: float = 0.0, four_player: bool = False) -> List[Dict[str, Any]]:
    game = make_synthetic_game(seed, four_player=four_player)
    episode = []
    max_turns = int(config.get("max_turns", 100))
    temperature = _linear_schedule(float(config.get("temperature_start", 1.2)), float(config.get("temperature_end", 0.35)), progress)
    for t in range(max_turns):
        encoded = encode_game_state(game, config)
        candidates = __import__("neural_network.src.policy", fromlist=["build_action_candidates"]).build_action_candidates(game)
        candidate_features = np.stack([c.score_features for c in candidates]).astype(np.float32)
        import torch
        outputs = model(torch.tensor(encoded.features, dtype=torch.float32), torch.tensor(candidate_features, dtype=torch.float32))
        cand, log_prob, entropy = choose_action(outputs, game, temperature=temperature, explore=True, return_entropy=True)
        action = reconstruct_action(cand, game)
        next_game = _advance_game(game, action, t)
        reward = compute_reward(game, {"ships": action[2]}, next_game, terminal=bool(next_game.get("terminal", False)))
        episode.append(
            {
                "state": encoded.features,
                "action": action,
                "reward": reward,
                "next_state": next_game,
                "log_prob": log_prob,
                "entropy": entropy,
            }
        )
        game = next_game
        if next_game.get("terminal"):
            break
    return episode
