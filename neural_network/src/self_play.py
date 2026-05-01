from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np

from .encoder import encode_game_state
from .policy import choose_action, reconstruct_action
from .reward import compute_reward


def make_synthetic_game(seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    planets = []
    for i in range(6):
        owner = -1 if i < 3 else int(i % 2)
        planets.append({
            "id": i,
            "owner": owner,
            "x": float(rng.uniform(0, 100)),
            "y": float(rng.uniform(0, 100)),
            "radius": 1.0,
            "production": float(rng.integers(1, 5)),
            "ships": float(rng.integers(10, 40)),
        })
    fleets = []
    return {"my_id": 0, "player_ids": [0, 1], "turn": 0, "planets": planets, "fleets": fleets, "is_four_player": False}


def play_episode(model, config: Dict[str, Any], seed: int = 0) -> List[Dict[str, Any]]:
    game = make_synthetic_game(seed)
    episode = []
    for t in range(3):
        game["turn"] = t
        encoded = encode_game_state(game, config)
        outputs = model.forward(encoded.features)
        cand = choose_action(outputs, game)
        action = reconstruct_action(cand, game)
        next_game = dict(game)
        next_game["planets"] = [dict(p) for p in game["planets"]]
        if action[0] >= 0:
            src = next(p for p in next_game["planets"] if p["id"] == action[0])
            tgt = next(p for p in next_game["planets"] if p["id"] == action[1])
            src["ships"] = max(0.0, src["ships"] - action[2])
            tgt["ships"] += 0.5 * action[2]
        reward = compute_reward(game, {"ships": action[2]}, next_game, terminal=(t == 2))
        episode.append({"state": encoded.features, "action": action, "reward": reward, "next_state": next_game})
        game = next_game
    return episode

