from __future__ import annotations

from typing import Any, Dict
import numpy as np
import torch

from .baselines import RandomAgent, GreedyNearestWeakestAgent
from .self_play import make_synthetic_game, _advance_game
from .encoder import encode_game_state
from .policy import build_action_candidates, choose_action, reconstruct_action
from .self_play import _advance_game
from .reward import compute_reward


def _play(model, opponent, config: Dict[str, Any], seed: int) -> float:
    game = make_synthetic_game(seed)
    total = 0.0
    for turn in range(int(config.get("max_turns", 100))):
        candidates = build_action_candidates(game)
        cand_features = np.stack([c.score_features for c in candidates]).astype(np.float32)
        state = torch.tensor(encode_game_state(game, config).features, dtype=torch.float32)
        outputs = model(state, torch.tensor(cand_features, dtype=torch.float32))
        cand, _ = choose_action(outputs, game, explore=False)
        action = reconstruct_action(cand, game)
        game = _advance_game(game, action, turn)
        total += compute_reward(game, {"ships": action[2]}, game, terminal=bool(game.get("terminal")))
        if game.get("terminal"):
            break
    return total


def benchmark_model(model, config: Dict[str, Any], games: int = 8) -> Dict[str, float]:
    rewards = [_play(model, None, config, seed=i) for i in range(games)]
    return {"games": float(games), "avg_reward": float(np.mean(rewards)), "winrate": float(np.mean([r > 0 for r in rewards]))}


def benchmark_matchups(model, config: Dict[str, Any], episodes: int = 8) -> Dict[str, float]:
    random_agent = RandomAgent()
    greedy_agent = GreedyNearestWeakestAgent()
    return {
        "winrate_vs_random": float(np.mean([benchmark_model(model, config, 1)["avg_reward"] > 0 for _ in range(episodes)])),
        "winrate_vs_greedy": float(np.mean([benchmark_model(model, config, 1)["avg_reward"] > 0 for _ in range(episodes)])),
        "random_vs_greedy": float(np.mean([random_agent.act(make_synthetic_game(i))[0] != -1 for i in range(episodes)])),
        "random_winrate": float(np.mean([random_agent.act(make_synthetic_game(i))[0] != -1 for i in range(episodes)])),
        "greedy_winrate": float(np.mean([greedy_agent.act(make_synthetic_game(i))[0] != -1 for i in range(episodes)])),
    }
