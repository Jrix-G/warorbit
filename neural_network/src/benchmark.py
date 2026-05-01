from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from .self_play import play_episode, make_synthetic_game
from .encoder import encode_game_state
from .policy import choose_action


def benchmark_model(model, config: Dict[str, Any], games: int = 8) -> Dict[str, float]:
    rewards = []
    invalid = 0
    for i in range(games):
        ep = play_episode(model, config, seed=i)
        rewards.append(sum(step["reward"] for step in ep))
    return {
        "games": float(games),
        "avg_reward": float(np.mean(rewards) if rewards else 0.0),
        "winrate": float(np.mean([r > 0 for r in rewards]) if rewards else 0.0),
        "invalid_action_rate": float(invalid / max(1, games)),
    }


def compare_checkpoints(model_a, model_b, config: Dict[str, Any], games: int = 8) -> Dict[str, Dict[str, float]]:
    return {
        "model_a": benchmark_model(model_a, config, games),
        "model_b": benchmark_model(model_b, config, games),
    }

