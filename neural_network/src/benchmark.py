from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from .baselines import GreedyNearestWeakestAgent, RandomAgent
from .encoder import encode_game_state
from .policy import build_action_candidates, choose_action, reconstruct_action
from .reward import compute_reward
from .self_play import _advance_game, make_synthetic_game


def _model_action(model, game: Dict[str, Any], config: Dict[str, Any]) -> tuple[int, int, int]:
    encoded = encode_game_state(game, config)
    candidates = build_action_candidates(game)
    candidate_features = np.stack([c.score_features for c in candidates]).astype(np.float32)
    outputs = model(
        torch.tensor(encoded.features, dtype=torch.float32),
        torch.tensor(candidate_features, dtype=torch.float32),
    )
    cand, _ = choose_action(outputs, game, explore=False)
    return reconstruct_action(cand, game)


def _action_for(agent, game: Dict[str, Any], config: Dict[str, Any]) -> tuple[int, int, int]:
    if hasattr(agent, "act"):
        return agent.act(game)
    return _model_action(agent, game, config)


def _run_match(agent_a, agent_b, config: Dict[str, Any], seed: int, max_turns: int) -> float:
    game = make_synthetic_game(seed)
    total_a = 0.0
    total_b = 0.0
    for turn in range(max_turns):
        game_a = dict(game)
        game_a["my_id"] = 0
        game_b = dict(game)
        game_b["my_id"] = 1
        action_a = _action_for(agent_a, game_a, config)
        action_b = _action_for(agent_b, game_b, config)
        next_game = _advance_game(game, action_a, turn)
        next_game = _advance_game(next_game, action_b, turn)
        total_a += compute_reward(game_a, {"ships": action_a[2]}, next_game, terminal=bool(next_game.get("terminal")))
        total_b += compute_reward(game_b, {"ships": action_b[2]}, next_game, terminal=bool(next_game.get("terminal")))
        game = next_game
        if game.get("terminal"):
            break
    return total_a - total_b


def benchmark_model(model, config: Dict[str, Any], games: int = 8) -> Dict[str, float]:
    rewards = [_run_match(model, RandomAgent(), config, seed=i, max_turns=int(config.get("max_turns", 100))) for i in range(games)]
    return {
        "games": float(games),
        "avg_reward": float(np.mean(rewards) if rewards else 0.0),
        "winrate": float(np.mean([r > 0 for r in rewards]) if rewards else 0.0),
    }


def benchmark_matchups(model, config: Dict[str, Any], episodes: int = 8) -> Dict[str, float]:
    random_agent = RandomAgent()
    greedy_agent = GreedyNearestWeakestAgent()
    nn_vs_random = [_run_match(model, random_agent, config, i, int(config.get("max_turns", 100))) for i in range(episodes)]
    nn_vs_greedy = [_run_match(model, greedy_agent, config, i + 1000, int(config.get("max_turns", 100))) for i in range(episodes)]
    random_vs_greedy = [_run_match(random_agent, greedy_agent, config, i + 2000, int(config.get("max_turns", 100))) for i in range(episodes)]
    return {
        "winrate_vs_random": float(np.mean([r > 0 for r in nn_vs_random]) if nn_vs_random else 0.0),
        "winrate_vs_greedy": float(np.mean([r > 0 for r in nn_vs_greedy]) if nn_vs_greedy else 0.0),
        "random_vs_greedy": float(np.mean([r > 0 for r in random_vs_greedy]) if random_vs_greedy else 0.0),
    }
