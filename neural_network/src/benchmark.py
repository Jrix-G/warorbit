from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

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


def _run_symmetric_match(agent_a, agent_b, config: Dict[str, Any], seed: int, max_turns: int) -> Dict[str, float]:
    first = _run_match(agent_a, agent_b, config, seed, max_turns)
    second = _run_match(agent_b, agent_a, config, seed, max_turns)
    return {
        "first": first,
        "second": second,
        "avg_delta": 0.5 * (first - second),
        "win_first": float(first > 0),
        "win_second": float(second > 0),
    }


def benchmark_model(model, config: Dict[str, Any], games: int = 8, symmetric: bool = False, seed: int = 0) -> Dict[str, float]:
    seeds = [seed + i for i in range(games)]
    if symmetric:
        results = [_run_symmetric_match(model, RandomAgent(), config, s, int(config.get("max_turns", 100))) for s in seeds]
        deltas = [r["avg_delta"] for r in results]
        wins = [r["win_first"] for r in results] + [r["win_second"] for r in results]
        avg_len = float(int(config.get("max_turns", 100)))
        return {
            "games": float(games * 2),
            "avg_reward": float(np.mean(deltas) if deltas else 0.0),
            "winrate": float(np.mean(wins) if wins else 0.0),
            "reward_std": float(np.std(deltas) if deltas else 0.0),
            "avg_episode_length": avg_len,
            "seeds": seeds,
        }
    rewards = [_run_match(model, RandomAgent(), config, seed=s, max_turns=int(config.get("max_turns", 100))) for s in seeds]
    return {
        "games": float(games),
        "avg_reward": float(np.mean(rewards) if rewards else 0.0),
        "winrate": float(np.mean([r > 0 for r in rewards]) if rewards else 0.0),
        "reward_std": float(np.std(rewards) if rewards else 0.0),
        "avg_episode_length": float(int(config.get("max_turns", 100))),
        "seeds": seeds,
    }


def benchmark_matchups(model, config: Dict[str, Any], episodes: int = 8, seed_offset: int = 0) -> Dict[str, float]:
    random_agent = RandomAgent()
    greedy_agent = GreedyNearestWeakestAgent()
    seeds = [seed_offset + i for i in range(episodes)]
    nn_vs_random = [_run_symmetric_match(model, random_agent, config, s, int(config.get("max_turns", 100))) for s in seeds]
    nn_vs_greedy = [_run_symmetric_match(model, greedy_agent, config, s + 1000, int(config.get("max_turns", 100))) for s in seeds]
    random_vs_greedy = [_run_symmetric_match(random_agent, greedy_agent, config, s + 2000, int(config.get("max_turns", 100))) for s in seeds]
    return {
        "winrate_vs_random": float(np.mean([r["win_first"] for r in nn_vs_random] + [r["win_second"] for r in nn_vs_random]) if nn_vs_random else 0.0),
        "winrate_vs_greedy": float(np.mean([r["win_first"] for r in nn_vs_greedy] + [r["win_second"] for r in nn_vs_greedy]) if nn_vs_greedy else 0.0),
        "random_vs_greedy": float(np.mean([r["win_first"] for r in random_vs_greedy] + [r["win_second"] for r in random_vs_greedy]) if random_vs_greedy else 0.0),
        "winrate_by_position": {
            "agent_a_first": float(np.mean([r["win_first"] for r in nn_vs_random]) if nn_vs_random else 0.0),
            "agent_a_second": float(np.mean([r["win_second"] for r in nn_vs_random]) if nn_vs_random else 0.0),
        },
        "adversaries": {
            "random": {"episodes": episodes, "seeds": seeds},
            "greedy": {"episodes": episodes, "seeds": [s + 1000 for s in seeds]},
        },
    }


def compare_checkpoints(model_a, model_b, config: Dict[str, Any], games: int = 8) -> Dict[str, float]:
    deltas = [_run_match(model_a, model_b, config, seed=i, max_turns=int(config.get("max_turns", 100))) for i in range(games)]
    return {
        "games": float(games),
        "avg_reward_delta": float(np.mean(deltas) if deltas else 0.0),
        "winrate_a_vs_b": float(np.mean([d > 0 for d in deltas]) if deltas else 0.0),
        "std_reward_delta": float(np.std(deltas) if deltas else 0.0),
    }
