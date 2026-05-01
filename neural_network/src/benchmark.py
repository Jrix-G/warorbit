from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from .baselines import GreedyNearestWeakestAgent, RandomAgent
from .encoder import encode_game_state
from .policy import build_action_candidates, choose_action, reconstruct_action
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


def _player_score(game: Dict[str, Any], player_id: int) -> float:
    planets = game.get("planets", [])
    fleets = game.get("fleets", [])
    owned_planets = [p for p in planets if p["owner"] == player_id]
    owned_fleets = [f for f in fleets if f["owner"] == player_id]
    return float(
        100.0 * len(owned_planets)
        + 2.5 * sum(p.get("production", 0.0) for p in owned_planets)
        + sum(p.get("ships", 0.0) for p in owned_planets)
        + sum(f.get("ships", 0.0) for f in owned_fleets)
    )


def _final_verdict(game: Dict[str, Any]) -> int:
    players = list(game.get("player_ids", [0, 1]))
    alive = [pid for pid in players if any(p["owner"] == pid for p in game.get("planets", [])) or any(f["owner"] == pid for f in game.get("fleets", []))]
    if len(alive) == 1:
        return int(alive[0])
    if len(alive) == 0:
        return -1
    score0 = _player_score(game, players[0])
    score1 = _player_score(game, players[1])
    if abs(score0 - score1) < 1e-6:
        return -1
    return int(players[0] if score0 > score1 else players[1])


def _run_match(agent_a, agent_b, config: Dict[str, Any], seed: int, max_turns: int) -> Dict[str, float]:
    game = make_synthetic_game(seed)
    for turn in range(max_turns):
        game_a = dict(game)
        game_a["my_id"] = 0
        game_b = dict(game)
        game_b["my_id"] = 1
        action_a = _action_for(agent_a, game_a, config)
        action_b = _action_for(agent_b, game_b, config)
        next_game = _advance_game(game, action_a, turn)
        next_game = _advance_game(next_game, action_b, turn)
        game = next_game
        if game.get("terminal"):
            break
    winner = _final_verdict(game)
    a_player = 0
    b_player = 1
    return {
        "winner": float(winner),
        "a_win": float(winner == a_player),
        "b_win": float(winner == b_player),
        "draw": float(winner == -1),
        "episode_length": float(max_turns if not game.get("terminal") else game.get("turn", max_turns)),
        "avg_reward": float(_player_score(game, 0) - _player_score(game, 1)),
    }


def _run_symmetric_match(agent_a, agent_b, config: Dict[str, Any], seed: int, max_turns: int) -> Dict[str, float]:
    first = _run_match(agent_a, agent_b, config, seed, max_turns)
    second = _run_match(agent_b, agent_a, config, seed, max_turns)
    return {
        "first": first,
        "second": second,
        "win_first": float(first["a_win"]),
        "win_second": float(second["a_win"]),
        "draw_first": float(first["draw"]),
        "draw_second": float(second["draw"]),
        "avg_reward": 0.5 * (first["avg_reward"] - second["avg_reward"]),
    }


def benchmark_model(model, config: Dict[str, Any], games: int = 8, symmetric: bool = False, seed: int = 0) -> Dict[str, float]:
    seeds = [seed + i for i in range(games)]
    if symmetric:
        results = [_run_symmetric_match(model, RandomAgent(), config, s, int(config.get("max_turns", 100))) for s in seeds]
        wins = [r["win_first"] for r in results] + [r["win_second"] for r in results]
        draws = [r["draw_first"] for r in results] + [r["draw_second"] for r in results]
        avg_reward = [r["avg_reward"] for r in results]
        avg_len = float(np.mean([r["first"]["episode_length"] for r in results] + [r["second"]["episode_length"] for r in results]))
        return {
            "games": float(games * 2),
            "avg_reward": float(np.mean(avg_reward) if avg_reward else 0.0),
            "winrate": float(np.mean(wins) if wins else 0.0),
            "draw_rate": float(np.mean(draws) if draws else 0.0),
            "reward_std": float(np.std(avg_reward) if avg_reward else 0.0),
            "avg_episode_length": avg_len,
            "seeds": seeds,
        }
    rewards = [_run_match(model, RandomAgent(), config, seed=s, max_turns=int(config.get("max_turns", 100))) for s in seeds]
    return {
        "games": float(games),
        "avg_reward": float(np.mean([r["avg_reward"] for r in rewards]) if rewards else 0.0),
        "winrate": float(np.mean([r["a_win"] for r in rewards]) if rewards else 0.0),
        "draw_rate": float(np.mean([r["draw"] for r in rewards]) if rewards else 0.0),
        "reward_std": float(np.std([r["avg_reward"] for r in rewards]) if rewards else 0.0),
        "avg_episode_length": float(np.mean([r["episode_length"] for r in rewards]) if rewards else int(config.get("max_turns", 100))),
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
        "draw_rate_vs_random": float(np.mean([r["draw_first"] for r in nn_vs_random] + [r["draw_second"] for r in nn_vs_random]) if nn_vs_random else 0.0),
        "draw_rate_vs_greedy": float(np.mean([r["draw_first"] for r in nn_vs_greedy] + [r["draw_second"] for r in nn_vs_greedy]) if nn_vs_greedy else 0.0),
        "draw_rate_random_vs_greedy": float(np.mean([r["draw_first"] for r in random_vs_greedy] + [r["draw_second"] for r in random_vs_greedy]) if random_vs_greedy else 0.0),
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
