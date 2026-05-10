"""Measure whether scripted opponent moves are represented by NN action candidates."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_network.src.notebook_4p_training import _agent_for_name, _candidate_move, run_match
from neural_network.src.orbit_wars_adapter import obs_to_game_dict
from neural_network.src.policy import build_action_candidates, reconstruct_action


def _angle_distance(a: float, b: float) -> float:
    return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)


def _best_candidate_distance(game: Dict[str, Any], move: Sequence[Any], config: Dict[str, Any]) -> float | None:
    if not isinstance(move, (list, tuple)) or len(move) < 3:
        return None
    try:
        src_id = int(move[0])
        angle = float(move[1])
        ships = max(1.0, float(move[2]))
    except (TypeError, ValueError):
        return None
    candidates = build_action_candidates(
        game,
        send_ratios=config["send_ratios"],
        min_expand_attack_ships=int(config["min_expand_attack_ships"]),
    )
    best: float | None = None
    for candidate in candidates:
        if candidate.mission == "do_nothing" or int(candidate.source_id) != src_id:
            continue
        action = reconstruct_action(candidate, game)
        candidate_move = _candidate_move(game, action)
        if not candidate_move:
            continue
        cand_angle = float(candidate_move[0][1])
        cand_ships = max(1.0, float(candidate_move[0][2]))
        distance = _angle_distance(cand_angle, angle) + 0.15 * abs(math.log(cand_ships / ships))
        best = distance if best is None else min(best, distance)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="distance")
    parser.add_argument("--opponent", default="random")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=900000)
    parser.add_argument("--n-players", type=int, default=2)
    parser.add_argument("--coverage-threshold", type=float, default=0.35)
    parser.add_argument("--send-ratios", default="0.25,0.35,0.50,0.65,0.80,0.95")
    parser.add_argument("--min-expand-attack-ships", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=100)
    args = parser.parse_args()

    config = {
        "send_ratios": [float(v) for v in args.send_ratios.split(",") if v.strip()],
        "min_expand_attack_ships": int(args.min_expand_attack_ships),
    }
    distances: List[float] = []
    missing = 0
    total = 0

    def tracked_teacher(obs, _config=None):
        nonlocal missing, total
        game = obs_to_game_dict(obs)
        moves = _agent_for_name(args.teacher)(obs, _config)
        for move in moves:
            total += 1
            distance = _best_candidate_distance(game, move, config)
            if distance is None:
                missing += 1
            else:
                distances.append(distance)
        return moves

    for i in range(int(args.episodes)):
        agents = [tracked_teacher] + [_agent_for_name(args.opponent) for _ in range(max(1, int(args.n_players) - 1))]
        run_match(
            agents,
            seed=int(args.seed_start) + i,
            n_players=int(args.n_players),
            max_steps=int(args.max_turns),
            stop_player=0,
        )

    covered = sum(1 for d in distances if d <= float(args.coverage_threshold))
    result = {
        "teacher": args.teacher,
        "opponent": args.opponent,
        "episodes": int(args.episodes),
        "total_actions": int(total),
        "represented_actions": int(covered),
        "missing_source_actions": int(missing),
        "coverage_threshold": float(args.coverage_threshold),
        "coverage": float(covered / max(1, total)),
        "mean_distance": float(np.mean(distances)) if distances else None,
        "median_distance": float(np.median(distances)) if distances else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
