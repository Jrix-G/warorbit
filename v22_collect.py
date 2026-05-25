"""Collect V22 combo-oracle samples from local simulated games."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

import v14_core
import v15_fast_sim as fsim
import v15_search
import v21_agent
import v22_agent
import v22_dataset
import v22_oracle
from local_simulator.official_fast import OfficialFastGame


@dataclass
class CollectConfig:
    n_players: int = 4
    games: int = 2
    steps: int = 100
    stride: int = 10
    seed_offset: int = 52_000_000
    horizon: int = 8
    det_horizon: int = 6
    top_k: int = 10
    beam_width: int = 24
    max_combo: int = 4
    max_samples: int = 128
    policy: str = "v15"
    policy_budget: float = 0.02
    min_advantage: float = 0.0


class AgentConfig:
    def __init__(self, n_players: int, episode_steps: int, seed: int) -> None:
        self.nPlayers = int(n_players)
        self.episodeSteps = int(episode_steps)
        self.actTimeout = 1.0
        self.shipSpeed = 6.0
        self.cometSpeed = 4.0
        self.remainingOverageTime = 60.0
        self.seed = int(seed)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def collect(cfg: CollectConfig) -> list[dict[str, Any]]:
    if cfg.n_players not in (2, 4):
        raise ValueError("n_players must be 2 or 4")
    if cfg.games <= 0 or cfg.steps <= 0 or cfg.stride <= 0:
        raise ValueError("games, steps and stride must be positive")
    samples: list[dict[str, Any]] = []
    for game_idx in range(int(cfg.games)):
        seed = int(cfg.seed_offset) + game_idx
        random.seed(seed)
        np.random.seed(seed)
        game = OfficialFastGame(int(cfg.n_players), seed=seed, episode_steps=int(cfg.steps), use_c_accel=False)
        agent_cfg = AgentConfig(cfg.n_players, cfg.steps, seed)
        episode_id = f"v22-{seed}"
        for step in range(int(cfg.steps)):
            if getattr(game, "done", False):
                break
            if len(samples) >= int(cfg.max_samples):
                return samples
            actions: list[list] = []
            for player in range(int(cfg.n_players)):
                obs = v14_core.obs_as_dict(game.observation(player))
                if step % int(cfg.stride) == 0 and len(samples) < int(cfg.max_samples):
                    try:
                        fs = fsim.from_obs(obs, n_players=int(cfg.n_players), episode_steps=int(cfg.steps), ship_speed=6.0)
                        samples.append(
                            v22_oracle.sample_from_state(
                                fs,
                                player,
                                episode_id=episode_id,
                                source=f"v22_oracle_{cfg.policy}",
                                horizon=int(cfg.horizon),
                                det_horizon=int(cfg.det_horizon),
                                top_k=int(cfg.top_k),
                                beam_width=int(cfg.beam_width),
                                max_combo=int(cfg.max_combo),
                                min_advantage=float(cfg.min_advantage),
                            )
                        )
                    except Exception:
                        pass
                actions.append(_policy_action(cfg.policy, obs, agent_cfg, cfg.policy_budget))
            if not getattr(game, "done", False):
                game.step(actions)
    return samples


def _policy_action(policy: str, obs: dict[str, Any], config: AgentConfig, budget: float) -> list:
    if policy == "pass":
        return []
    try:
        if policy == "v15":
            move = v15_search.search(obs, config, time_budget=float(budget), horizon=10)
        elif policy == "v21":
            move = v21_agent.agent(obs, config, time_budget=float(budget), horizon=10)
        elif policy == "v22":
            move = v22_agent.agent(obs, config, time_budget=float(budget), horizon=10)
        else:
            raise ValueError(f"unsupported policy {policy!r}")
        return move if isinstance(move, list) else []
    except Exception:
        return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect V22 combo oracle samples")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-players", type=int, default=4)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=52_000_000)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--det-horizon", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--beam-width", type=int, default=24)
    parser.add_argument("--max-combo", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--policy", choices=["pass", "v15", "v21", "v22"], default="v15")
    parser.add_argument("--policy-budget", type=float, default=0.02)
    parser.add_argument("--min-advantage", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CollectConfig(
        n_players=args.n_players,
        games=args.games,
        steps=args.steps,
        stride=args.stride,
        seed_offset=args.seed_offset,
        horizon=args.horizon,
        det_horizon=args.det_horizon,
        top_k=args.top_k,
        beam_width=args.beam_width,
        max_combo=args.max_combo,
        max_samples=args.max_samples,
        policy=args.policy,
        policy_budget=args.policy_budget,
        min_advantage=args.min_advantage,
    )
    samples = collect(cfg)
    written = v22_dataset.write_jsonl(args.out, samples)
    print(json.dumps({"samples": len(samples), "written": written, "out": args.out}, sort_keys=True))


if __name__ == "__main__":
    main()
