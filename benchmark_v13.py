#!/usr/bin/env python3
"""Benchmark V13 checkpoint against V7/V12 on c_engine.CGame."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, Sequence

import bot_v7
import bot_v12
import bot_v13
from c_engine import CGame
from opponents import ZOO


DEFAULT_OPPONENTS = [
    "notebook_orbitbotnext",
    "notebook_distance_prioritized",
    "notebook_physics_accurate",
    "notebook_pascalledesma_orbitwork_v14",
]


@dataclass
class MatchStats:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    seconds: float = 0.0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0


def _obs_as_dict(obs):
    if isinstance(obs, dict):
        return obs
    data = vars(obs).copy()
    data.setdefault("remainingOverageTime", 60.0)
    return data


def _call_agent(fn: Callable, obs, config) -> list:
    obs = _obs_as_dict(obs)
    try:
        move = fn(obs, config)
    except TypeError:
        move = fn(obs)
    return move if isinstance(move, list) else []


def _agent(name: str, weights: str) -> Callable:
    if name == "v7":
        return bot_v7.agent
    if name == "v12":
        return bot_v12.agent
    if name == "v13":
        os.environ["V13_WEIGHTS"] = weights
        bot_v13._MLP_CACHE.clear()
        return bot_v13.agent
    if name not in ZOO:
        raise KeyError(name)
    return ZOO[name]


def _lineup(
    bot_name: str,
    opponent_names: Sequence[str],
    n_players: int,
    seed_i: int,
    weights: str,
) -> tuple[list[Callable], int]:
    our = _agent(bot_name, weights)
    if n_players <= 2:
        opp = _agent(opponent_names[seed_i % len(opponent_names)], weights)
        if seed_i % 2 == 0:
            return [our, opp], 0
        return [opp, our], 1

    chosen = [_agent(opponent_names[(seed_i + j) % len(opponent_names)], weights) for j in range(3)]
    our_idx = seed_i % 4
    agents = []
    opp_iter = iter(chosen)
    for i in range(4):
        agents.append(our if i == our_idx else next(opp_iter))
    return agents, our_idx


def _play_task(task: tuple[str, tuple[str, ...], int, int, int, str]) -> tuple[int, float]:
    bot_name, opponents, n_players, seed, max_steps, weights = task
    start = time.time()
    agents, our_idx = _lineup(bot_name, opponents, n_players, seed, weights)
    game = CGame(n_players=n_players, seed=seed, episode_steps=max_steps)

    while not game.done:
        actions = [
            _call_agent(agent_fn, game.observation(player), game.configuration)
            for player, agent_fn in enumerate(agents)
        ]
        game.step(actions)

    scores = game.scores()
    best_other = max(s for i, s in enumerate(scores) if i != our_idx)
    if scores[our_idx] > best_other and scores[our_idx] > 0:
        outcome = 1
    elif scores[our_idx] == best_other:
        outcome = 0
    else:
        outcome = -1
    return outcome, time.time() - start


def run_suite(
    bot_name: str,
    opponents: Sequence[str],
    games: int,
    n_players: int,
    seed_offset: int,
    workers: int,
    max_steps: int,
    weights: str,
) -> MatchStats:
    tasks = [
        (bot_name, tuple(opponents), n_players, seed_offset + i, max_steps, weights)
        for i in range(games)
    ]
    if workers <= 1:
        results = [_play_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_play_task, tasks))

    stats = MatchStats()
    for outcome, seconds in results:
        stats.seconds += seconds
        if outcome > 0:
            stats.wins += 1
        elif outcome < 0:
            stats.losses += 1
        else:
            stats.draws += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="evaluations/scorer_v13_2h.best.npz")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-offset", type=int, default=41000)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--bots", nargs="*", default=["v7", "v12", "v13"])
    parser.add_argument("--modes", nargs="*", default=["4p", "2p"])
    parser.add_argument("--opponents", nargs="*", default=DEFAULT_OPPONENTS)
    args = parser.parse_args()

    missing = [name for name in args.opponents if name not in ZOO]
    if missing:
        raise SystemExit(f"Opponents not found in ZOO: {missing}")

    print(
        f"V13 benchmark | weights={args.weights} games={args.games} "
        f"modes={','.join(args.modes)} bots={','.join(args.bots)} "
        f"opponents={len(args.opponents)} workers={args.workers} max_steps={args.max_steps}"
    )
    for mode in args.modes:
        n_players = 4 if mode == "4p" else 2
        print(f"\nMode {mode}")
        for bot_i, bot_name in enumerate(args.bots):
            stats = run_suite(
                bot_name,
                args.opponents,
                games=args.games,
                n_players=n_players,
                seed_offset=args.seed_offset + bot_i * 100000 + n_players * 1000,
                workers=max(1, args.workers),
                max_steps=args.max_steps,
                weights=args.weights,
            )
            print(
                f"- {bot_name:4s} W/L/D={stats.wins}/{stats.losses}/{stats.draws} "
                f"WR={stats.win_rate:.3f} seconds={stats.seconds:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
