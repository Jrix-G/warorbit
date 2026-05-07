#!/usr/bin/env python3
"""Benchmark V11 against V7 and notebook opponents."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from SimGame import run_match
import bot_v7
import bot_v11
from opponents import ZOO


DEFAULT_OPPONENTS = [
    "notebook_orbitbotnext",
    "notebook_distance_prioritized",
    "notebook_physics_accurate",
    "notebook_pascalledesma_orbitwork_v14",
    "notebook_romantamrazov_orbit_star_wars_lb_max_1224",
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


def _agent(name: str) -> Callable:
    if name == "v7":
        return bot_v7.agent
    if name == "v11":
        return bot_v11.agent
    if name not in ZOO:
        raise KeyError(name)
    return ZOO[name]


def _lineup(bot_name: str, opponent_names: Sequence[str], n_players: int, seed_i: int) -> tuple[list[Callable], int]:
    our = _agent(bot_name)
    if n_players <= 2:
        opp = _agent(opponent_names[seed_i % len(opponent_names)])
        if seed_i % 2 == 0:
            return [our, opp], 0
        return [opp, our], 1
    chosen = [_agent(opponent_names[(seed_i + j) % len(opponent_names)]) for j in range(3)]
    our_idx = seed_i % 4
    agents = []
    opp_iter = iter(chosen)
    for i in range(4):
        agents.append(our if i == our_idx else next(opp_iter))
    return agents, our_idx


def _play_task(task: tuple[str, tuple[str, ...], int, int, int]) -> tuple[int, float]:
    bot_name, opponents, n_players, seed, max_steps = task
    agents, our_idx = _lineup(bot_name, opponents, n_players, seed)
    result = run_match(agents, seed=seed, n_players=n_players, max_steps=max_steps)
    winner = int(result.get("winner", -1))
    if winner == our_idx:
        outcome = 1
    elif winner < 0:
        outcome = 0
    else:
        outcome = -1
    return outcome, float(result.get("seconds", 0.0))


def run_suite(bot_name: str, opponents: Sequence[str], games: int, n_players: int, seed_offset: int,
              workers: int, max_steps: int) -> MatchStats:
    stats = MatchStats()
    tasks = [
        (bot_name, tuple(opponents), n_players, seed_offset + i, max_steps)
        for i in range(games)
    ]
    if workers <= 1:
        results = [_play_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_play_task, tasks))
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
    parser = argparse.ArgumentParser(description="Benchmark V11 vs V7 on local SimGame.")
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed-offset", type=int, default=11000)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--opponents", nargs="*", default=DEFAULT_OPPONENTS)
    parser.add_argument("--modes", nargs="*", default=["4p", "2p"])
    args = parser.parse_args()

    missing = [name for name in args.opponents if name not in ZOO]
    if missing:
        raise SystemExit(f"Opponents not found in ZOO: {missing}")

    print(
        f"V11 benchmark | games={args.games} modes={','.join(args.modes)} "
        f"opponents={len(args.opponents)} workers={args.workers} max_steps={args.max_steps}"
    )
    for mode in args.modes:
        n_players = 4 if mode == "4p" else 2
        print(f"\nMode {mode}")
        for bot_name in ("v7", "v11"):
            stats = run_suite(
                bot_name,
                args.opponents,
                games=args.games,
                n_players=n_players,
                seed_offset=args.seed_offset + (0 if bot_name == "v7" else 100000) + n_players * 1000,
                workers=max(1, args.workers),
                max_steps=args.max_steps,
            )
            print(
                f"- {bot_name:4s} W/L/D={stats.wins}/{stats.losses}/{stats.draws} "
                f"WR={stats.win_rate:.3f} seconds={stats.seconds:.1f}"
            )


if __name__ == "__main__":
    main()
