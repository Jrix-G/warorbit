#!/usr/bin/env python3
"""Benchmark bot_v7 AoW tuning against notebook opponents."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

from SimGame import run_match
import bot_v7
from opponents import ZOO


DEFAULT_OPPONENTS = [
    "notebook_orbitbotnext",
    "notebook_distance_prioritized",
    "notebook_physics_accurate",
]


@dataclass
class MatchStats:
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0


def _play_game(our_first: bool, opponent_name: str, seed: int) -> int:
    """Return +1 if we win, 0 draw, -1 loss."""
    opponent = ZOO[opponent_name]
    if our_first:
        result = run_match([bot_v7.agent, opponent], seed=seed)
        our_idx, opp_idx = 0, 1
    else:
        result = run_match([opponent, bot_v7.agent], seed=seed)
        our_idx, opp_idx = 1, 0

    winner = int(result.get("winner", -1))
    if winner == our_idx:
        return 1
    if winner == opp_idx:
        return -1

    # Fallback for edge cases where the engine returns no winner.
    return 0


def _play_task(task: Tuple[str, bool, int]) -> int:
    opp_name, our_first, seed = task
    return _play_game(our_first=our_first, opponent_name=opp_name, seed=seed)


def run_benchmark(
    opponents: List[str],
    games_per_opp: int,
    seed_offset: int,
    workers: int,
) -> Tuple[Dict[str, MatchStats], MatchStats]:
    per_opp: Dict[str, MatchStats] = {}
    total = MatchStats()

    max_workers = max(1, int(workers))
    for opp in opponents:
        stats = MatchStats()
        print(f"Benchmarking vs {opp}...", flush=True)
        tasks = [(opp, i % 2 == 0, seed_offset + i) for i in range(games_per_opp)]
        if max_workers == 1:
            outcomes = [_play_task(task) for task in tasks]
        else:
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as pool:
                    outcomes = list(pool.map(_play_task, tasks))
            except Exception as exc:
                print(f"  (parallel fallback: {exc.__class__.__name__}; running serial)", flush=True)
                outcomes = [_play_task(task) for task in tasks]
        for outcome in outcomes:
            if outcome > 0:
                stats.wins += 1
                total.wins += 1
            elif outcome < 0:
                stats.losses += 1
                total.losses += 1
            else:
                stats.draws += 1
                total.draws += 1
        per_opp[opp] = stats
    return per_opp, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark bot_v7 AoW against notebook opponents.")
    parser.add_argument("--games-per-opp", type=int, default=20)
    parser.add_argument("--opponents", nargs="*", default=DEFAULT_OPPONENTS)
    parser.add_argument("--seed-offset", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    missing = [name for name in args.opponents if name not in ZOO]
    if missing:
        raise SystemExit(f"Opponents not found in ZOO: {missing}")

    print(
        f"V7 AoW benchmark | games_per_opp={args.games_per_opp} | "
        f"opponents={len(args.opponents)} | seed_offset={args.seed_offset} | workers={args.workers}"
    )

    per_opp, total = run_benchmark(args.opponents, args.games_per_opp, args.seed_offset, args.workers)

    print("\nPer-opponent results")
    for opp, stats in per_opp.items():
        print(
            f"- {opp:33s}  W/L/D={stats.wins:2d}/{stats.losses:2d}/{stats.draws:2d}  "
            f"WR={stats.win_rate * 100:5.1f}%"
        )

    print("\nGlobal")
    print(
        f"- Games={total.games}  W/L/D={total.wins}/{total.losses}/{total.draws}  "
        f"WR={total.win_rate * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
