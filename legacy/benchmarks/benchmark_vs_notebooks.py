#!/usr/bin/env python3
"""Benchmark bot_v6 against notebook opponents with profile presets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

from kaggle_environments import make

import bot_v6
from opponents import ZOO


DEFAULT_OPPONENTS = [
    "notebook_orbitbotnext",
    "notebook_distance_prioritized",
    "notebook_physics_accurate",
    "notebook_tactical_heuristic",
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


def _profile_games(profile: str) -> int:
    if profile == "smoke":
        return 2
    if profile == "rocket":
        return 12
    return 30


def _play_game(our_first: bool, opponent_name: str) -> int:
    """Return +1 if we win, 0 draw, -1 loss."""
    opponent = ZOO[opponent_name]
    env = make("orbit_wars", configuration={"episodeSteps": 500}, debug=False)
    if our_first:
        env.run([bot_v6.agent, opponent])
        our_idx, opp_idx = 0, 1
    else:
        env.run([opponent, bot_v6.agent])
        our_idx, opp_idx = 1, 0

    our_reward = env.steps[-1][our_idx].reward
    opp_reward = env.steps[-1][opp_idx].reward

    if our_reward is not None and opp_reward is not None:
        if our_reward > opp_reward:
            return 1
        if our_reward < opp_reward:
            return -1
        return 0

    # Fallback when reward is unavailable: compare final ships from observation.
    final_obs = env.steps[-1][our_idx].observation
    planets = final_obs.get("planets", []) if isinstance(final_obs, dict) else getattr(final_obs, "planets", [])
    fleets = final_obs.get("fleets", []) if isinstance(final_obs, dict) else getattr(final_obs, "fleets", [])

    def owner(row):
        if isinstance(row, dict):
            return int(row.get("owner", -999))
        return int(row[1]) if len(row) > 1 else -999

    def ships(row, idx):
        if isinstance(row, dict):
            return float(row.get("ships", 0.0))
        return float(row[idx]) if len(row) > idx else 0.0

    our_ships = 0.0
    opp_ships = 0.0
    for p in planets:
        o = owner(p)
        s = ships(p, 5)
        if o == our_idx:
            our_ships += s
        elif o == opp_idx:
            opp_ships += s
    for f in fleets:
        o = owner(f)
        s = ships(f, 6)
        if o == our_idx:
            our_ships += s
        elif o == opp_idx:
            opp_ships += s

    if our_ships > opp_ships:
        return 1
    if our_ships < opp_ships:
        return -1
    return 0


def run_benchmark(opponents: List[str], games_per_opp: int) -> Tuple[Dict[str, MatchStats], MatchStats]:
    per_opp: Dict[str, MatchStats] = {}
    total = MatchStats()

    for opp in opponents:
        stats = MatchStats()
        for i in range(games_per_opp):
            outcome = _play_game(our_first=(i % 2 == 0), opponent_name=opp)
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
    parser = argparse.ArgumentParser(description="Benchmark bot_v6 vs notebook opponents.")
    parser.add_argument("--profile", choices=("smoke", "rocket", "full"), default="rocket")
    parser.add_argument("--games-per-opp", type=int, default=None)
    parser.add_argument("--opponents", nargs="*", default=DEFAULT_OPPONENTS)
    args = parser.parse_args()

    missing = [name for name in args.opponents if name not in ZOO]
    if missing:
        raise SystemExit(f"Opponents not found in ZOO: {missing}")

    games_per_opp = args.games_per_opp if args.games_per_opp is not None else _profile_games(args.profile)
    print(f"Profile={args.profile} | games_per_opp={games_per_opp} | opponents={len(args.opponents)}")

    per_opp, total = run_benchmark(args.opponents, games_per_opp)

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
