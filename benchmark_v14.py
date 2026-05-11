#!/usr/bin/env python3
"""Unified c_engine benchmark for V7/V12/V13/V14."""

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
import bot_v14
from opponents import ZOO
from local_simulator.official_fast import OfficialFastGame
import v14_core


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


def _call_agent(fn: Callable, obs, config) -> list:
    obs = v14_core.obs_as_dict(obs)
    try:
        move = fn(obs, config)
    except TypeError:
        move = fn(obs)
    return move if isinstance(move, list) else []


def _agent(
    name: str,
    v13_weights: str,
    v14_weights: str,
    v14_4p_runtime: str,
    v14_4p_agent: str,
    v14_4p_profile: str,
) -> Callable:
    if name == "v7":
        return bot_v7.agent
    if name == "v12":
        return bot_v12.agent
    if name == "v13":
        os.environ["V13_WEIGHTS"] = v13_weights
        bot_v13._MLP_CACHE.clear()
        return bot_v13.agent
    if name == "v14":
        os.environ["V14_WEIGHTS"] = v14_weights
        os.environ["V14_4P_RUNTIME"] = v14_4p_runtime
        os.environ["V14_4P_AGENT"] = v14_4p_agent
        os.environ["V14_4P_PROFILE"] = v14_4p_profile
        bot_v14._CACHE.clear()
        bot_v14._FOUR_PLAYER_AGENT = None
        bot_v14._FOUR_PLAYER_MODULES = None
        bot_v14._FOUR_PLAYER_AGENT_LOADED = False
        return bot_v14.agent
    if name not in ZOO:
        raise KeyError(name)
    return ZOO[name]


def _lineup(
    bot_name: str,
    opponents: Sequence[str],
    n_players: int,
    seed_i: int,
    v13_weights: str,
    v14_weights: str,
    v14_4p_runtime: str,
    v14_4p_agent: str,
    v14_4p_profile: str,
) -> tuple[list[Callable], int]:
    our = _agent(bot_name, v13_weights, v14_weights, v14_4p_runtime, v14_4p_agent, v14_4p_profile)
    if n_players <= 2:
        opp = _agent(
            opponents[seed_i % len(opponents)],
            v13_weights,
            v14_weights,
            v14_4p_runtime,
            v14_4p_agent,
            v14_4p_profile,
        )
        return ([our, opp], 0) if seed_i % 2 == 0 else ([opp, our], 1)
    chosen = [
        _agent(
            opponents[(seed_i + j) % len(opponents)],
            v13_weights,
            v14_weights,
            v14_4p_runtime,
            v14_4p_agent,
            v14_4p_profile,
        )
        for j in range(3)
    ]
    our_idx = seed_i % 4
    agents = []
    opp_iter = iter(chosen)
    for i in range(4):
        agents.append(our if i == our_idx else next(opp_iter))
    return agents, our_idx


def _play_task(task: tuple[str, tuple[str, ...], int, int, int, str, str, str, str, str]) -> tuple[int, float]:
    (
        bot_name,
        opponents,
        n_players,
        seed,
        max_steps,
        v13_weights,
        v14_weights,
        v14_4p_runtime,
        v14_4p_agent,
        v14_4p_profile,
    ) = task
    start = time.time()
    agents, our_idx = _lineup(
        bot_name,
        opponents,
        n_players,
        seed,
        v13_weights,
        v14_weights,
        v14_4p_runtime,
        v14_4p_agent,
        v14_4p_profile,
    )
    game = OfficialFastGame(
        n_players=n_players,
        seed=seed,
        episode_steps=max_steps,
        use_c_accel=True,
    )
    while not game.done:
        actions = [
            _call_agent(agent_fn, game.observation(player), game.configuration)
            for player, agent_fn in enumerate(agents)
        ]
        game.step(actions)
    scores = game.scores()
    best_other = max(s for i, s in enumerate(scores) if i != our_idx)
    if scores[our_idx] > best_other and scores[our_idx] > 0:
        return 1, time.time() - start
    if scores[our_idx] == best_other:
        return 0, time.time() - start
    return -1, time.time() - start


def run_suite(bot_name: str, opponents: Sequence[str], games: int, n_players: int, seed_offset: int,
              workers: int, max_steps: int, v13_weights: str, v14_weights: str,
              v14_4p_runtime: str, v14_4p_agent: str, v14_4p_profile: str) -> MatchStats:
    tasks = [
        (
            bot_name,
            tuple(opponents),
            n_players,
            seed_offset + i,
            max_steps,
            v13_weights,
            v14_weights,
            v14_4p_runtime,
            v14_4p_agent,
            v14_4p_profile,
        )
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
    parser.add_argument("--v13-weights", default="evaluations/scorer_v13_2h.best.npz")
    parser.add_argument("--v14-weights", default="evaluations/scorer_v14.npz")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-offset", type=int, default=51000)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--bots", nargs="*", default=["v7", "v12", "v13", "v14"])
    parser.add_argument("--modes", nargs="*", default=["4p", "2p"])
    parser.add_argument("--opponents", nargs="*", default=DEFAULT_OPPONENTS)
    parser.add_argument(
        "--v14-4p-runtime",
        choices=["ml", "notebook"],
        default="ml",
        help="Use 'ml' to benchmark V14 weights in 4p; 'notebook' tests the notebook fallback.",
    )
    parser.add_argument("--v14-4p-agent", default="distance")
    parser.add_argument("--v14-4p-profile", default="closer")
    args = parser.parse_args()

    missing = [name for name in args.opponents if name not in ZOO]
    if missing:
        raise SystemExit(f"Opponents not found in ZOO: {missing}")
    print(
        f"V14 benchmark | games={args.games} modes={','.join(args.modes)} "
        f"bots={','.join(args.bots)} opponents={len(args.opponents)} "
        f"workers={args.workers} max_steps={args.max_steps} "
        f"v14_4p_runtime={args.v14_4p_runtime}"
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
                v13_weights=args.v13_weights,
                v14_weights=args.v14_weights,
                v14_4p_runtime=args.v14_4p_runtime,
                v14_4p_agent=args.v14_4p_agent,
                v14_4p_profile=args.v14_4p_profile,
            )
            print(
                f"- {bot_name:4s} W/L/D={stats.wins}/{stats.losses}/{stats.draws} "
                f"WR={stats.win_rate:.3f} seconds={stats.seconds:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
