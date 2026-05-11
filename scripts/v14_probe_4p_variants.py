#!/usr/bin/env python3
"""Probe V14 4p runtime variants seed-by-seed.

This is intentionally separate from benchmark_v14.py: it prints enough context
to understand where a variant wins or collapses, not just aggregate WR.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bot_v14
import v14_core
from benchmark_v14 import DEFAULT_OPPONENTS, _agent, _call_agent, _lineup
from local_simulator.official_fast import OfficialFastGame


@dataclass(frozen=True)
class Variant:
    agent: str
    profile: str

    @property
    def label(self) -> str:
        return f"{self.agent}:{self.profile}"


def _home_signature(obs: dict) -> str:
    me = int(obs.get("player", 0) or 0)
    planets = list(obs.get("planets", []) or [])
    mine = [p for p in planets if int(p[1]) == me]
    if not mine:
        return "none"
    home = max(mine, key=lambda p: float(p[6]))
    x = float(home[2])
    y = float(home[3])
    if x >= 50.0 and y >= 50.0:
        quad = "NE"
    elif x < 50.0 <= y:
        quad = "NW"
    elif x < 50.0 and y < 50.0:
        quad = "SW"
    else:
        quad = "SE"
    return f"{quad}@{x:.1f},{y:.1f}"


def _set_variant(variant: Variant) -> Callable:
    os.environ["V14_4P_AGENT"] = variant.agent
    os.environ["V14_4P_PROFILE"] = variant.profile
    bot_v14._FOUR_PLAYER_AGENT = None
    bot_v14._FOUR_PLAYER_MODULES = None
    bot_v14._FOUR_PLAYER_AGENT_LOADED = False
    return _agent("v14", "evaluations/scorer_v13_2h.best.npz", os.environ.get("V14_WEIGHTS", "evaluations/scorer_v14.npz"))


def play_seed(seed: int, variant: Variant, opponents: tuple[str, ...], max_steps: int) -> tuple[bool, int, int, list[int], str, float]:
    start = time.time()
    game = OfficialFastGame(n_players=4, seed=seed, episode_steps=max_steps, use_c_accel=True)
    our_idx = seed % 4
    home_sig = _home_signature(v14_core.obs_as_dict(game.observation(our_idx)))

    our = _set_variant(variant)
    chosen = [_agent(opponents[(seed + j) % len(opponents)], "", "") for j in range(3)]
    agents = []
    opp_iter = iter(chosen)
    for i in range(4):
        agents.append(our if i == our_idx else next(opp_iter))

    while not game.done:
        actions = [
            _call_agent(agent_fn, game.observation(player), game.configuration)
            for player, agent_fn in enumerate(agents)
        ]
        game.step(actions)
    scores = list(game.scores())
    our_score = scores[our_idx]
    best_other = max(s for i, s in enumerate(scores) if i != our_idx)
    rank = 1 + sum(1 for s in scores if s > our_score)
    win = our_score > best_other and our_score > 0
    return win, rank, our_idx, scores, home_sig, time.time() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-offset", type=int, default=55000)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--opponents", nargs="*", default=DEFAULT_OPPONENTS)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[
            "distance:eco",
            "orbitbotnext:eco",
            "distance:closer",
            "orbitbotnext:closer",
            "distance:base",
            "physics:eco",
        ],
    )
    args = parser.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    variants = [Variant(*item.split(":", 1)) for item in args.variants]
    opponents = tuple(args.opponents)

    totals: dict[str, int] = {variant.label: 0 for variant in variants}
    print(f"probe_4p games={args.games} seeds={args.seed_offset}..{args.seed_offset + args.games - 1} max_steps={args.max_steps}")
    print("seed\tidx\thome\tvariant\twin\trank\tscores\tsec")
    for seed in range(args.seed_offset, args.seed_offset + args.games):
        best_line = None
        for variant in variants:
            win, rank, idx, scores, home_sig, seconds = play_seed(seed, variant, opponents, args.max_steps)
            totals[variant.label] += int(win)
            line = (seed, idx, home_sig, variant.label, win, rank, scores, seconds)
            if best_line is None or (line[4], -line[5], scores[idx]) > (best_line[4], -best_line[5], best_line[6][best_line[1]]):
                best_line = line
            print(f"{seed}\t{idx}\t{home_sig}\t{variant.label}\t{int(win)}\t{rank}\t{scores}\t{seconds:.1f}", flush=True)
        if best_line is not None:
            print(
                f"best\t{best_line[0]}\t{best_line[1]}\t{best_line[2]}\t{best_line[3]}"
                f"\t{int(best_line[4])}\t{best_line[5]}\t{best_line[6]}",
                flush=True,
            )

    print("totals")
    for label, wins in sorted(totals.items(), key=lambda item: (-item[1], item[0])):
        print(f"{label}\t{wins}/{args.games}\t{wins / max(1, args.games):.3f}")


if __name__ == "__main__":
    main()
