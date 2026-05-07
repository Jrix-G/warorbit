#!/usr/bin/env python3
"""Measure V11 behavior against top1 replay targets."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SimGame import SimGame
import bot_v11
from opponents import ZOO


P_ID, P_OWNER, P_X, P_Y, P_R, P_SHIPS, P_PROD = range(7)


DEFAULT_OPPONENTS = [
    "notebook_orbitbotnext",
    "notebook_distance_prioritized",
    "notebook_physics_accurate",
    "notebook_pascalledesma_orbitwork_v14",
    "notebook_romantamrazov_orbit_star_wars_lb_max_1224",
]


@dataclass
class Profile:
    games: int = 0
    wins: int = 0
    actions: int = 0
    active_turns: int = 0
    turns: int = 0
    friendly: int = 0
    neutral: int = 0
    hostile: int = 0
    unknown: int = 0
    ships: List[int] = field(default_factory=list)
    planets_t60: List[int] = field(default_factory=list)
    planets_t100: List[int] = field(default_factory=list)
    garrison_p10: List[float] = field(default_factory=list)

    def update_turn(self, obs: dict, moves: list) -> None:
        self.turns += 1
        if moves:
            self.active_turns += 1
        planets = obs.get("planets", []) or []
        player = int(obs.get("player", 0) or 0)
        for move in moves or []:
            if len(move) != 3:
                continue
            self.actions += 1
            self.ships.append(int(move[2]))
            target_owner = _infer_target_owner(planets, move)
            if target_owner == player:
                self.friendly += 1
            elif target_owner == -1:
                self.neutral += 1
            elif target_owner is None:
                self.unknown += 1
            else:
                self.hostile += 1
        step = int(obs.get("step", 0) or 0)
        my_planets = [p for p in planets if int(p[P_OWNER]) == player]
        if step == 60:
            self.planets_t60.append(len(my_planets))
        elif step == 100:
            self.planets_t100.append(len(my_planets))
        if my_planets:
            self.garrison_p10.append(float(np.percentile([float(p[P_SHIPS]) for p in my_planets], 10)))

    def merge(self, other: "Profile") -> None:
        for name in ("games", "wins", "actions", "active_turns", "turns", "friendly", "neutral", "hostile", "unknown"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.ships.extend(other.ships)
        self.planets_t60.extend(other.planets_t60)
        self.planets_t100.extend(other.planets_t100)
        self.garrison_p10.extend(other.garrison_p10)


def _infer_target_owner(planets: Sequence[Sequence[float]], move: Sequence[float]) -> int | None:
    src_id, angle, _ships = int(move[0]), float(move[1]), int(move[2])
    src = None
    for p in planets:
        if int(p[P_ID]) == src_id:
            src = p
            break
    if src is None:
        return None
    sx, sy = float(src[P_X]), float(src[P_Y])
    dx, dy = math.cos(angle), math.sin(angle)
    best = None
    for p in planets:
        if int(p[P_ID]) == src_id:
            continue
        px, py = float(p[P_X]), float(p[P_Y])
        vx, vy = px - sx, py - sy
        proj = vx * dx + vy * dy
        if proj <= 0:
            continue
        perp = abs(vx * dy - vy * dx)
        threshold = float(p[P_R]) + 5.5
        if perp > threshold:
            continue
        score = (perp, proj)
        if best is None or score < best[0]:
            best = (score, int(p[P_OWNER]))
    return None if best is None else best[1]


def _run_profile_game(agents: Sequence[Callable], our_idx: int, seed: int, n_players: int, max_steps: int) -> Profile:
    game = SimGame.random_game(seed=seed, n_players=n_players, max_steps=max_steps)
    prof = Profile(games=1)
    while not game.is_terminal():
        actions: Dict[int, list] = {}
        for player, agent in enumerate(agents):
            obs = game.observation(player)
            try:
                move = agent(obs, None)
            except TypeError:
                move = agent(obs)
            move = move if isinstance(move, list) else []
            if player == our_idx:
                prof.update_turn(obs, move)
            actions[player] = move
        game.step(actions)
    if game.winner() == our_idx:
        prof.wins += 1
    return prof


def _lineup(opponents: Sequence[str], n_players: int, seed_i: int) -> tuple[list[Callable], int]:
    our_idx = seed_i % n_players
    opps = [ZOO[opponents[(seed_i + j) % len(opponents)]] for j in range(max(1, n_players - 1))]
    agents = []
    opp_iter = iter(opps)
    for i in range(n_players):
        agents.append(bot_v11.agent if i == our_idx else next(opp_iter))
    return agents, our_idx


def _fmt_percent(num: int, den: int) -> str:
    return f"{(num / den if den else 0.0):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile V11 top1-like behavior.")
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--mode", choices=["2p", "4p"], default="4p")
    parser.add_argument("--seed-offset", type=int, default=21000)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--opponents", nargs="*", default=DEFAULT_OPPONENTS)
    args = parser.parse_args()

    missing = [name for name in args.opponents if name not in ZOO]
    if missing:
        raise SystemExit(f"Opponents not found in ZOO: {missing}")

    n_players = 4 if args.mode == "4p" else 2
    total = Profile()
    for i in range(args.games):
        agents, our_idx = _lineup(args.opponents, n_players, args.seed_offset + i)
        total.merge(_run_profile_game(agents, our_idx, args.seed_offset + i, n_players, args.max_steps))

    ships = np.asarray(total.ships, dtype=np.float32) if total.ships else np.asarray([0], dtype=np.float32)
    print(f"V11 profile | mode={args.mode} games={total.games} wins={total.wins} WR={_fmt_percent(total.wins, total.games)}")
    print(f"actions={total.actions} active_turns={total.active_turns} turns={total.turns}")
    print(f"actions_per_active_turn={total.actions / max(1, total.active_turns):.3f}")
    print(f"transfer_ratio={_fmt_percent(total.friendly, total.actions)} neutral_ratio={_fmt_percent(total.neutral, total.actions)} hostile_ratio={_fmt_percent(total.hostile, total.actions)} unknown_ratio={_fmt_percent(total.unknown, total.actions)}")
    print(f"ships_p50={np.percentile(ships, 50):.1f} ships_p75={np.percentile(ships, 75):.1f} ships_p90={np.percentile(ships, 90):.1f} ships_p99={np.percentile(ships, 99):.1f}")
    if total.planets_t60:
        print(f"planets_t60_mean={np.mean(total.planets_t60):.2f}")
    if total.planets_t100:
        print(f"planets_t100_mean={np.mean(total.planets_t100):.2f}")
    if total.garrison_p10:
        print(f"garrison_p10_median={np.median(total.garrison_p10):.1f}")


if __name__ == "__main__":
    main()
