"""Fast local Orbit Wars simulator.

This module is intentionally separate from sim.py:
- sim.py is the fidelity target against Kaggle.
- SimGame.py is the fast training/benchmark runner we can grow step by step.

The first version focuses on the hot path: local map generation, fleet launch,
production, orbit updates, sun/planet collisions, combat, observations, and
full bot-vs-bot match execution without kaggle_environments.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0
TOTAL_TURNS = 500
START_POSITIONS_2P = [(18.0, 50.0), (82.0, 50.0)]
START_POSITIONS_4P = [(18.0, 50.0), (82.0, 50.0), (50.0, 18.0), (50.0, 82.0)]

P_ID, P_OWNER, P_X, P_Y, P_R, P_SHIPS, P_PROD = range(7)
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = range(7)


def fleet_speed(ships: float) -> float:
    if ships <= 1:
        return 1.0
    return min(MAX_SPEED, 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5)


def fleet_speeds(ships: np.ndarray) -> np.ndarray:
    speeds = np.ones_like(ships, dtype=np.float32)
    mask = ships > 1
    if np.any(mask):
        speeds[mask] = 1.0 + (MAX_SPEED - 1.0) * (np.log(ships[mask]) / math.log(1000.0)) ** 1.5
        np.minimum(speeds, MAX_SPEED, out=speeds)
    return speeds


def segment_point_dist_sq(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    lsq = dx * dx + dy * dy
    safe = np.where(lsq < 1e-12, 1.0, lsq)
    t = ((px - x1) * dx + (py - y1) * dy) / safe
    t = np.clip(t, 0.0, 1.0)
    qx = x1 + t * dx
    qy = y1 + t * dy
    return (px - qx) * (px - qx) + (py - qy) * (py - qy)


@dataclass
class FastState:
    planets: np.ndarray
    fleets: np.ndarray
    initial_planets: np.ndarray
    angular_velocity: float
    step: int = 0
    next_fleet_id: int = 0
    max_steps: int = TOTAL_TURNS

    def copy(self) -> "FastState":
        return FastState(
            planets=self.planets.copy(),
            fleets=self.fleets.copy(),
            initial_planets=self.initial_planets.copy(),
            angular_velocity=float(self.angular_velocity),
            step=int(self.step),
            next_fleet_id=int(self.next_fleet_id),
            max_steps=int(self.max_steps),
        )


class SimGame:
    """Fast in-process game runner.

    Public API:
    - `SimGame.random_state(seed=...)`
    - `game.step(actions_by_player)`
    - `game.run([agent0, agent1, ...])`
    """

    def __init__(self, state: FastState, n_players: int = 2):
        self.state = state
        self.n_players = int(n_players)

    @classmethod
    def random_state(
        cls,
        seed: Optional[int] = None,
        n_players: int = 2,
        neutral_pairs: int = 8,
        max_steps: int = TOTAL_TURNS,
    ) -> "FastState":
        rng = random.Random(seed)
        planets = []
        next_id = 0

        if n_players <= 2:
            starts = START_POSITIONS_2P
        elif n_players <= 4:
            starts = START_POSITIONS_4P
        else:
            starts = [
                (
                    CENTER + 28.0 * math.cos(2.0 * math.pi * i / max(1, n_players)),
                    CENTER + 28.0 * math.sin(2.0 * math.pi * i / max(1, n_players)),
                )
                for i in range(n_players)
            ]
        if n_players > len(starts):
            raise ValueError(f"Unsupported player count: {n_players}")

        for player in range(n_players):
            x, y = starts[player]
            planets.append([next_id, player, x, y, 3.0, 100.0, 3.0])
            next_id += 1

        def add_neutral(x: float, y: float) -> bool:
            nonlocal next_id
            if math.hypot(x - CENTER, y - CENTER) < SUN_RADIUS + 7.0:
                return False
            if all(math.hypot(x - p[P_X], y - p[P_Y]) > p[P_R] + 5.0 for p in planets):
                radius = rng.choice((1.6, 2.0, 2.4, 2.8))
                ships = rng.randint(5, 32)
                prod = rng.randint(1, 4)
                planets.append([next_id, -1, x, y, radius, ships, prod])
                next_id += 1
                return True
            return False

        for _ in range(neutral_pairs):
            if n_players <= 2:
                points = [
                    (
                        rng.uniform(18.0, 48.0),
                        rng.uniform(16.0, 84.0),
                    ),
                ]
                points = points + [(BOARD_SIZE - x, y) for x, y in points]
            else:
                x = rng.uniform(18.0, 48.0)
                y = rng.uniform(16.0, 34.0)
                points = [
                    (x, y),
                    (BOARD_SIZE - x, y),
                    (x, BOARD_SIZE - y),
                    (BOARD_SIZE - x, BOARD_SIZE - y),
                ]
            for x, y in points:
                for _tries in range(200):
                    jitter_x = x + rng.uniform(-1.5, 1.5)
                    jitter_y = y + rng.uniform(-1.5, 1.5)
                    if add_neutral(jitter_x, jitter_y):
                        break

        arr = np.array(planets, dtype=np.float32)
        return FastState(
            planets=arr,
            fleets=np.zeros((0, 7), dtype=np.float32),
            initial_planets=arr.copy(),
            angular_velocity=rng.uniform(-0.045, 0.045),
            max_steps=int(max_steps),
        )

    @classmethod
    def random_game(
        cls,
        seed: Optional[int] = None,
        n_players: int = 2,
        neutral_pairs: int = 8,
        max_steps: int = TOTAL_TURNS,
    ) -> "SimGame":
        return cls(cls.random_state(seed, n_players, neutral_pairs, max_steps), n_players=n_players)

    def observation(self, player: int) -> dict:
        s = self.state
        return {
            "player": int(player),
            "step": int(s.step),
            "planets": s.planets.astype(float).tolist(),
            "fleets": s.fleets.astype(float).tolist(),
            "initial_planets": s.initial_planets.astype(float).tolist(),
            "angular_velocity": float(s.angular_velocity),
            "next_fleet_id": int(s.next_fleet_id),
            "remainingOverageTime": 60.0,
            "comets": [],
            "comet_planet_ids": [],
        }

    def update_planet_positions(self) -> None:
        s = self.state
        if len(s.planets) == 0:
            return
        if (
            len(s.planets) == len(s.initial_planets)
            and np.array_equal(s.planets[:, P_ID].astype(np.int32), s.initial_planets[:, P_ID].astype(np.int32))
        ):
            dx = s.initial_planets[:, P_X] - CENTER
            dy = s.initial_planets[:, P_Y] - CENTER
            r = np.hypot(dx, dy)
            orbit = r + s.planets[:, P_R] < ROTATION_RADIUS_LIMIT
            if np.any(orbit):
                angle = np.arctan2(dy[orbit], dx[orbit]) + s.angular_velocity * s.step
                s.planets[orbit, P_X] = CENTER + r[orbit] * np.cos(angle)
                s.planets[orbit, P_Y] = CENTER + r[orbit] * np.sin(angle)
            if np.any(~orbit):
                s.planets[~orbit, P_X] = s.initial_planets[~orbit, P_X]
                s.planets[~orbit, P_Y] = s.initial_planets[~orbit, P_Y]
            return

        initial_by_id = {int(p[P_ID]): p for p in s.initial_planets}
        for i in range(len(s.planets)):
            pid = int(s.planets[i, P_ID])
            init = initial_by_id.get(pid)
            if init is None:
                continue
            dx = float(init[P_X] - CENTER)
            dy = float(init[P_Y] - CENTER)
            r = math.hypot(dx, dy)
            if r + float(s.planets[i, P_R]) >= ROTATION_RADIUS_LIMIT:
                s.planets[i, P_X] = init[P_X]
                s.planets[i, P_Y] = init[P_Y]
                continue
            angle = math.atan2(dy, dx) + s.angular_velocity * s.step
            s.planets[i, P_X] = CENTER + r * math.cos(angle)
            s.planets[i, P_Y] = CENTER + r * math.sin(angle)

    def step(self, actions_by_player: Dict[int, Sequence[Sequence[float]]]) -> FastState:
        s = self.state
        planets = s.planets
        fleets = s.fleets

        if len(planets) and actions_by_player:
            pid_to_idx = {int(planets[i, P_ID]): i for i in range(len(planets))}
            new_fleets = []
            for player, moves in actions_by_player.items():
                for move in moves or ():
                    if len(move) != 3:
                        continue
                    src_id, angle, ships_req = int(move[0]), float(move[1]), int(move[2])
                    row = pid_to_idx.get(src_id)
                    if row is None or ships_req <= 0:
                        continue
                    if int(planets[row, P_OWNER]) != int(player):
                        continue
                    if int(planets[row, P_SHIPS]) < ships_req:
                        continue
                    planets[row, P_SHIPS] -= ships_req
                    pr = float(planets[row, P_R])
                    new_fleets.append([
                        s.next_fleet_id,
                        int(player),
                        float(planets[row, P_X]) + math.cos(angle) * (pr + 0.1),
                        float(planets[row, P_Y]) + math.sin(angle) * (pr + 0.1),
                        angle,
                        src_id,
                        ships_req,
                    ])
                    s.next_fleet_id += 1
            if new_fleets:
                add = np.array(new_fleets, dtype=np.float32)
                fleets = add if len(fleets) == 0 else np.vstack((fleets, add))
                s.fleets = fleets

        if len(planets):
            owned = planets[:, P_OWNER] >= 0
            planets[owned, P_SHIPS] += planets[owned, P_PROD]

        fleet_snapshot = fleets.copy() if len(fleets) else np.zeros((0, 7), dtype=np.float32)
        dead = np.zeros(len(fleets), dtype=bool)
        arrivals: Dict[int, List[int]] = {}

        if len(fleets):
            old_x = fleets[:, F_X].copy()
            old_y = fleets[:, F_Y].copy()
            speeds = fleet_speeds(fleets[:, F_SHIPS])
            fleets[:, F_X] = old_x + np.cos(fleets[:, F_ANGLE]) * speeds
            fleets[:, F_Y] = old_y + np.sin(fleets[:, F_ANGLE]) * speeds
            new_x = fleets[:, F_X]
            new_y = fleets[:, F_Y]

            dead |= (new_x < 0.0) | (new_x > BOARD_SIZE) | (new_y < 0.0) | (new_y > BOARD_SIZE)
            sun_d2 = segment_point_dist_sq(CENTER, CENTER, old_x, old_y, new_x, new_y)
            dead |= sun_d2 < SUN_RADIUS * SUN_RADIUS

            live = np.where(~dead)[0]
            if len(live) and len(planets):
                d2 = segment_point_dist_sq(
                    planets[:, P_X, None],
                    planets[:, P_Y, None],
                    old_x[live][None, :],
                    old_y[live][None, :],
                    new_x[live][None, :],
                    new_y[live][None, :],
                )
                hit_matrix = d2 < (planets[:, P_R, None] * planets[:, P_R, None])
                hit_cols = np.where(np.any(hit_matrix, axis=0))[0]
                for col in hit_cols:
                    fi = int(live[int(col)])
                    pi = int(np.argmax(hit_matrix[:, int(col)]))
                    arrivals.setdefault(pi, []).append(fi)
                    dead[fi] = True

        old_px = planets[:, P_X].copy() if len(planets) else np.zeros(0, dtype=np.float32)
        old_py = planets[:, P_Y].copy() if len(planets) else np.zeros(0, dtype=np.float32)
        s.step += 1
        self.update_planet_positions()

        if len(fleets) and len(planets):
            survivors = np.where(~dead)[0]
            for pi in range(len(planets)):
                if old_px[pi] == planets[pi, P_X] and old_py[pi] == planets[pi, P_Y]:
                    continue
                if len(survivors) == 0:
                    break
                d2 = segment_point_dist_sq(
                    fleets[survivors, F_X], fleets[survivors, F_Y],
                    old_px[pi], old_py[pi], planets[pi, P_X], planets[pi, P_Y],
                )
                hits = np.where(d2 < planets[pi, P_R] * planets[pi, P_R])[0]
                for hit in hits:
                    fi = int(survivors[int(hit)])
                    arrivals.setdefault(pi, []).append(fi)
                    dead[fi] = True
                survivors = np.where(~dead)[0]

        self._resolve_arrivals(arrivals, fleet_snapshot)

        if len(fleets):
            s.fleets = fleets[~dead] if np.any(~dead) else np.zeros((0, 7), dtype=np.float32)
        return s

    def _resolve_arrivals(self, arrivals: Dict[int, List[int]], fleet_snapshot: np.ndarray) -> None:
        planets = self.state.planets
        for pi, fleet_ids in arrivals.items():
            if pi >= len(planets):
                continue
            by_owner: Dict[int, int] = {}
            for fi in fleet_ids:
                if fi >= len(fleet_snapshot):
                    continue
                owner = int(fleet_snapshot[fi, F_OWNER])
                ships = int(fleet_snapshot[fi, F_SHIPS])
                by_owner[owner] = by_owner.get(owner, 0) + ships
            if not by_owner:
                continue
            ordered = sorted(by_owner.items(), key=lambda item: item[1], reverse=True)
            winner, top = ordered[0]
            second = ordered[1][1] if len(ordered) > 1 else 0
            incoming = top - second
            if incoming <= 0:
                continue
            current_owner = int(planets[pi, P_OWNER])
            if current_owner == winner:
                planets[pi, P_SHIPS] += incoming
            else:
                planets[pi, P_SHIPS] -= incoming
                if planets[pi, P_SHIPS] < 0:
                    planets[pi, P_OWNER] = winner
                    planets[pi, P_SHIPS] = -planets[pi, P_SHIPS]

    def alive_players(self) -> List[int]:
        alive = set()
        if len(self.state.planets):
            alive.update(int(p) for p in self.state.planets[:, P_OWNER] if int(p) >= 0)
        if len(self.state.fleets):
            alive.update(int(f) for f in self.state.fleets[:, F_OWNER] if int(f) >= 0)
        return sorted(alive)

    def is_terminal(self) -> bool:
        return self.state.step >= self.state.max_steps or len(self.alive_players()) <= 1

    def scores(self) -> List[int]:
        scores = [0 for _ in range(self.n_players)]
        for player in range(self.n_players):
            if len(self.state.planets):
                mask = self.state.planets[:, P_OWNER] == player
                if np.any(mask):
                    scores[player] += int(self.state.planets[mask, P_SHIPS].sum())
            if len(self.state.fleets):
                mask = self.state.fleets[:, F_OWNER] == player
                if np.any(mask):
                    scores[player] += int(self.state.fleets[mask, F_SHIPS].sum())
        return scores

    def winner(self) -> int:
        scores = self.scores()
        best = max(scores) if scores else 0
        winners = [i for i, score in enumerate(scores) if score == best]
        return winners[0] if len(winners) == 1 else -1

    def run(self, agents: Sequence[Callable], max_steps: Optional[int] = None) -> dict:
        if max_steps is not None:
            self.state.max_steps = int(max_steps)

        started = time.perf_counter()
        while not self.is_terminal():
            actions = {}
            for player, agent in enumerate(agents):
                obs = self.observation(player)
                try:
                    move = agent(obs, None)
                except TypeError:
                    move = agent(obs)
                actions[player] = move if isinstance(move, list) else []
            self.step(actions)

        elapsed = time.perf_counter() - started
        return {
            "winner": self.winner(),
            "scores": self.scores(),
            "steps": int(self.state.step),
            "seconds": elapsed,
            "steps_per_second": self.state.step / max(elapsed, 1e-9),
        }

    def run_state_policies(self, policies: Sequence[Callable], max_steps: Optional[int] = None) -> dict:
        """Run policies that read SimGame/FastState directly.

        Policy signature: `policy(game, player) -> [[from_id, angle, ships], ...]`.
        This avoids creating Kaggle-style observations every turn and is the
        preferred path for high-volume local training.
        """
        if max_steps is not None:
            self.state.max_steps = int(max_steps)

        started = time.perf_counter()
        while not self.is_terminal():
            actions = {}
            for player, policy in enumerate(policies):
                move = policy(self, player)
                actions[player] = move if isinstance(move, list) else []
            self.step(actions)

        elapsed = time.perf_counter() - started
        return {
            "winner": self.winner(),
            "scores": self.scores(),
            "steps": int(self.state.step),
            "seconds": elapsed,
            "steps_per_second": self.state.step / max(elapsed, 1e-9),
        }


def run_match(
    agents: Sequence[Callable],
    seed: Optional[int] = None,
    n_players: Optional[int] = None,
    neutral_pairs: int = 8,
    max_steps: int = TOTAL_TURNS,
) -> dict:
    players = int(n_players or len(agents))
    if len(agents) != players:
        raise ValueError(f"agent count ({len(agents)}) must match n_players ({players})")
    game = SimGame.random_game(seed=seed, n_players=players, neutral_pairs=neutral_pairs, max_steps=max_steps)
    return game.run(agents, max_steps=max_steps)


def benchmark(agent_a: Callable, agent_b: Callable, games: int = 20, seed: int = 1) -> dict:
    results = []
    for i in range(games):
        results.append(run_match([agent_a, agent_b], seed=seed + i))
    wins_a = sum(1 for r in results if r["winner"] == 0)
    wins_b = sum(1 for r in results if r["winner"] == 1)
    ties = games - wins_a - wins_b
    total_steps = sum(r["steps"] for r in results)
    total_seconds = sum(r["seconds"] for r in results)
    return {
        "games": games,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "steps": total_steps,
        "seconds": total_seconds,
        "steps_per_second": total_steps / max(total_seconds, 1e-9),
    }


def fast_greedy_policy(game: SimGame, player: int) -> list:
    state = game.state
    planets = state.planets
    if len(planets) == 0:
        return []
    my_idx = np.where(planets[:, P_OWNER] == player)[0]
    target_idx = np.where(planets[:, P_OWNER] != player)[0]
    if len(my_idx) == 0 or len(target_idx) == 0:
        return []

    targets = planets[target_idx]
    actions = []
    for idx in my_idx:
        src = planets[idx]
        ships = int(src[P_SHIPS])
        if ships < 10:
            continue
        dx = targets[:, P_X] - src[P_X]
        dy = targets[:, P_Y] - src[P_Y]
        dist = np.hypot(dx, dy)
        score = targets[:, P_PROD] * 25.0 - targets[:, P_SHIPS] * 0.5 - dist
        target = targets[int(np.argmax(score))]
        angle = math.atan2(float(target[P_Y] - src[P_Y]), float(target[P_X] - src[P_X]))
        actions.append([int(src[P_ID]), angle, ships // 2])
    return actions


def passive_policy(game: SimGame, player: int) -> list:
    return []


def benchmark_state_policies(
    policy_a: Callable = fast_greedy_policy,
    policy_b: Callable = fast_greedy_policy,
    games: int = 100,
    seed: int = 1,
) -> dict:
    results = []
    for i in range(games):
        game = SimGame.random_game(seed=seed + i)
        results.append(game.run_state_policies([policy_a, policy_b]))
    wins_a = sum(1 for r in results if r["winner"] == 0)
    wins_b = sum(1 for r in results if r["winner"] == 1)
    ties = games - wins_a - wins_b
    total_steps = sum(r["steps"] for r in results)
    total_seconds = sum(r["seconds"] for r in results)
    return {
        "games": games,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "steps": total_steps,
        "seconds": total_seconds,
        "steps_per_second": total_steps / max(total_seconds, 1e-9),
    }


__all__ = [
    "SimGame",
    "FastState",
    "run_match",
    "benchmark",
    "benchmark_state_policies",
    "fast_greedy_policy",
    "fleet_speed",
    "fleet_speeds",
    "passive_policy",
]
