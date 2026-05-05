"""Fast local runner for the official Kaggle Orbit Wars engine.

This module deliberately keeps the game rules in Kaggle's own
``orbit_wars.py``.  The speedup comes from bypassing the generic
``kaggle_environments`` wrapper during local training: no jsonschema
validation, no recursive structify/deepcopy on every step, and no captured
stdout/stderr around every agent call.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
KAGGLE_ENV_ROOT = ROOT / "github" / "kaggle-environments"
if str(KAGGLE_ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(KAGGLE_ENV_ROOT))

try:
    from kaggle_environments.envs.orbit_wars import orbit_wars  # noqa: E402
    from kaggle_environments.utils import Struct, structify  # noqa: E402
except ModuleNotFoundError:
    orbit_wars_path = KAGGLE_ENV_ROOT / "kaggle_environments" / "envs" / "orbit_wars" / "orbit_wars.py"
    spec = importlib_util.spec_from_file_location("orbit_wars_official_fast", orbit_wars_path)
    if spec is None or spec.loader is None:
        raise
    orbit_wars = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(orbit_wars)

    class Struct(dict):
        def __init__(self, **entries: Any) -> None:
            entries = {k: v for k, v in entries.items() if k != "items"}
            dict.__init__(self, entries)
            self.__dict__.update(entries)

        def __setattr__(self, attr: str, value: Any) -> None:
            self.__dict__[attr] = value
            self[attr] = value

    def structify(o: Any) -> Any:
        if isinstance(o, list):
            return [structify(item) for item in o]
        if isinstance(o, dict):
            return Struct(**{key: structify(value) for key, value in o.items()})
        return o


@dataclass
class FastConfig:
    episodeSteps: int = 500
    actTimeout: float = 1.0
    shipSpeed: float = 6.0
    cometSpeed: float = 4.0
    seed: int | None = None
    remainingOverageTime: float = 60.0

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)


class OfficialFastGame:
    """Run Orbit Wars using the official interpreter with minimal overhead."""

    def __init__(
        self,
        n_players: int = 2,
        *,
        seed: int | None = None,
        episode_steps: int = 500,
        ship_speed: float = 6.0,
        comet_speed: float = 4.0,
        remaining_overage_time: float = 60.0,
        use_c_accel: bool = True,
    ) -> None:
        self.n_players = int(n_players)
        self.seed = seed
        self.c_accel_enabled = False
        if use_c_accel:
            from local_simulator.c_accel import orbit_wars_accel

            self.c_accel_enabled = orbit_wars_accel.enable(orbit_wars)
        self.configuration = FastConfig(
            episodeSteps=int(episode_steps),
            shipSpeed=float(ship_speed),
            cometSpeed=float(comet_speed),
            seed=seed,
            remainingOverageTime=float(remaining_overage_time),
        )
        self.info: dict[str, Any] = {}
        self.state: list[Struct] = []
        self.steps: list[list[Struct]] = []
        self.logs: list[list[dict[str, Any]]] = []
        self.done = False
        self.reset()

    @property
    def agents(self) -> dict[str, Callable]:
        return {
            "random": orbit_wars.random_agent,
            "starter": orbit_wars.starter_agent,
        }

    def reset(self) -> list[Struct]:
        self.done = False
        self.info = {}
        self.configuration.seed = self.seed
        self.state = structify(
            [
                {
                    "action": [],
                    "info": {},
                    "observation": {
                        "remainingOverageTime": self.configuration.remainingOverageTime,
                        "step": 0,
                    },
                    "reward": 0,
                    "status": "ACTIVE",
                }
                for _ in range(self.n_players)
            ]
        )
        self.state = self._call_interpreter(self.state)
        self._set_step(0)
        self.steps = [self.state]
        self.logs = [[]]
        return self.state

    def observation(self, player: int) -> Struct:
        return self.state[int(player)].observation

    def step(self, actions: Sequence[Any]) -> list[Struct]:
        if self.done:
            raise RuntimeError("Environment done, reset required.")
        if len(actions) != self.n_players:
            raise ValueError(f"{self.n_players} actions required")

        next_state = []
        for index, action in enumerate(actions):
            agent_state = self.state[index]
            next_state.append(
                {
                    "action": action if isinstance(action, list) else [],
                    "info": dict(getattr(agent_state, "info", {}) or {}),
                    "observation": agent_state.observation,
                    "reward": agent_state.reward,
                    "status": agent_state.status,
                }
            )
        self.state = self._call_interpreter(structify(next_state))
        self._set_step(len(self.steps))
        if self.state[0].observation.step >= self.configuration.episodeSteps - 1:
            for agent_state in self.state:
                if agent_state.status in ("ACTIVE", "INACTIVE"):
                    agent_state.status = "DONE"
        self.done = all(agent_state.status != "ACTIVE" for agent_state in self.state)
        self.steps.append(self.state)
        self.logs.append([])
        return self.state

    def run(self, agents: Sequence[str | Callable]) -> list[list[Struct]]:
        callables = [self._resolve_agent(agent) for agent in agents]
        while not self.done:
            actions = []
            for player, agent in enumerate(callables):
                obs = self.state[player].observation
                try:
                    action = agent(obs, self.configuration)
                except TypeError:
                    action = agent(obs)
                actions.append(action if isinstance(action, list) else [])
            self.step(actions)
        return self.steps

    def scores(self) -> list[int]:
        scores = [0 for _ in range(self.n_players)]
        obs = self.state[0].observation
        for planet in obs.get("planets", []) or []:
            owner = int(planet[1])
            if 0 <= owner < self.n_players:
                scores[owner] += int(planet[5])
        for fleet in obs.get("fleets", []) or []:
            owner = int(fleet[1])
            if 0 <= owner < self.n_players:
                scores[owner] += int(fleet[4])
        return scores

    def winner(self) -> int:
        scores = self.scores()
        best = max(scores) if scores else 0
        winners = [i for i, score in enumerate(scores) if score == best and best > 0]
        return winners[0] if len(winners) == 1 else -1

    def result(self, tracked_player: int = 0, started: float | None = None) -> dict[str, Any]:
        elapsed = time.perf_counter() - started if started is not None else 0.0
        step = int(self.state[0].observation.get("step", 0) or 0)
        return {
            "winner": self.winner(),
            "scores": self.scores(),
            "steps": step,
            "seconds": elapsed,
            "steps_per_second": step / max(elapsed, 1e-9) if elapsed else 0.0,
            "tracked_player": int(tracked_player),
        }

    def _call_interpreter(self, state: list[Struct]) -> list[Struct]:
        env = SimpleNamespace(
            configuration=self.configuration,
            done=self.done,
            info=self.info,
        )
        new_state = orbit_wars.interpreter(state, env)
        self.info = env.info
        return new_state

    def _set_step(self, step: int) -> None:
        # Kaggle core.py updates only new_state[0].observation.step after the
        # interpreter returns.  The shared game data is copied to all players,
        # but the base "step" field is not.  Preserve that quirk exactly.
        if self.state:
            self.state[0].observation.step = int(step)

    def _resolve_agent(self, agent: str | Callable) -> Callable:
        if isinstance(agent, str):
            if agent not in self.agents:
                raise ValueError(f"Unknown built-in Orbit Wars agent: {agent}")
            return self.agents[agent]
        return agent


def run_fast_game(
    agents: Sequence[str | Callable],
    *,
    n_players: int | None = None,
    seed: int | None = None,
    max_steps: int = 500,
    tracked_player: int = 0,
    stop_player: int | None = None,
    overage_time: float = 60.0,
    use_c_accel: bool = True,
) -> dict[str, Any]:
    """Convenience wrapper used by training code."""

    n = int(n_players if n_players is not None else len(agents))
    game = OfficialFastGame(
        n,
        seed=seed,
        episode_steps=max_steps,
        remaining_overage_time=overage_time,
        use_c_accel=use_c_accel,
    )
    started = time.perf_counter()
    stop_player = None if stop_player is None else int(stop_player)
    tracked_player = int(tracked_player)
    initial_player = stop_player if stop_player is not None else tracked_player
    initial_state = game.observation(initial_player)
    max_planets = _owned_planets(game.observation(tracked_player), tracked_player)
    planets_t60 = None
    planets_t100 = None

    callables = [game._resolve_agent(agent) for agent in agents]
    while not game.done:
        actions = []
        for player, agent in enumerate(callables):
            obs = game.observation(player)
            try:
                move = agent(obs, game.configuration)
            except TypeError:
                move = agent(obs)
            actions.append(move if isinstance(move, list) else [])
        game.step(actions)
        if stop_player is not None and not _player_alive(game.observation(stop_player), stop_player):
            break
        tracked_obs = game.observation(tracked_player)
        owned = _owned_planets(tracked_obs, tracked_player)
        max_planets = max(max_planets, owned)
        step = int(tracked_obs.get("step", 0) or 0)
        if planets_t60 is None and step >= 60:
            planets_t60 = owned
        if planets_t100 is None and step >= 100:
            planets_t100 = owned

    result = game.result(tracked_player, started)
    result["max_planets"] = int(max_planets)
    result["planets_t60"] = int(planets_t60 if planets_t60 is not None else max_planets)
    result["planets_t100"] = int(planets_t100 if planets_t100 is not None else max_planets)
    result["initial_state"] = initial_state
    result["final_state"] = game.observation(initial_player)
    return result


def comparable_state(state: Sequence[Any]) -> list[dict[str, Any]]:
    """Convert a state list into plain data for exact equality checks."""

    out = []
    for agent_state in state:
        obs = agent_state.observation
        out.append(
            {
                "action": list(getattr(agent_state, "action", []) or []),
                "reward": getattr(agent_state, "reward", 0),
                "status": getattr(agent_state, "status", ""),
                "observation": {
                    "step": int(obs.get("step", 0) or 0),
                    "player": int(obs.get("player", 0) or 0),
                    "angular_velocity": obs.get("angular_velocity", 0),
                    "planets": obs.get("planets", []),
                    "fleets": obs.get("fleets", []),
                    "initial_planets": obs.get("initial_planets", []),
                    "next_fleet_id": int(obs.get("next_fleet_id", 0) or 0),
                    "comets": obs.get("comets", []),
                    "comet_planet_ids": obs.get("comet_planet_ids", []),
                },
            }
        )
    return out


def _owned_planets(obs: Any, player: int) -> int:
    return sum(1 for planet in obs.get("planets", []) or [] if int(planet[1]) == int(player))


def _player_alive(obs: Any, player: int) -> bool:
    return any(int(planet[1]) == int(player) for planet in obs.get("planets", []) or []) or any(
        int(fleet[1]) == int(player) for fleet in obs.get("fleets", []) or []
    )


def noop_agent(obs: Any, config: Any = None) -> list:
    return []


def deterministic_expand_agent(obs: Any, config: Any = None) -> list[list[float]]:
    """Small deterministic agent for equivalence tests."""

    player = int(obs.get("player", 0) or 0)
    planets = list(obs.get("planets", []) or [])
    my_planets = [p for p in planets if int(p[1]) == player and int(p[5]) >= 12]
    targets = [p for p in planets if int(p[1]) != player]
    if not my_planets or not targets:
        return []

    moves = []
    for src in sorted(my_planets, key=lambda p: (-int(p[5]), int(p[0])))[:2]:
        target = min(
            targets,
            key=lambda p: (
                int(p[1]) != -1,
                float(p[5]) - 2.0 * float(p[6]),
                (float(p[2]) - float(src[2])) ** 2 + (float(p[3]) - float(src[3])) ** 2,
                int(p[0]),
            ),
        )
        angle = __import__("math").atan2(float(target[3]) - float(src[3]), float(target[2]) - float(src[2]))
        ships = max(1, int(src[5]) // 2)
        moves.append([int(src[0]), angle, ships])
    return moves
