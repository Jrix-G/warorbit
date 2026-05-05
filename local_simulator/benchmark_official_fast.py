from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_ENV_ROOT = ROOT / "github" / "kaggle-environments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(KAGGLE_ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(KAGGLE_ENV_ROOT))

from kaggle_environments import make

from local_simulator.official_fast import OfficialFastGame, deterministic_expand_agent, noop_agent


def bench_kaggle(agents, games: int, episode_steps: int) -> tuple[float, int]:
    total_steps = 0
    start = time.perf_counter()
    for seed in range(games):
        env = make(
            "orbit_wars",
            configuration={"episodeSteps": episode_steps, "seed": seed},
            debug=False,
        )
        steps = env.run(list(agents))
        total_steps += len(steps)
    return time.perf_counter() - start, total_steps


def bench_fast(agents, games: int, episode_steps: int) -> tuple[float, int]:
    total_steps = 0
    start = time.perf_counter()
    for seed in range(games):
        game = OfficialFastGame(len(agents), seed=seed, episode_steps=episode_steps, use_c_accel=False)
        steps = game.run(list(agents))
        total_steps += len(steps)
    return time.perf_counter() - start, total_steps


def bench_fast_c(agents, games: int, episode_steps: int) -> tuple[float, int, bool]:
    total_steps = 0
    enabled = False
    start = time.perf_counter()
    for seed in range(games):
        game = OfficialFastGame(len(agents), seed=seed, episode_steps=episode_steps, use_c_accel=True)
        enabled = enabled or game.c_accel_enabled
        steps = game.run(list(agents))
        total_steps += len(steps)
    return time.perf_counter() - start, total_steps, enabled


def report(label: str, seconds: float, games: int, steps: int) -> None:
    print(
        f"{label}: {seconds:.4f}s total, "
        f"{seconds / max(1, games):.4f}s/game, "
        f"{steps / max(seconds, 1e-9):.1f} steps/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument(
        "--agent-set",
        choices=["noop", "expand"],
        default="noop",
    )
    args = parser.parse_args()

    if args.agent_set == "noop":
        agents = [noop_agent, noop_agent]
    else:
        agents = [deterministic_expand_agent, deterministic_expand_agent]

    kaggle_seconds, kaggle_steps = bench_kaggle(agents, args.games, args.episode_steps)
    fast_seconds, fast_steps = bench_fast(agents, args.games, args.episode_steps)
    fast_c_seconds, fast_c_steps, c_enabled = bench_fast_c(agents, args.games, args.episode_steps)
    report("kaggle_make_run", kaggle_seconds, args.games, kaggle_steps)
    report("official_fast_python", fast_seconds, args.games, fast_steps)
    report("official_fast_c", fast_c_seconds, args.games, fast_c_steps)
    print(f"c_accel_enabled: {c_enabled}")
    print(f"speedup_vs_kaggle_python_runner: {kaggle_seconds / max(fast_seconds, 1e-9):.2f}x")
    print(f"speedup_vs_kaggle_c_runner: {kaggle_seconds / max(fast_c_seconds, 1e-9):.2f}x")
    print(f"speedup_c_over_fast_python: {fast_seconds / max(fast_c_seconds, 1e-9):.2f}x")
    print(f"saved_per_game_vs_kaggle: {(kaggle_seconds - fast_c_seconds) / max(1, args.games):.4f}s")


if __name__ == "__main__":
    main()
