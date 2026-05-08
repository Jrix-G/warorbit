"""5-minute speed benchmark: kaggle.make vs OfficialFastGame (py) vs +c_accel vs CGame.

Budget per engine ≈ 60s. Workload: 2-player V12 vs V12, episode_steps=200.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from kaggle_environments import make
from local_simulator.official_fast import OfficialFastGame
from c_engine import CGame
from local_simulator.official_fast import noop_agent

AGENT = noop_agent
EP_STEPS = 200
BUDGET_S = 60.0


def bench_kaggle(seed_start: int) -> tuple[float, int, int]:
    t0 = time.perf_counter()
    games = 0
    total_steps = 0
    seed = seed_start
    while time.perf_counter() - t0 < BUDGET_S:
        env = make("orbit_wars",
                   configuration={"episodeSteps": EP_STEPS, "seed": seed},
                   debug=False)
        steps = env.run([AGENT, AGENT])
        total_steps += len(steps)
        games += 1
        seed += 1
    return time.perf_counter() - t0, games, total_steps


def bench_local(use_c_accel: bool, seed_start: int) -> tuple[float, int, int]:
    t0 = time.perf_counter()
    games = 0
    total_steps = 0
    seed = seed_start
    while time.perf_counter() - t0 < BUDGET_S:
        g = OfficialFastGame(2, seed=seed, episode_steps=EP_STEPS,
                             use_c_accel=use_c_accel)
        n = 0
        while not g.done and n < EP_STEPS:
            a0 = AGENT(g.observation(0))
            a1 = AGENT(g.observation(1))
            g.step([a0, a1])
            n += 1
        total_steps += n
        games += 1
        seed += 1
    return time.perf_counter() - t0, games, total_steps


def bench_cgame(seed_start: int) -> tuple[float, int, int]:
    t0 = time.perf_counter()
    games = 0
    total_steps = 0
    seed = seed_start
    while time.perf_counter() - t0 < BUDGET_S:
        g = CGame(2, seed=seed, episode_steps=EP_STEPS)
        n = 0
        while not g.done and n < EP_STEPS:
            a0 = AGENT(g.observation(0))
            a1 = AGENT(g.observation(1))
            g.step([a0, a1])
            n += 1
        total_steps += n
        games += 1
        seed += 1
    return time.perf_counter() - t0, games, total_steps


def report(label: str, dt: float, games: int, steps: int) -> None:
    print(f"  {label:32s} {dt:6.2f}s  {games:4d} games  {steps:6d} steps  "
          f"{steps/dt:8.1f} steps/s  {games/dt:6.2f} games/s")


def main() -> None:
    print(f"=== 5min benchmark — V12 vs V12, episode_steps={EP_STEPS}, "
          f"budget {BUDGET_S}s/engine ===\n")

    print("Warmup…")
    bench_cgame(50000)  # warmup, discard

    print("\nResults:")
    k_dt, k_g, k_s = bench_kaggle(3000)
    report("kaggle.make (official)", k_dt, k_g, k_s)

    p_dt, p_g, p_s = bench_local(False, 4000)
    report("local_simulator (python)", p_dt, p_g, p_s)

    pc_dt, pc_g, pc_s = bench_local(True, 5000)
    report("local_simulator (+c_accel)", pc_dt, pc_g, pc_s)

    c_dt, c_g, c_s = bench_cgame(6000)
    report("c_engine (full C)", c_dt, c_g, c_s)

    print()
    print(f"Speedup vs kaggle.make:")
    print(f"  local_simulator python : {(k_s/k_dt) and (p_s/p_dt)/(k_s/k_dt):.2f}x")
    print(f"  local_simulator c_accel: {(pc_s/pc_dt)/(k_s/k_dt):.2f}x")
    print(f"  c_engine full          : {(c_s/c_dt)/(k_s/k_dt):.2f}x")
    print(f"\nSpeedup c_engine vs local_simulator python: {(c_s/c_dt)/(p_s/p_dt):.2f}x")
    print(f"Speedup c_engine vs local_simulator c_accel: {(c_s/c_dt)/(pc_s/pc_dt):.2f}x")


if __name__ == "__main__":
    main()
