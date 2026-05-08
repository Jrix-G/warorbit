"""
test_speed.py — Benchmark Python vs C engine on identical workloads.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from local_simulator.official_fast import OfficialFastGame  # noqa: E402
from c_engine import CGame  # noqa: E402
from opponents import ZOO  # noqa: E402


def run_game(GameCls, seed, agent0, agent1, max_steps=220, **kwargs):
    g = GameCls(2, seed=seed, episode_steps=max_steps, **kwargs)
    n = 0
    while not g.done and n < max_steps:
        a0 = agent0(g.observation(0))
        a1 = agent1(g.observation(1))
        g.step([a0, a1])
        n += 1
    return n


def bench(label, GameCls, seeds, agent0, agent1, **kwargs):
    t0 = time.perf_counter()
    total_steps = 0
    for seed in seeds:
        total_steps += run_game(GameCls, seed, agent0, agent1, **kwargs)
    dt = time.perf_counter() - t0
    sps = total_steps / dt if dt > 0 else 0
    return dt, total_steps, sps


def main():
    import bot_v12

    seeds = list(range(2000, 2030))   # 30 games per engine

    print("=== Benchmark Python (OfficialFastGame, no C-accel) vs C engine ===")
    print(f"30 games × ~220 steps each, V12 bot vs V12 bot\n")

    py_dt, py_steps, py_sps = bench(
        "Python", OfficialFastGame, seeds, bot_v12.agent, bot_v12.agent,
        use_c_accel=False)
    print(f"Python: {py_dt:6.2f}s  {py_steps} steps  {py_sps:6.1f} steps/s")

    py_caccel_dt, py_caccel_steps, py_caccel_sps = bench(
        "Python+c_accel", OfficialFastGame, seeds, bot_v12.agent, bot_v12.agent,
        use_c_accel=True)
    print(f"Python+c_accel: {py_caccel_dt:6.2f}s  {py_caccel_steps} steps  {py_caccel_sps:6.1f} steps/s")

    c_dt, c_steps, c_sps = bench(
        "C engine", CGame, seeds, bot_v12.agent, bot_v12.agent)
    print(f"C engine: {c_dt:6.2f}s  {c_steps} steps  {c_sps:6.1f} steps/s")

    print()
    print(f"  Speedup vs Python pure: {py_dt/c_dt:.2f}x")
    print(f"  Speedup vs Python+c_accel: {py_caccel_dt/c_dt:.2f}x")
    print(f"  Steps/s gain: {(c_sps/py_sps - 1)*100:+.0f}%")


if __name__ == '__main__':
    main()
