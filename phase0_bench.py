"""Phase 0 — measure the self-play loop cost BEFORE committing to the plan.

Five mini-tests (~5-10 min total):
  T4  GPU / library availability        -> batch ceiling
  T5  raw engine step throughput        -> GPU-batching payoff estimate
  T1  RCC search time (early/mid/late)  -> per-move cost
  T2  single self-play game duration    -> s/game
  T3  24-game throughput on 8 workers   -> extrapolated time/generation

Run:
    KMP_DUPLICATE_LIB_OK=TRUE python -u phase0_bench.py
"""

from __future__ import annotations

import random
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import v14_core
import v15_fast_sim as fsim
import v15_search
from local_simulator.official_fast import OfficialFastGame


def gpu_info():
    lines = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        txt = (out.stdout or out.stderr).strip()
        lines.append("nvidia-smi: " + (txt or "no GPU detected"))
    except Exception as e:
        lines.append(f"nvidia-smi: unavailable ({type(e).__name__})")
    for mod in ("jax", "cupy", "torch"):
        try:
            m = __import__(mod)
            lines.append(f"  {mod}: installed {getattr(m, '__version__', '?')}")
        except Exception:
            lines.append(f"  {mod}: not installed")
    return lines


def engine_throughput():
    random.seed(1)
    np.random.seed(1)
    g = OfficialFastGame(4, seed=1, episode_steps=500, use_c_accel=False)
    for _ in range(30):
        g.step([[] for _ in range(4)])
    obs = v14_core.obs_as_dict(g.observation(0))
    base = fsim.from_obs(obs, n_players=4)
    base.n_players = 4
    empty = [[] for _ in range(4)]
    n = 3000
    cur = base
    t = time.monotonic()
    for _ in range(n):
        cur = fsim.step(cur, empty)
        if cur.done:
            cur = base.copy()
    dt = time.monotonic() - t
    return n / dt, dt / n * 1e6


def search_times():
    out = {}
    for stepn in (30, 110, 180):
        random.seed(5)
        np.random.seed(5)
        g = OfficialFastGame(4, seed=5, episode_steps=200, use_c_accel=False)
        for _ in range(stepn):
            g.step([[] for _ in range(4)])
        obs = v14_core.obs_as_dict(g.observation(0))
        ts = []
        for _ in range(5):
            t = time.monotonic()
            v15_search.search(dict(obs), g.configuration, time_budget=0.7)
            ts.append(time.monotonic() - t)
        out[stepn] = sum(ts) / len(ts) * 1000
    return out


def _rcc(obs, config):
    obs = v14_core.obs_as_dict(obs)
    m = v15_search.search(obs, config, time_budget=0.7)
    return m if isinstance(m, list) else []


def _play(task):
    n_players, seed = task
    random.seed(seed)
    np.random.seed(seed)
    g = OfficialFastGame(n_players, seed=seed, episode_steps=200,
                         use_c_accel=False)
    t = time.monotonic()
    while not g.done:
        g.step([_rcc(g.observation(p), g.configuration)
                for p in range(n_players)])
    return n_players, time.monotonic() - t


def main():
    print("=== Phase 0 — self-play loop cost ===\n")

    print("[T4] GPU / libraries")
    for ln in gpu_info():
        print("  " + ln)

    print("\n[T5] engine raw throughput (single core)")
    sps, usp = engine_throughput()
    print(f"  v15_fast_sim: {sps:.0f} steps/sec  ({usp:.0f} us/step)")

    print("\n[T1] RCC search time (depth-1, passive continuation)")
    for k, v in search_times().items():
        print(f"  game step {k:3d}: {v:.0f} ms/move")

    print("\n[T2/T3] self-play games (RCC vs RCC)")
    tasks = ([(2, 500000 + i) for i in range(12)]
             + [(4, 510000 + i) for i in range(12)])
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=8) as pool:
        res = list(pool.map(_play, tasks))
    wall = time.monotonic() - t0
    g2 = [d for n, d in res if n == 2]
    g4 = [d for n, d in res if n == 4]
    print(f"  2p: avg {sum(g2)/len(g2):.0f}s/game (single-core time)")
    print(f"  4p: avg {sum(g4)/len(g4):.0f}s/game (single-core time)")
    print(f"  24 games on 8 workers: {wall:.0f}s wall  "
          f"({wall/24:.1f}s/game effective)")

    per = wall / 24
    print("\n[extrapolation] depth-1 RCC, 8 CPU workers:")
    for gen in (3000, 5000):
        print(f"  generation of {gen} games: {gen*per/3600:.1f}h")
    print("  depth-2 adversarial search ~= 3-5x slower per move:")
    for gen in (3000, 5000):
        print(f"  generation of {gen} games (depth-2): "
              f"{gen*per*4/3600:.1f}h (mid estimate)")


if __name__ == "__main__":
    main()
