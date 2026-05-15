"""benchmark_v15 — winrate of bot_v15 vs a fixed panel, with Wilson 95% CI.

bot_v15 reads its V15Config from V15_* env vars (see v15_config.from_env).
This script does NOT set them — set them in the shell before launching so the
worker subprocesses inherit them.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python -u benchmark_v15.py --games 200 --workers 4
    V15_FOUR_PLAYER_SEND_RATIO=0.90 KMP_DUPLICATE_LIB_OK=TRUE python -u benchmark_v15.py --games 200

Panel (deliberately mixed strength so winrate is measurable, not 0% or 100%):
    bot_v7 self, bot_v12, notebook_distance_prioritized, notebook_orbitbotnext
"""

from __future__ import annotations

import argparse
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import bot_v7
import bot_v12
import bot_v15
import v14_core
from local_simulator.official_fast import OfficialFastGame
from opponents import ZOO

PANEL = ["v7", "v12", "notebook_distance_prioritized", "notebook_orbitbotnext"]


def _resolve(name):
    if name == "v7":
        return bot_v7.agent
    if name == "v12":
        return bot_v12.agent
    if name == "v15":
        return bot_v15.agent
    return ZOO[name]


def _call(fn, obs, config):
    obs = v14_core.obs_as_dict(obs)
    try:
        move = fn(obs, config)
    except TypeError:
        move = fn(obs)
    return move if isinstance(move, list) else []


def _wilson_lb(wins: int, n: int, z: float = 1.96) -> float:
    """Wilson score interval lower bound for a binomial proportion."""
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (centre - margin) / denom


def _play(task):
    n_players, seed, max_steps = task
    random.seed(seed)
    np.random.seed(seed)
    # Opponent selection deterministic per seed.
    rng = random.Random(seed)
    our = bot_v15.agent
    if n_players == 2:
        opp = _resolve(PANEL[seed % len(PANEL)])
        our_idx = seed % 2
        agents = [our, opp] if our_idx == 0 else [opp, our]
    else:
        opps = [_resolve(PANEL[(seed + j) % len(PANEL)]) for j in range(3)]
        our_idx = seed % 4
        agents = []
        it = iter(opps)
        for i in range(4):
            agents.append(our if i == our_idx else next(it))
    game = OfficialFastGame(n_players=n_players, seed=seed,
                            episode_steps=max_steps, use_c_accel=True)
    start = time.time()
    while not game.done:
        actions = [_call(fn, game.observation(p), game.configuration)
                   for p, fn in enumerate(agents)]
        game.step(actions)
    scores = game.scores()
    best_other = max(s for i, s in enumerate(scores) if i != our_idx)
    mine = scores[our_idx]
    if mine > best_other and mine > 0:
        return 1, time.time() - start
    if mine == best_other:
        return 0, time.time() - start
    return -1, time.time() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200, help="games per mode")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed-offset", type=int, default=70000)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--modes", nargs="*", default=["2p", "4p"])
    args = ap.parse_args()

    cfg = bot_v15._get_config()
    print(f"benchmark_v15 | games/mode={args.games} workers={args.workers}")
    print(f"config passthrough={cfg.is_passthrough()} overrides={cfg.overrides_dict()} "
          f"flags: multi_bonus={cfg.enable_multi_source_early_bonus} "
          f"opp_gate={cfg.enable_opportunistic_expand_gate}")

    grand_w = grand_n = 0
    for mode in args.modes:
        n_players = 2 if mode == "2p" else 4
        tasks = [(n_players, args.seed_offset + i, args.max_steps)
                 for i in range(args.games)]
        if args.workers <= 1:
            results = [_play(t) for t in tasks]
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(_play, tasks))
        w = sum(1 for o, _ in results if o > 0)
        d = sum(1 for o, _ in results if o == 0)
        n = len(results)
        secs = sum(s for _, s in results)
        lb = _wilson_lb(w, n)
        print(f"- {mode}  W/D/L={w}/{d}/{n - w - d}  WR={w / n:.3f}  "
              f"Wilson95%LB={lb:.3f}  secs={secs:.0f}")
        grand_w += w
        grand_n += n
    lb = _wilson_lb(grand_w, grand_n)
    print(f"AGG  W={grand_w}/{grand_n}  WR={grand_w / grand_n:.3f}  Wilson95%LB={lb:.3f}")


if __name__ == "__main__":
    main()
