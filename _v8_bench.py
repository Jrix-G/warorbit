"""V8 vs V7 benchmark on training_pool(15) with 70% 4p / 30% 2p mix."""
from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ProcessPoolExecutor

from SimGame import run_match
import bot_v7
import bot_v8
from opponents import ZOO, training_pool


def _play(task):
    bot_label, opp_name, n_players, our_slot, seed = task
    bot_agent = bot_v8.agent if bot_label == "v8" else bot_v7.agent
    opp = ZOO[opp_name]
    agents = [opp] * n_players
    agents[our_slot] = bot_agent
    t0 = time.time()
    r = run_match(agents, seed=seed, n_players=n_players)
    dt = time.time() - t0
    winner = int(r.get("winner", -1))
    return bot_label, opp_name, n_players, (winner == our_slot), dt, int(r.get("steps", 0))


def build_tasks(bot_label, opps, games_per_opp, ratio_4p, seed_offset):
    tasks = []
    rng = random.Random(seed_offset)
    for opp_name in opps:
        for i in range(games_per_opp):
            seed = seed_offset + i + hash(opp_name) % 9999
            n_players = 4 if rng.random() < ratio_4p else 2
            our_slot = i % n_players
            tasks.append((bot_label, opp_name, n_players, our_slot, seed))
    return tasks


def run(bot_label, opps, games_per_opp, ratio_4p, workers, seed_offset):
    tasks = build_tasks(bot_label, opps, games_per_opp, ratio_4p, seed_offset)
    t0 = time.time()
    if workers <= 1:
        results = [_play(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_play, tasks))
    total_t = time.time() - t0

    by_opp = {}
    wins = 0
    total_dt = 0.0
    total_steps = 0
    for _, opp_name, _, won, dt, steps in results:
        d = by_opp.setdefault(opp_name, [0, 0])
        d[1] += 1
        if won:
            d[0] += 1
            wins += 1
        total_dt += dt
        total_steps += steps
    n = len(results)
    print(f"\n=== {bot_label.upper()} | {n} games | {total_t:.1f}s wall | "
          f"{total_dt/n:.1f}s/game cpu | {total_steps/n:.0f} steps/game ===")
    for opp, (w, g) in sorted(by_opp.items()):
        print(f"  {opp:50s} {w:2d}/{g:2d}  {100*w/g:5.1f}%")
    print(f"  TOTAL: {wins}/{n}  {100*wins/n:.1f}%")
    return wins / n, total_dt / n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per-opp", type=int, default=6)
    parser.add_argument("--ratio-4p", type=float, default=0.70)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed-offset", type=int, default=4242)
    parser.add_argument("--bots", nargs="+", default=["v8", "v7"])
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    opps = training_pool(args.limit)
    print(f"training_pool({args.limit}) -> {len(opps)} opps:")
    for o in opps:
        print(f"  - {o}")

    summary = {}
    for label in args.bots:
        wr, sg = run(label, opps, args.games_per_opp, args.ratio_4p, args.workers, args.seed_offset)
        summary[label] = (wr, sg)

    print("\n==== SUMMARY ====")
    for label, (wr, sg) in summary.items():
        print(f"  {label}: WR={100*wr:.1f}%  s/game={sg:.2f}")
