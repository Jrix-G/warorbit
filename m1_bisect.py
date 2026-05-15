"""M1 — per-patch bisect of Layer 2a constant patches.

Paired design: baseline and every patch play the SAME seed set, so the delta
estimate has variance dominated by patch effect, not map luck. Decision rule
(review V15_REVIEW.md §8 reco 4): keep a patch iff the 95% bootstrap CI lower
bound of (patch_winrate - baseline_winrate) >= -0.01.

Output: analysis/V15_M1_BISECT.json + console table.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE python -u m1_bisect.py --games 100 --workers 8
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import bot_v7
import bot_v12
import bot_v15
import v14_core
import v15_config
from local_simulator.official_fast import OfficialFastGame
from opponents import ZOO

PANEL = ["v7", "v12", "notebook_distance_prioritized", "notebook_orbitbotnext"]

# name -> V15Config kwargs. "baseline" is passthrough.
PATCHES: dict[str, dict] = {
    "baseline":            {},
    "multi_k_14":          {"multi_source_top_k": 14},
    "multi_pen_099":       {"multi_source_plan_penalty": 0.99},
    "tri_pen_097":         {"three_source_plan_penalty": 0.97},
    "send_4p_090":         {"four_player_rotating_send_ratio": 0.90},
    "neut_4p_086":         {"four_player_neutral_score_mult": 0.86},
    "turn_limit_4p_22":    {"four_player_rotating_turn_limit": 22},
}


def _resolve(name):
    if name == "v7":
        return bot_v7.agent
    if name == "v12":
        return bot_v12.agent
    return ZOO[name]


def _call(fn, obs, config):
    obs = v14_core.obs_as_dict(obs)
    try:
        move = fn(obs, config)
    except TypeError:
        move = fn(obs)
    return move if isinstance(move, list) else []


def _play(task):
    patch_name, kwargs, n_players, seed, max_steps = task
    bot_v15.set_config(v15_config.V15Config(**kwargs))
    random.seed(seed)
    np.random.seed(seed)
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
    while not game.done:
        actions = [_call(fn, game.observation(p), game.configuration)
                   for p, fn in enumerate(agents)]
        game.step(actions)
    scores = game.scores()
    best_other = max(s for i, s in enumerate(scores) if i != our_idx)
    mine = scores[our_idx]
    win = 1 if (mine > best_other and mine > 0) else 0
    return patch_name, n_players, seed, win


def _bootstrap_delta_ci(patch_wins, base_wins, iters=10000, seed=12345):
    """Paired bootstrap CI for mean(patch - base) over games."""
    rng = np.random.default_rng(seed)
    diffs = np.array(patch_wins, dtype=float) - np.array(base_wins, dtype=float)
    n = len(diffs)
    means = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, n)
        means[i] = diffs[idx].mean()
    return float(diffs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100, help="seeds per mode")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed-offset", type=int, default=80000)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--out", default="analysis/V15_M1_BISECT.json")
    args = ap.parse_args()

    modes = [("2p", 2), ("4p", 4)]
    tasks = []
    for patch_name, kwargs in PATCHES.items():
        for _mode, n_players in modes:
            for i in range(args.games):
                tasks.append((patch_name, kwargs, n_players,
                              args.seed_offset + i, args.max_steps))
    print(f"M1 bisect | {len(PATCHES)} configs x {len(modes)} modes x {args.games} "
          f"seeds = {len(tasks)} games | workers={args.workers}")

    t0 = time.time()
    if args.workers <= 1:
        results = [_play(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_play, tasks))
    print(f"played in {time.time() - t0:.0f}s")

    # Index: (patch, n_players, seed) -> win
    table: dict = {}
    for patch_name, n_players, seed, win in results:
        table[(patch_name, n_players, seed)] = win

    seeds = [args.seed_offset + i for i in range(args.games)]
    summary = {}
    base_key = "baseline"
    # Per-mode and aggregate paired vectors.
    for patch_name in PATCHES:
        if patch_name == base_key:
            continue
        rec = {}
        agg_patch, agg_base = [], []
        for mode, n_players in modes:
            pv = [table[(patch_name, n_players, s)] for s in seeds]
            bv = [table[(base_key, n_players, s)] for s in seeds]
            d, lo, hi = _bootstrap_delta_ci(pv, bv)
            rec[mode] = {"patch_wr": sum(pv) / len(pv), "base_wr": sum(bv) / len(bv),
                         "delta": d, "ci_lo": lo, "ci_hi": hi}
            agg_patch += pv
            agg_base += bv
        d, lo, hi = _bootstrap_delta_ci(agg_patch, agg_base)
        keep = lo >= -0.01
        rec["agg"] = {"patch_wr": sum(agg_patch) / len(agg_patch),
                      "base_wr": sum(agg_base) / len(agg_base),
                      "delta": d, "ci_lo": lo, "ci_hi": hi, "keep": keep}
        summary[patch_name] = rec

    base_agg = []
    for _mode, n_players in modes:
        base_agg += [table[(base_key, n_players, s)] for s in seeds]
    base_wr = sum(base_agg) / len(base_agg)

    print(f"\nbaseline agg WR = {base_wr:.3f}\n")
    print(f"{'patch':<20} {'delta':>8} {'95%CI':>20} {'keep':>6}")
    for patch_name, rec in summary.items():
        a = rec["agg"]
        print(f"{patch_name:<20} {a['delta']:+8.3f} "
              f"[{a['ci_lo']:+.3f},{a['ci_hi']:+.3f}]   {'YES' if a['keep'] else 'no':>6}")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"baseline_wr": base_wr, "games_per_mode": args.games,
                               "patches": summary}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
