#!/usr/bin/env python3
"""Compare two V8 checkpoints on a fast opponent panel.

Use this to validate that training moved the model in a useful direction,
without paying the cost of benchmarking against the slow notebook opponents.

Reports, per opponent and globally:
- win rate of model A
- win rate of model B
- delta (B - A) and a normal-approx confidence interval

Example:
    python diagnostics/compare_v8_models.py \
        --model-a evaluations/v8_zero.npz \
        --model-b evaluations/v8_policy_best.npz \
        --games 60 --opponents greedy,distance,structured,sun_dodge
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import bot_v7  # noqa: E402
import bot_v8  # noqa: E402
from SimGame import run_match  # noqa: E402
from v8_core import LinearV8Model  # noqa: E402


def _load_opponent(name: str) -> Callable:
    from opponents import ZOO

    if name not in ZOO:
        raise SystemExit(f"unknown opponent {name!r}; available: {sorted(ZOO.keys())}")
    return ZOO[name]


def _load_model(path: Optional[str]) -> Tuple[LinearV8Model, str]:
    if path is None or path == "" or path.lower() == "zero":
        return LinearV8Model.zero(), "zero"
    if not os.path.exists(path):
        raise SystemExit(f"checkpoint not found: {path}")
    return LinearV8Model.load(path), path


def _winrate_with_ci(wins: int, games: int, z: float = 1.96) -> Tuple[float, float]:
    if games <= 0:
        return 0.0, 0.0
    p = wins / games
    se = math.sqrt(max(p * (1.0 - p) / games, 1e-12))
    return p, z * se


def benchmark_model(
    model: LinearV8Model,
    opponent_name: str,
    games: int,
    seed_start: int,
) -> Tuple[int, float]:
    """Returns (wins, mean_seconds_per_game)."""
    bot_v7.set_scorer(None)
    bot_v8.set_model(model)
    opp = _load_opponent(opponent_name)

    wins = 0
    total_sec = 0.0
    for i in range(games):
        seed = seed_start + i
        if i % 2 == 0:
            res = run_match([bot_v8.agent, opp], seed=seed)
            wins += 1 if res["winner"] == 0 else 0
        else:
            res = run_match([opp, bot_v8.agent], seed=seed)
            wins += 1 if res["winner"] == 1 else 0
        total_sec += float(res.get("seconds", 0.0))
    return wins, total_sec / max(1, games)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-a", default="zero", help="path to checkpoint A or 'zero' for zero weights")
    p.add_argument("--model-b", default="evaluations/v8_policy_best.npz")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--opponents", default="greedy,distance,structured,sun_dodge")
    p.add_argument("--seed-start", type=int, default=4242)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_a, label_a = _load_model(args.model_a)
    model_b, label_b = _load_model(args.model_b)
    opps = [s.strip() for s in args.opponents.split(",") if s.strip()]

    print(f"A = {label_a}")
    print(f"B = {label_b}")
    print(f"Opponents: {opps}")
    print(f"Games per opponent: {args.games} (alternating sides, {args.games // 2} per side)")
    print()

    rows = []
    total_a = total_b = 0
    total_games = 0
    t0 = time.perf_counter()

    for opp_name in opps:
        wa, ta = benchmark_model(model_a, opp_name, args.games, args.seed_start)
        wb, tb = benchmark_model(model_b, opp_name, args.games, args.seed_start)
        pa, ea = _winrate_with_ci(wa, args.games)
        pb, eb = _winrate_with_ci(wb, args.games)
        delta = pb - pa
        # Variance of paired difference, using independent-sample bound (conservative).
        delta_se = math.sqrt(ea * ea + eb * eb) / 1.96
        rows.append({
            "opp": opp_name,
            "a": pa, "a_err": ea, "wa": wa,
            "b": pb, "b_err": eb, "wb": wb,
            "delta": delta, "delta_ci": 1.96 * delta_se,
            "ta": ta, "tb": tb,
        })
        total_a += wa
        total_b += wb
        total_games += args.games
        print(
            f"  {opp_name:<14s}  A={pa:.3f}+/-{ea:.3f} ({wa}/{args.games})   "
            f"B={pb:.3f}+/-{eb:.3f} ({wb}/{args.games})   "
            f"delta={delta:+.3f}+/-{1.96*delta_se:.3f}   "
            f"({ta:.2f}s/{tb:.2f}s per game)"
        )

    pa_g, ea_g = _winrate_with_ci(total_a, total_games)
    pb_g, eb_g = _winrate_with_ci(total_b, total_games)
    delta_g = pb_g - pa_g
    delta_se_g = math.sqrt(ea_g * ea_g + eb_g * eb_g) / 1.96
    elapsed = time.perf_counter() - t0
    print()
    print(f"Global: A={pa_g:.3f}+/-{ea_g:.3f}  B={pb_g:.3f}+/-{eb_g:.3f}  "
          f"delta={delta_g:+.3f}+/-{1.96*delta_se_g:.3f}  "
          f"({total_games} games per model, wall={elapsed:.1f}s)")

    # Verdict at the 95% level on the global delta.
    if abs(delta_g) < 1.96 * delta_se_g:
        print("Verdict: no statistically significant difference at the 95% level on this panel.")
    elif delta_g > 0:
        print("Verdict: model B beats model A on this panel at the 95% level.")
    else:
        print("Verdict: model B loses to model A on this panel at the 95% level.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
