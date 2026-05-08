#!/usr/bin/env python3
"""Promotion gate for V14 checkpoints.

The gate is deliberately simple: a checkpoint is only promotable if it is not
regressing against the best known heuristic baseline on both 2p and 4p.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_v14 import DEFAULT_OPPONENTS, MatchStats, run_suite


def _stats_dict(stats: MatchStats) -> dict:
    return {
        "wins": stats.wins,
        "losses": stats.losses,
        "draws": stats.draws,
        "games": stats.games,
        "win_rate": stats.win_rate,
        "seconds": stats.seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v14-weights", default="evaluations/scorer_v14.npz")
    parser.add_argument("--v13-weights", default="evaluations/scorer_v13_2h.best.npz")
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--seed-offset", type=int, default=714000)
    parser.add_argument("--allowed-drop", type=float, default=0.03)
    parser.add_argument("--min-avg-delta", type=float, default=0.0)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--opponents", nargs="*", default=DEFAULT_OPPONENTS)
    args = parser.parse_args()

    bots = ("v7", "v12", "v14")
    modes = (("4p", 4), ("2p", 2))
    results: dict[str, dict[str, dict]] = {}

    print(
        f"V14 promotion gate | games={args.games} workers={args.workers} "
        f"max_steps={args.max_steps} allowed_drop={args.allowed_drop:.3f}"
    )
    for mode_i, (mode_name, n_players) in enumerate(modes):
        print(f"\nMode {mode_name}")
        results[mode_name] = {}
        for bot_i, bot_name in enumerate(bots):
            stats = run_suite(
                bot_name,
                args.opponents,
                games=args.games,
                n_players=n_players,
                seed_offset=args.seed_offset + mode_i * 10000 + bot_i * 100000,
                workers=max(1, args.workers),
                max_steps=args.max_steps,
                v13_weights=args.v13_weights,
                v14_weights=args.v14_weights,
            )
            results[mode_name][bot_name] = _stats_dict(stats)
            print(
                f"- {bot_name:4s} W/L/D={stats.wins}/{stats.losses}/{stats.draws} "
                f"WR={stats.win_rate:.3f} seconds={stats.seconds:.1f}",
                flush=True,
            )

    failures: list[str] = []
    avg_v14 = 0.0
    avg_baseline = 0.0
    for mode_name, _ in modes:
        v14_wr = float(results[mode_name]["v14"]["win_rate"])
        baseline_wr = max(
            float(results[mode_name]["v7"]["win_rate"]),
            float(results[mode_name]["v12"]["win_rate"]),
        )
        avg_v14 += v14_wr
        avg_baseline += baseline_wr
        if v14_wr + args.allowed_drop < baseline_wr:
            failures.append(
                f"{mode_name}: v14 {v14_wr:.3f} < baseline {baseline_wr:.3f} "
                f"- allowed_drop {args.allowed_drop:.3f}"
            )
    avg_v14 /= len(modes)
    avg_baseline /= len(modes)
    if avg_v14 < avg_baseline + args.min_avg_delta:
        failures.append(
            f"avg: v14 {avg_v14:.3f} < baseline {avg_baseline:.3f} "
            f"+ min_avg_delta {args.min_avg_delta:.3f}"
        )

    summary = {
        "passed": not failures,
        "failures": failures,
        "avg_v14": avg_v14,
        "avg_baseline": avg_baseline,
        "results": results,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nGate:", "PASS" if summary["passed"] else "FAIL")
    for failure in failures:
        print(f"- {failure}")
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
