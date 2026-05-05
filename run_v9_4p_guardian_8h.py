#!/usr/bin/env python3
"""Launch the V9 4-player-only guardian training run with sane defaults."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TRAIN_OPPONENTS = [
    "random",
    "noisy_greedy",
    "starter",
    "distance",
    "sun_dodge",
    "structured",
    "orbit_stars",
    "bot_v7",
    "notebook_tactical_heuristic",
    "notebook_mdmahfuzsumon_how_my_ai_wins_space_wars",
    "notebook_sigmaborov_orbit_wars_2026_starter",
    "notebook_sigmaborov_orbit_wars_2026_tactical_heuristic",
    "notebook_orbitbotnext",
    "notebook_distance_prioritized",
    "notebook_physics_accurate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V9 guardian training in 4p-only mode.")
    parser.add_argument("--fresh", action="store_true", help="Do not resume from the latest checkpoint.")
    parser.add_argument("--minutes", type=float, default=480.0, help="Training budget in minutes.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes.")
    parser.add_argument("--pairs", type=int, default=6, help="Evolution-strategy perturbation pairs per generation.")
    parser.add_argument("--dry-run", action="store_true", help="Print the expanded command without running it.")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    eval_dir = Path("VPS") / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "run_v9.py",
        "--minutes",
        str(args.minutes),
        "--hard-timeout-minutes",
        str(args.minutes),
        "--workers",
        str(args.workers),
        "--pairs",
        str(args.pairs),
        "--games-per-eval",
        "3",
        "--eval-games",
        "12",
        "--benchmark-games",
        "24",
        "--min-promotion-benchmark-games",
        "24",
        "--benchmark-progress-every",
        "4",
        "--eval-every",
        "1",
        "--benchmark-every",
        "1",
        "--game-engine",
        "official_fast",
        "--max-steps",
        "120",
        "--eval-max-steps",
        "220",
        "--four-player-ratio",
        "1.0",
        "--eval-four-player-ratio",
        "1.0",
        "--benchmark-four-player-ratio",
        "1.0",
        "--four-p-signal-boost",
        "1.4",
        "--train-search-width",
        "3",
        "--train-simulation-depth",
        "0",
        "--train-simulation-rollouts",
        "0",
        "--train-opponent-samples",
        "1",
        "--front-lock-turns",
        "22",
        "--target-active-fronts",
        "2.0",
        "--target-backbone-turn-frac",
        "0.15",
        "--front-penalty-weight",
        "0.055",
        "--front-penalty-cap",
        "0.12",
        "--front-ok-bonus",
        "0.070",
        "--front-partial-bonus",
        "0.035",
        "--backbone-penalty-weight",
        "0.120",
        "--backbone-bonus-weight",
        "0.100",
        "--front-pressure-plan-bias",
        "0.16",
        "--front-pressure-attack-penalty",
        "0.14",
        "--guardian-enabled",
        "1",
        "--guardian-min-benchmark-4p",
        "0.42",
        "--guardian-min-benchmark-backbone",
        "0.08",
        "--guardian-max-benchmark-fronts",
        "2.70",
        "--guardian-max-generalization-gap",
        "0.18",
        "--export-best-on-finish",
        "1",
        "--min-benchmark-score",
        "0.35",
        "--max-generalization-gap",
        "0.18",
        "--exploration-rate",
        "0.08",
        "--reward-noise",
        "0.008",
        "--pool-limit",
        "15",
        "--checkpoint",
        str(eval_dir / "v9_4p_guardian_8h_latest.npz"),
        "--best-checkpoint",
        str(eval_dir / "v9_4p_guardian_8h_best.npz"),
        "--export-checkpoint",
        str(eval_dir / "v9_4p_guardian_8h_policy.npz"),
        "--log-jsonl",
        str(eval_dir / "v9_4p_guardian_8h_train.jsonl"),
        "--train-opponents",
        *TRAIN_OPPONENTS,
    ]
    if args.fresh:
        cmd.append("--no-resume")
    return cmd


def main() -> int:
    args = parse_args()
    cmd = build_command(args)
    print("[run_v9_4p_guardian_8h] launching 4p-only guardian run", flush=True)
    print("[run_v9_4p_guardian_8h] jsonl=VPS/evaluations/v9_4p_guardian_8h_train.jsonl", flush=True)
    if args.dry_run:
        print(" ".join(cmd))
        return 0
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
