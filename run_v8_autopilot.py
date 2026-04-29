#!/usr/bin/env python3
"""Autonomous overnight controller for V8 offline training.

The controller runs short probes first. A long run is allowed only when a
probe beats the minimum validation score. This prevents another unattended
night on a visibly bad proxy.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ProbeConfig:
    name: str
    min_gap: float
    oracle_horizon: int
    target_examples: int
    max_snapshots: int


CONFIGS = [
    ProbeConfig("strict60", min_gap=0.015, oracle_horizon=60, target_examples=160, max_snapshots=1000),
    ProbeConfig("medium80", min_gap=0.010, oracle_horizon=80, target_examples=192, max_snapshots=1400),
    ProbeConfig("wide100", min_gap=0.0075, oracle_horizon=100, target_examples=224, max_snapshots=1800),
]


def _run_logged(cmd: List[str], log_path: str) -> int:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return int(proc.wait())


def _last_score(log_path: str) -> Optional[float]:
    if not os.path.exists(log_path):
        return None
    text = open(log_path, "r", encoding="utf-8", errors="replace").read()
    matches = re.findall(r"score\s*:\s*([+-]?\d+(?:\.\d+)?)", text)
    if not matches:
        return None
    return float(matches[-1])


def _last_min(log_path: str) -> Optional[float]:
    if not os.path.exists(log_path):
        return None
    text = open(log_path, "r", encoding="utf-8", errors="replace").read()
    matches = re.findall(r"min\s*:\s*([+-]?\d+(?:\.\d+)?)", text)
    if not matches:
        return None
    return float(matches[-1])


def _remove_artifacts(prefix: str) -> None:
    for suffix in [
        ".npz",
        "_best.npz",
        "_offline_state.pkl",
        "_offline_dataset.pkl",
    ]:
        path = f"{prefix}{suffix}"
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def _train_cmd(args, cfg: ProbeConfig, hours: float, out_path: str, dataset_path: str, state_path: str, refresh: bool) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
        "train_v8_offline.py",
        "--hours",
        f"{hours:.4f}",
        "--dataset-states",
        "48",
        "--target-examples",
        str(cfg.target_examples),
        "--max-snapshots",
        str(cfg.max_snapshots),
        "--oracle-horizon",
        str(cfg.oracle_horizon),
        "--min-gap",
        str(cfg.min_gap),
        "--benchmark-games",
        str(args.benchmark_games),
        "--benchmark-seconds",
        str(args.benchmark_seconds),
        "--save-seconds",
        str(args.save_seconds),
        "--skip-initial-benchmark",
        "--out",
        out_path,
        "--dataset-out",
        dataset_path,
        "--state-out",
        state_path,
    ]
    if refresh:
        cmd.append("--refresh-dataset")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--probe-hours", type=float, default=0.33)
    parser.add_argument("--benchmark-games", type=int, default=10)
    parser.add_argument("--benchmark-seconds", type=int, default=900)
    parser.add_argument("--save-seconds", type=int, default=900)
    parser.add_argument("--accept-score", type=float, default=0.24)
    parser.add_argument("--accept-min", type=float, default=0.05)
    args = parser.parse_args()

    started = time.time()
    chosen: Optional[ProbeConfig] = None
    chosen_prefix = os.path.join("evaluations", "v8_policy")
    chosen_dataset = os.path.join("evaluations", "v8_policy_offline_dataset.pkl")
    chosen_state = os.path.join("evaluations", "v8_policy_offline_state.pkl")

    for cfg in CONFIGS:
        prefix = os.path.join("evaluations", f"v8_policy_{cfg.name}")
        out_path = f"{prefix}.npz"
        dataset_path = f"{prefix}_offline_dataset.pkl"
        state_path = f"{prefix}_offline_state.pkl"
        log_path = f"training_autopilot_probe_{cfg.name}.log"
        _remove_artifacts(prefix)

        print(f"\n=== Probe {cfg.name} ===", flush=True)
        rc = _run_logged(
            _train_cmd(args, cfg, args.probe_hours, out_path, dataset_path, state_path, refresh=True),
            log_path,
        )
        score = _last_score(log_path)
        min_rate = _last_min(log_path)
        print(f"Probe {cfg.name}: rc={rc} score={score} min={min_rate}", flush=True)
        if rc == 0 and score is not None and min_rate is not None and score >= args.accept_score and min_rate >= args.accept_min:
            chosen = cfg
            chosen_prefix = prefix
            chosen_dataset = dataset_path
            chosen_state = state_path
            break

    if chosen is None:
        print("\nNo probe passed the acceptance gate. Overnight run aborted.", flush=True)
        return 2

    elapsed_h = (time.time() - started) / 3600.0
    remaining = max(0.05, float(args.hours) - elapsed_h)
    best_path = f"{chosen_prefix}_best.npz"
    out_path = os.path.join("evaluations", "v8_policy.npz")
    dataset_path = os.path.join("evaluations", "v8_policy_offline_dataset.pkl")
    state_path = os.path.join("evaluations", "v8_policy_offline_state.pkl")

    print(f"\n=== Long run from {chosen.name}: {remaining:.2f}h remaining ===", flush=True)
    cmd = _train_cmd(args, chosen, remaining, out_path, chosen_dataset, state_path, refresh=False)
    cmd.extend(["--resume", best_path, "--resume-state", chosen_state])
    return _run_logged(cmd, "training_autopilot_night.log")


if __name__ == "__main__":
    raise SystemExit(main())
