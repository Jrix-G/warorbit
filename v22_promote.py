"""Promotion gate for V22 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import v21_gate
import v22_dataset


DEFAULT_RULES = {
    "min_samples": 200,
    "min_games_total": 16,
    "min_games_4p": 16,
    "min_wr_4p": 0.80,
    "min_ci_low_4p": 0.50,
    "min_avg_margin_4p": 0.0,
    "max_agent_errors": 0,
    "require_workers1": True,
}


def evaluate(dataset_path: str | Path, bench_log: str | Path, rules: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_RULES)
    if rules:
        cfg.update(rules)
    failures: list[str] = []
    samples = v22_dataset.load_jsonl(dataset_path)
    if len(samples) < int(cfg["min_samples"]):
        failures.append("min_samples")
    bench = v21_gate.summarize_log(bench_log)
    gate = v21_gate.evaluate_gate(
        bench,
        rules={
            "min_games_total": int(cfg["min_games_total"]),
            "min_games_4p": int(cfg["min_games_4p"]),
            "min_wr_4p": float(cfg["min_wr_4p"]),
            "min_ci_low_4p": float(cfg["min_ci_low_4p"]),
            "min_avg_margin_4p": float(cfg["min_avg_margin_4p"]),
            "max_agent_errors": int(cfg["max_agent_errors"]),
            "min_wr_2p": 0.0,
        },
    )
    failures.extend(f"bench_{name}" for name in gate["failures"])
    meta = _bench_meta(bench_log)
    if bool(cfg["require_workers1"]) and int(meta.get("workers", 1) or 1) != 1:
        failures.append("workers1_required")
    return {
        "passed": not failures,
        "failures": failures,
        "rules": cfg,
        "samples": len(samples),
        "bench": gate["summary"],
        "meta": meta,
    }


def _bench_meta(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") == "meta":
                return row
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V22 promotion gate")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bench-log", required=True)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_RULES["min_samples"])
    parser.add_argument("--min-games-4p", type=int, default=DEFAULT_RULES["min_games_4p"])
    parser.add_argument("--min-wr-4p", type=float, default=DEFAULT_RULES["min_wr_4p"])
    parser.add_argument("--min-ci-low-4p", type=float, default=DEFAULT_RULES["min_ci_low_4p"])
    parser.add_argument("--min-avg-margin-4p", type=float, default=DEFAULT_RULES["min_avg_margin_4p"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        args.dataset,
        args.bench_log,
        rules={
            "min_samples": args.min_samples,
            "min_games_total": args.min_games_4p,
            "min_games_4p": args.min_games_4p,
            "min_wr_4p": args.min_wr_4p,
            "min_ci_low_4p": args.min_ci_low_4p,
            "min_avg_margin_4p": args.min_avg_margin_4p,
        },
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
