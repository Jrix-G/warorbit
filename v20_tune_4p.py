"""Short V20 4p env-config tuner against raw V15 via v20_bench.py.

This is intentionally a small orchestration layer, not a new benchmark.  It
runs a bounded set of V20 environment configurations through v20_bench.py,
stores every per-game log, and ranks the configs by the 4p summary.

Default run is short:
    python -u v20_tune_4p.py

Inspect without running:
    python v20_tune_4p.py --list-configs
    python v20_tune_4p.py --dry-run
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


@dataclass(frozen=True)
class TuneConfig:
    name: str
    env: dict[str, str]
    note: str = ""


BASE_CONFIGS: tuple[TuneConfig, ...] = (
    TuneConfig("baseline", {}, "current V20 defaults"),
    TuneConfig(
        "empty_returns_v15",
        {"V20_EMPTY_RETURNS_V15": "1"},
        "fallback to V15 when V20 finds no combo",
    ),
    TuneConfig(
        "bias035",
        {"V20_BIAS_WEIGHT": "0.035"},
        "slightly stronger macro bias",
    ),
    TuneConfig(
        "bias045",
        {"V20_BIAS_WEIGHT": "0.045"},
        "stronger macro bias",
    ),
    TuneConfig(
        "staging070",
        {"V20_STAGING_BIAS_WEIGHT": "0.070"},
        "more support for staging/consolidation moves",
    ),
    TuneConfig(
        "rank015",
        {"V20_RANK_VALUE_WEIGHT": "0.015"},
        "small 4p rank-aware value bonus",
    ),
    TuneConfig(
        "no_pressure_leader",
        {"V20_DISABLE_PRESSURE_LEADER": "1"},
        "avoid leader pressure intent",
    ),
    TuneConfig(
        "strict_safe_bias",
        {
            "V20_BIAS_WEIGHT": "0.035",
            "V20_MAX_ESC_LOSS": "0.002",
            "V20_MIN_OBJECTIVE_GAIN": "0.0010",
        },
        "macro bias with tighter ESC loss guard",
    ),
)

OPTIONAL_CONFIGS: tuple[TuneConfig, ...] = (
    TuneConfig(
        "macro_candidates",
        {"V20_ENABLE_MACRO_CANDIDATES": "1"},
        "adds macro source-target pairs as candidates",
    ),
    TuneConfig(
        "top10_policy",
        {"V20_ENABLE_TOP10_POLICY": "1"},
        "enables optional top10 policy path if available",
    ),
    TuneConfig(
        "v15_search_only_macro_policy",
        {"V20_V15_SEARCH_ONLY": "1"},
        "vanilla V15 search with macro policy_fn",
    ),
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"type": "parse_error", "line": line[:500]})
    return rows


def _summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in reversed(rows):
        if row.get("type") == "summary":
            return row
    games = [row for row in rows if row.get("type") == "game"]
    wins = sum(int(row.get("win", 0)) for row in games)
    total = len(games)
    margins = [float(row.get("score_margin", 0.0)) for row in games]
    return {
        "type": "summary",
        "modes": {
            "4p": {
                "wins": wins,
                "total": total,
                "winrate": wins / total if total else 0.0,
                "avg_margin": sum(margins) / total if total else 0.0,
            }
        },
        "aggregate": {
            "wins": wins,
            "total": total,
            "winrate": wins / total if total else 0.0,
        },
    }


def _mode_summary(summary: dict[str, Any], mode: str = "4p") -> dict[str, Any]:
    modes = summary.get("modes", {})
    if isinstance(modes, dict) and isinstance(modes.get(mode), dict):
        return modes[mode]
    agg = summary.get("aggregate", {})
    return agg if isinstance(agg, dict) else {}


def _rank_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
    mode = _mode_summary(record.get("summary", {}))
    winrate = float(mode.get("winrate", 0.0))
    avg_margin = float(mode.get("avg_margin", 0.0))
    total = float(mode.get("total", 0.0))
    errors = float(record.get("candidate_errors", 0.0))
    error_penalty = errors / max(1.0, total)
    return (winrate, avg_margin, total, -error_penalty)


def _bench_command(args: argparse.Namespace, config: TuneConfig, log_path: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        "v20_bench.py",
        "--candidate",
        args.candidate,
        "--candidate-style",
        args.candidate_style,
        "--games",
        str(args.games),
        "--modes",
        "4",
        "--seat-rotation",
        args.seat_rotation,
        "--workers",
        str(args.workers),
        "--seed-offset",
        str(args.seed_offset),
        "--episode-steps",
        str(args.episode_steps),
        "--candidate-budget",
        str(args.candidate_budget),
        "--candidate-horizon",
        str(args.candidate_horizon),
        "--v15-budget",
        str(args.v15_budget),
        "--progress-every",
        str(args.progress_every),
        "--log",
        str(log_path),
    ]


def _run_one(args: argparse.Namespace, out_dir: Path, config: TuneConfig) -> dict[str, Any]:
    log_path = out_dir / f"{config.name}.jsonl"
    stdout_path = out_dir / f"{config.name}.stdout.txt"
    stderr_path = out_dir / f"{config.name}.stderr.txt"
    cmd = _bench_command(args, config, log_path)
    env = os.environ.copy()
    env.update(config.env)

    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parent,
        env=env,
        text=True,
        capture_output=True,
        timeout=args.timeout_sec,
        check=False,
    )
    elapsed = time.time() - started
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    rows = _jsonl_rows(log_path)
    games = [row for row in rows if row.get("type") == "game"]
    summary = _summary_from_rows(rows)
    candidate_errors = sum(int(row.get("candidate_errors", 0)) for row in games)
    record = {
        "type": "tune_result",
        "name": config.name,
        "note": config.note,
        "env": config.env,
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 3),
        "log": str(log_path),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "summary": summary,
        "candidate_errors": candidate_errors,
    }
    record["rank_key"] = list(_rank_key(record))
    return record


def _print_config(config: TuneConfig) -> None:
    env = " ".join(f"{k}={v}" for k, v in sorted(config.env.items())) or "(default env)"
    suffix = f" - {config.note}" if config.note else ""
    print(f"{config.name}: {env}{suffix}")


def _write_summaries(out_dir: Path, records: list[dict[str, Any]]) -> None:
    ranked = sorted(records, key=_rank_key, reverse=True)
    jsonl_path = out_dir / "sweep_results.jsonl"
    summary_path = out_dir / "summary.json"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    payload = {
        "type": "v20_tune_4p_summary",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "best": ranked[0] if ranked else None,
        "ranked": ranked,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="v20_agent:agent")
    ap.add_argument("--candidate-style", choices=("auto", "state", "obs"), default="obs")
    ap.add_argument("--games", type=int, default=4, help="short sweep: base seeds per config")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed-offset", type=int, default=40_000_000)
    ap.add_argument("--episode-steps", type=int, default=250)
    ap.add_argument("--seat-rotation", choices=("cycle", "full", "seed"), default="cycle")
    ap.add_argument("--candidate-budget", type=float, default=0.55)
    ap.add_argument("--candidate-horizon", type=int, default=24)
    ap.add_argument("--v15-budget", type=float, default=0.7)
    ap.add_argument("--progress-every", type=int, default=4)
    ap.add_argument("--timeout-sec", type=int, default=900)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--configs", default="", help="comma-separated config names; default uses base sweep")
    ap.add_argument("--include-optional", action="store_true")
    ap.add_argument("--list-configs", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    configs = list(BASE_CONFIGS)
    if args.include_optional:
        configs.extend(OPTIONAL_CONFIGS)

    by_name = {config.name: config for config in configs}
    if args.configs:
        selected: list[TuneConfig] = []
        for name in [part.strip() for part in args.configs.split(",") if part.strip()]:
            if name not in by_name:
                known = ", ".join(sorted(by_name))
                raise SystemExit(f"unknown config '{name}'. Known configs: {known}")
            selected.append(by_name[name])
        configs = selected

    if args.list_configs:
        for config in configs:
            _print_config(config)
        return 0

    out_dir = Path(args.out_dir or Path("analysis") / f"v20_tune_4p_{_timestamp()}")
    if args.dry_run:
        print(f"output dir: {out_dir}")
        for config in configs:
            log_path = out_dir / f"{config.name}.jsonl"
            env = " ".join(f"{k}={v}" for k, v in sorted(config.env.items()))
            print(f"\n[{config.name}] {env}".rstrip())
            print(" ".join(_bench_command(args, config, log_path)))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    print(
        f"V20 4p tune: configs={len(configs)} games={args.games} "
        f"seat_rotation={args.seat_rotation} workers={args.workers} out={out_dir}",
        flush=True,
    )
    for idx, config in enumerate(configs, start=1):
        print(f"\n[{idx}/{len(configs)}] {config.name}", flush=True)
        _print_config(config)
        try:
            record = _run_one(args, out_dir, config)
        except subprocess.TimeoutExpired as exc:
            record = {
                "type": "tune_result",
                "name": config.name,
                "note": config.note,
                "env": config.env,
                "returncode": 124,
                "elapsed_sec": args.timeout_sec,
                "log": str(out_dir / f"{config.name}.jsonl"),
                "stdout": "",
                "stderr": "",
                "summary": {"type": "summary", "modes": {}, "aggregate": {}},
                "candidate_errors": 0,
                "timeout": str(exc),
            }
            record["rank_key"] = list(_rank_key(record))
        records.append(record)
        mode = _mode_summary(record.get("summary", {}))
        print(
            f"  rc={record['returncode']} W={mode.get('wins', 0)}/{mode.get('total', 0)} "
            f"WR={float(mode.get('winrate', 0.0)):.3f} "
            f"margin={float(mode.get('avg_margin', 0.0)):.1f} "
            f"errors={record.get('candidate_errors', 0)}",
            flush=True,
        )
        _write_summaries(out_dir, records)

    ranked = sorted(records, key=_rank_key, reverse=True)
    best = ranked[0] if ranked else None
    print("\n=== V20 4p tune ranking ===", flush=True)
    for rank, record in enumerate(ranked, start=1):
        mode = _mode_summary(record.get("summary", {}))
        print(
            f"{rank:2d}. {record['name']}: W={mode.get('wins', 0)}/{mode.get('total', 0)} "
            f"WR={float(mode.get('winrate', 0.0)):.3f} "
            f"margin={float(mode.get('avg_margin', 0.0)):.1f} "
            f"rc={record['returncode']}",
            flush=True,
        )
    if best is not None:
        print(f"\nmost promising: {best['name']} env={best['env']}", flush=True)
    print(f"summary -> {out_dir / 'summary.json'}", flush=True)
    print(f"results -> {out_dir / 'sweep_results.jsonl'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
