"""V21 gate evaluation over benchmark JSONL logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import v21_metrics


DEFAULT_RULES = {
    "min_games_total": 16,
    "min_games_4p": 8,
    "min_wr_2p": 0.55,
    "min_wr_4p": 0.45,
    "min_ci_low_4p": 0.20,
    "min_avg_margin_4p": 0.0,
    "max_agent_errors": 0,
}


def load_bench_log(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if row.get("type") == "game":
                rows.append(row)
    return rows


def summarize_log(path: str | Path) -> dict[str, Any]:
    rows = load_bench_log(path)
    by_mode: dict[str, list[dict[str, Any]]] = {}
    total_errors = 0
    for row in rows:
        mode = str(row.get("mode") or f"{row.get('n_players', 0)}p")
        by_mode.setdefault(mode, []).append(_metrics_row(row))
        total_errors += int(row.get("candidate_errors", 0) or 0)
    modes: dict[str, Any] = {}
    for mode, mode_rows in sorted(by_mode.items()):
        summary = v21_metrics.summarize_games(mode_rows, bootstrap_iters=200)
        modes[mode] = _summary_to_dict(summary)
    return {"path": str(path), "games": len(rows), "candidate_errors": total_errors, "modes": modes}


def evaluate_gate(summary: dict[str, Any], rules: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_RULES)
    if rules:
        cfg.update(rules)
    failures: list[str] = []
    if int(summary.get("games", 0)) < int(cfg["min_games_total"]):
        failures.append("min_games_total")
    if int(summary.get("candidate_errors", 0)) > int(cfg["max_agent_errors"]):
        failures.append("candidate_errors")

    modes = summary.get("modes", {})
    m2 = modes.get("2p", {})
    m4 = modes.get("4p", {})
    if m2 and float(m2.get("winrate", 0.0)) < float(cfg["min_wr_2p"]):
        failures.append("min_wr_2p")
    if not m4 or int(m4.get("total", 0)) < int(cfg["min_games_4p"]):
        failures.append("min_games_4p")
    elif float(m4.get("winrate", 0.0)) < float(cfg["min_wr_4p"]):
        failures.append("min_wr_4p")
    if m4 and float(m4.get("ci_low", 0.0)) < float(cfg["min_ci_low_4p"]):
        failures.append("min_ci_low_4p")
    if m4 and float(m4.get("avg_margin", 0.0)) < float(cfg["min_avg_margin_4p"]):
        failures.append("min_avg_margin_4p")
    return {"passed": not failures, "failures": failures, "rules": cfg, "summary": summary}


def evaluate_log(path: str | Path, rules: dict[str, Any] | None = None) -> dict[str, Any]:
    return evaluate_gate(summarize_log(path), rules=rules)


def _metrics_row(row: dict[str, Any]) -> dict[str, Any]:
    scores = row.get("all_scores")
    player = int(row.get("our_seat", row.get("player", 0)) or 0)
    if isinstance(scores, list) and len(scores) >= 2:
        out = v21_metrics.game_row(scores, player, seat=player, n_players=int(row.get("n_players", len(scores)) or len(scores)))
    else:
        out = {
            "win": int(row.get("win", 0) or 0),
            "tie_score": float(row.get("win", 0) or 0),
            "rank": 1.0 if int(row.get("win", 0) or 0) else 2.0,
            "score_margin": float(row.get("score_margin", 0.0) or 0.0),
            "our_seat": player,
        }
    return out


def _summary_to_dict(summary: v21_metrics.GameSummary) -> dict[str, Any]:
    return {
        "total": summary.total,
        "wins": summary.wins,
        "winrate": summary.winrate.rate,
        "ci_low": summary.winrate.low,
        "ci_high": summary.winrate.high,
        "tie_score_mean": summary.tie_score_mean,
        "mean_rank": summary.mean_rank,
        "avg_margin": summary.avg_margin,
        "seat_bias_max_abs": max((abs(seat.bias) for seat in summary.seats), default=0.0),
    }


def _cmd_smoke() -> dict[str, Any]:
    summary = {
        "path": "smoke",
        "games": 24,
        "candidate_errors": 0,
        "modes": {
            "2p": {"total": 12, "winrate": 0.75, "ci_low": 0.45},
            "4p": {"total": 12, "winrate": 0.50, "ci_low": 0.25},
        },
    }
    return evaluate_gate(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V21 gate logs")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate-log")
    ev.add_argument("--log", required=True)
    ev.add_argument("--min-wr-2p", type=float, default=DEFAULT_RULES["min_wr_2p"])
    ev.add_argument("--min-wr-4p", type=float, default=DEFAULT_RULES["min_wr_4p"])
    ev.add_argument("--min-ci-low-4p", type=float, default=DEFAULT_RULES["min_ci_low_4p"])
    ev.add_argument("--min-avg-margin-4p", type=float, default=DEFAULT_RULES["min_avg_margin_4p"])
    sub.add_parser("smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "evaluate-log":
        report = evaluate_log(
            args.log,
            rules={
                "min_wr_2p": args.min_wr_2p,
                "min_wr_4p": args.min_wr_4p,
                "min_ci_low_4p": args.min_ci_low_4p,
                "min_avg_margin_4p": args.min_avg_margin_4p,
            },
        )
    else:
        report = _cmd_smoke()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
