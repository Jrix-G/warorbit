from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


Number = int | float


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _num(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _avg(rows: Iterable[dict[str, Any]], key: str, default: float = 0.0) -> float:
    values = [_num(row, key, default) for row in rows if key in row]
    return float(mean(values)) if values else default


def _avg_optional(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [_num(row, key) for row in rows if key in row]
    return float(mean(values)) if values else None


def _sum(rows: Iterable[dict[str, Any]], key: str) -> float:
    return float(sum(_num(row, key) for row in rows if key in row))


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            bool(row.get("checkpoint_promoted", False)),
            _num(row, "winrate"),
            -_num(row, "rank_mean", 99.0),
            _num(row, "score", _num(row, "composite_score")),
        ),
    )


def _group(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    return dict(grouped)


def _position_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    rank_buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_pos = row.get("winrate_by_position")
        if isinstance(by_pos, dict):
            for pos, winrate in by_pos.items():
                try:
                    buckets[str(pos)].append(float(winrate))
                except (TypeError, ValueError):
                    pass
        last = row.get("worker_last_record")
        if isinstance(last, dict) and "our_index" in last:
            pos = f"p{int(last.get('our_index'))}"
            rank_buckets[pos].append(_num(last, "rank", 0.0))
    positions = sorted(set(buckets) | set(rank_buckets))
    return {
        pos: {
            "mean_eval_winrate": float(mean(buckets[pos])) if buckets[pos] else 0.0,
            "mean_last_train_rank": float(mean(rank_buckets[pos])) if rank_buckets[pos] else 0.0,
            "last_train_samples": float(len(rank_buckets[pos])),
        }
        for pos in positions
    }


def _opponent_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    winrates: dict[str, list[float]] = defaultdict(list)
    ranks: dict[str, list[float]] = defaultdict(list)
    games: Counter[str] = Counter()
    for row in rows:
        by_opp = row.get("eval_by_opponent")
        if not isinstance(by_opp, dict):
            continue
        for name, data in by_opp.items():
            if not isinstance(data, dict):
                continue
            try:
                winrates[str(name)].append(float(data.get("winrate", 0.0)))
                ranks[str(name)].append(float(data.get("rank_mean", 0.0)))
                games[str(name)] += int(data.get("games", 0))
            except (TypeError, ValueError):
                continue
    return {
        name: {
            "mean_winrate": float(mean(winrates[name])) if winrates[name] else 0.0,
            "mean_rank": float(mean(ranks[name])) if ranks[name] else 0.0,
            "games_reported": float(games[name]),
        }
        for name in sorted(winrates)
    }


def _winner_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    winners: Counter[str] = Counter()
    total = 0
    for row in rows:
        last = row.get("worker_last_record")
        if isinstance(last, dict) and "winner" in last:
            winners[str(int(last.get("winner")))] += 1
            total += 1
    if not total:
        return {}
    return {winner: count / total for winner, count in sorted(winners.items())}


def _counter_rates(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in sorted(counter.items())}


def _mission_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for row in rows:
        mission_counts = row.get("mission_counts")
        if isinstance(mission_counts, dict):
            for mission, value in mission_counts.items():
                try:
                    counts[str(mission)] += int(value)
                except (TypeError, ValueError):
                    pass
            continue
        last = row.get("worker_last_record")
        if isinstance(last, dict):
            for key, mission in (
                ("mission_expand_count", "expand"),
                ("mission_attack_count", "attack"),
                ("mission_support_count", "support"),
                ("mission_do_nothing_count", "do_nothing"),
            ):
                if key in last:
                    counts[mission] += int(_num(last, key))
    return _counter_rates(counts)


def _final_cause_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for row in rows:
        causes = row.get("eval_final_cause_counts")
        if isinstance(causes, dict):
            for cause, value in causes.items():
                try:
                    counts[str(cause)] += int(value)
                except (TypeError, ValueError):
                    pass
        elif "final_cause" in row:
            counts[str(row.get("final_cause"))] += 1
    return _counter_rates(counts)


def _rows_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    best = _best_row(rows)
    return {
        "rows": len(rows),
        "promotions": int(sum(1 for row in rows if row.get("checkpoint_promoted"))),
        "mean_winrate": _avg(rows, "winrate"),
        "best_winrate": max(_num(row, "winrate") for row in rows),
        "mean_rank": _avg(rows, "rank_mean"),
        "mean_avg_score": _avg(rows, "avg_score"),
        "mean_eval_do_nothing_rate": _avg(rows, "eval_do_nothing_rate"),
        "mean_eval_avg_ships_sent": _avg(rows, "eval_avg_ships_sent"),
        "mean_eval_elimination_rate": _avg_optional(rows, "eval_elimination_rate"),
        "mean_eval_score_margin_to_winner": _avg_optional(rows, "eval_score_margin_to_winner"),
        "mean_eval_rank_reward": _avg_optional(rows, "eval_rank_reward_mean"),
        "mean_eval_final_strength": _avg_optional(rows, "eval_final_strength_mean"),
        "mean_train_winrate": _avg(rows, "train_winrate"),
        "mean_train_rank": _avg(rows, "train_rank_mean"),
        "mean_train_do_nothing_rate": _avg(rows, "train_do_nothing_rate"),
        "mean_train_avg_ships_sent": _avg(rows, "train_avg_ships_sent"),
        "train_games_completed": int(_sum(rows, "train_games_completed")),
        "mission_rates": _mission_summary(rows),
        "final_cause_rates": _final_cause_summary(rows),
        "best_row": {
            "stage": best.get("stage"),
            "generation": best.get("generation"),
            "worker_id": best.get("worker_id"),
            "winrate": _num(best, "winrate"),
            "rank_mean": _num(best, "rank_mean"),
            "avg_score": _num(best, "avg_score"),
            "eval_do_nothing_rate": _num(best, "eval_do_nothing_rate"),
            "eval_avg_ships_sent": _num(best, "eval_avg_ships_sent"),
            "checkpoint_promoted": bool(best.get("checkpoint_promoted", False)),
            "promotion_reason": best.get("promotion_reason", ""),
        },
    }


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage_groups = _group(rows, "stage")
    generation_groups = _group(rows, "stage", "generation")
    by_stage = {str(stage[0]): _rows_summary(items) for stage, items in sorted(stage_groups.items(), key=lambda item: str(item[0]))}
    by_generation = {
        f"{stage}:g{generation}": _rows_summary(items)
        for (stage, generation), items in sorted(generation_groups.items(), key=lambda item: (str(item[0][0]), int(item[0][1] or -1)))
    }
    has_missions = any(isinstance(row.get("mission_counts"), dict) for row in rows) or any(
        isinstance(row.get("worker_last_record"), dict) and "mission_counts" in row["worker_last_record"] for row in rows
    )
    has_final_cause = any("final_cause" in row or isinstance(row.get("eval_final_cause_counts"), dict) for row in rows)
    has_snapshots = any(isinstance(row.get("final_player_snapshots"), dict) for row in rows)
    missing = ["per-turn action logits and selected candidate mission"]
    if not has_missions:
        missing.append("mission_counts per evaluated episode: expand/attack/support/do_nothing")
    if not has_final_cause:
        missing.append("final cause flags: eliminated, timeout_survived, score_loss, winner_id")
    if not has_snapshots:
        missing.append("final player snapshot: planets, production, ship_share, score for every player")
    return {
        "overall": _rows_summary(rows),
        "by_stage": by_stage,
        "by_generation": by_generation,
        "by_position": _position_summary(rows),
        "by_opponent": _opponent_summary(rows),
        "last_train_winner_distribution": _winner_summary(rows),
        "missing_instrumentation": missing,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    print(f"\n## {title}")
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        print("| " + " | ".join(_fmt(row.get(col, "")) for col in columns) + " |")


def print_markdown(report: dict[str, Any]) -> None:
    print("# Run JSONL Analysis")
    overall = dict(report.get("overall", {}))
    best = overall.pop("best_row", {})
    mission_rates = overall.pop("mission_rates", {})
    cause_rates = overall.pop("final_cause_rates", {})
    for key, value in overall.items():
        if value is None:
            continue
        print(f"- {key}: {_fmt(value)}")
    if best:
        print("\n## Best Row")
        for key, value in best.items():
            print(f"- {key}: {_fmt(value)}")

    if mission_rates:
        print("\n## Mission Rates")
        for mission, rate in mission_rates.items():
            print(f"- {mission}: {_fmt(rate)}")

    if cause_rates:
        print("\n## Final Cause Rates")
        for cause, rate in cause_rates.items():
            print(f"- {cause}: {_fmt(rate)}")

    generation_rows = []
    for name, data in report.get("by_generation", {}).items():
        best_row = data.get("best_row", {})
        generation_rows.append(
            {
                "bucket": name,
                "rows": data.get("rows"),
                "best_winrate": data.get("best_winrate"),
                "mean_winrate": data.get("mean_winrate"),
                "mean_rank": data.get("mean_rank"),
                "mean_avg_score": data.get("mean_avg_score"),
                "mean_eval_do_nothing_rate": data.get("mean_eval_do_nothing_rate"),
                "mean_eval_elimination_rate": data.get("mean_eval_elimination_rate"),
                "mean_eval_rank_reward": data.get("mean_eval_rank_reward"),
                "mean_eval_final_strength": data.get("mean_eval_final_strength"),
                "mean_train_winrate": data.get("mean_train_winrate"),
                "promoted_worker": best_row.get("worker_id") if best_row.get("checkpoint_promoted") else "",
            }
        )
    _print_table(
        "Generation Progression",
        generation_rows,
        [
            "bucket",
            "rows",
            "best_winrate",
            "mean_winrate",
            "mean_rank",
            "mean_avg_score",
            "mean_eval_do_nothing_rate",
            "mean_eval_elimination_rate",
            "mean_eval_rank_reward",
            "mean_eval_final_strength",
            "mean_train_winrate",
            "promoted_worker",
        ],
    )

    position_rows = [{"position": key, **value} for key, value in report.get("by_position", {}).items()]
    _print_table("Position Sensitivity", position_rows, ["position", "mean_eval_winrate", "mean_last_train_rank", "last_train_samples"])

    opponent_rows = [{"opponent": key, **value} for key, value in report.get("by_opponent", {}).items()]
    _print_table("Opponent Sensitivity", opponent_rows, ["opponent", "mean_winrate", "mean_rank", "games_reported"])

    winner_dist = report.get("last_train_winner_distribution", {})
    if winner_dist:
        print("\n## Last Train Winner Distribution")
        for winner, rate in winner_dist.items():
            print(f"- winner {winner}: {_fmt(rate)}")

    print("\n## Missing Instrumentation")
    for item in report.get("missing_instrumentation", []):
        print(f"- {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a neural_network training.jsonl run file.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()
    report = analyze(_load_jsonl(args.jsonl))
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_markdown(report)


if __name__ == "__main__":
    main()
