#!/usr/bin/env python3
"""Compare V20 bench JSONL runs by seed/seat.

Usage:
    python analysis_v20_compare_runs.py analysis/v20_4p_*.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


Key = tuple[str, int, int]


@dataclass(frozen=True)
class GameRow:
    source: str
    key: Key
    win: int
    margin: float
    intents: dict[str, int]


def _short_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            matches = [pattern]
        for match in matches:
            path = Path(match)
            if path.is_file():
                resolved = path.resolve()
                if resolved not in seen:
                    paths.append(path)
                    seen.add(resolved)
    return paths


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_games(path: Path) -> tuple[list[GameRow], int, int]:
    rows: list[GameRow] = []
    bad_json = 0
    skipped = 0
    source = _short_path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            if not isinstance(data, dict) or data.get("type") != "game":
                continue
            if "seed" not in data or "our_seat" not in data:
                skipped += 1
                continue

            mode = str(data.get("mode") or f"{_as_int(data.get('n_players'))}p")
            seed = _as_int(data.get("seed"))
            seat = _as_int(data.get("our_seat"))
            key = (mode, seed, seat)
            intents_raw = data.get("intent_counts", {})
            intents: dict[str, int] = {}
            if isinstance(intents_raw, dict):
                intents = {
                    str(name): _as_int(count)
                    for name, count in intents_raw.items()
                    if _as_int(count) != 0
                }
            rows.append(
                GameRow(
                    source=source,
                    key=key,
                    win=1 if _as_int(data.get("win")) else 0,
                    margin=_as_float(data.get("score_margin")),
                    intents=intents,
                )
            )
    return rows, bad_json, skipped


def _format_key(key: Key) -> str:
    mode, seed, seat = key
    return f"{mode} seed={seed} seat={seat}"


def _format_margin(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _print_files(files: list[Path], games_by_source: dict[str, list[GameRow]]) -> None:
    print("FILES")
    for index, path in enumerate(files, 1):
        source = _short_path(path)
        rows = games_by_source.get(source, [])
        wins = sum(row.win for row in rows)
        total = len(rows)
        wr = wins / total if total else 0.0
        print(f"  [{index}] {source}: wins={wins}/{total} wr={wr:.3f}")
    print()


def _print_union_wins(games_by_key: dict[Key, list[GameRow]]) -> None:
    winning_keys = [key for key, rows in games_by_key.items() if any(row.win for row in rows)]
    print("UNION WINS")
    print(f"  keys={len(winning_keys)}/{len(games_by_key)}")
    for key in sorted(winning_keys):
        winners = [row.source for row in games_by_key[key] if row.win]
        print(f"  {_format_key(key)}: {', '.join(winners)}")
    print()


def _print_wins_by_file(games_by_source: dict[str, list[GameRow]]) -> None:
    print("WINS BY FILE")
    for source in sorted(games_by_source):
        won = [row.key for row in games_by_source[source] if row.win]
        print(f"  {source}: {len(won)} wins")
        for key in sorted(won):
            print(f"    {_format_key(key)}")
    print()


def _print_best_margins(games_by_key: dict[Key, list[GameRow]]) -> None:
    print("BEST MARGIN BY KEY")
    for key in sorted(games_by_key):
        rows = games_by_key[key]
        best_margin = max(row.margin for row in rows)
        best = [row.source for row in rows if row.margin == best_margin]
        all_bits = ", ".join(
            f"{row.source}={_format_margin(row.margin)}" for row in sorted(rows, key=lambda item: item.source)
        )
        print(f"  {_format_key(key)}: best={_format_margin(best_margin)} by {', '.join(best)} ({all_bits})")
    print()


def _intent_totals(rows: list[GameRow]) -> tuple[Counter[str], Counter[str]]:
    win_counts: Counter[str] = Counter()
    loss_counts: Counter[str] = Counter()
    for row in rows:
        target = win_counts if row.win else loss_counts
        target.update(row.intents)
    return win_counts, loss_counts


def _print_intents(all_rows: list[GameRow]) -> None:
    win_counts, loss_counts = _intent_totals(all_rows)
    names = sorted(set(win_counts) | set(loss_counts))
    print("INTENT COUNTERS WINS/LOSSES")
    print("  intent wins losses delta")
    for name in names:
        wins = win_counts[name]
        losses = loss_counts[name]
        print(f"  {name} {wins} {losses} {wins - losses:+d}")
    print()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="V20 JSONL log paths or shell-style globs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    files = _expand_paths(args.logs)
    if not files:
        print("No input files matched.", file=sys.stderr)
        return 2

    all_rows: list[GameRow] = []
    games_by_source: dict[str, list[GameRow]] = {}
    warnings: list[str] = []
    for path in files:
        rows, bad_json, skipped = _load_games(path)
        source = _short_path(path)
        games_by_source[source] = rows
        all_rows.extend(rows)
        if bad_json:
            warnings.append(f"{source}: {bad_json} malformed JSON lines")
        if skipped:
            warnings.append(f"{source}: {skipped} game rows missing seed/seat")

    games_by_key: dict[Key, list[GameRow]] = {}
    for row in all_rows:
        games_by_key.setdefault(row.key, []).append(row)

    _print_files(files, games_by_source)
    _print_union_wins(games_by_key)
    _print_wins_by_file(games_by_source)
    _print_best_margins(games_by_key)
    _print_intents(all_rows)

    if warnings:
        print("WARNINGS", file=sys.stderr)
        for warning in warnings:
            print(f"  {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
