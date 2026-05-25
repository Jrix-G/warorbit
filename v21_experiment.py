"""Small experiment ledger for V21 gates and artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import v21_gate


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("experiment config must be a JSON object")
    return data


def build_record(name: str, *, before_log: str | None = None, after_log: str | None = None, artifacts: dict[str, str] | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"name": str(name), "artifacts": dict(artifacts or {})}
    if before_log:
        record["before"] = v21_gate.evaluate_log(before_log)
    if after_log:
        record["after"] = v21_gate.evaluate_log(after_log)
    record["promotable"] = bool(record.get("after", {}).get("passed", False))
    return record


def write_record(path: str | Path, record: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")


def _cmd_smoke(args: argparse.Namespace) -> dict[str, Any]:
    record = build_record("smoke", artifacts={"ranker": "analysis/v21_ranker_smoke.npz"})
    if args.out:
        write_record(args.out, record)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V21 experiment ledger")
    sub = parser.add_subparsers(dest="cmd", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--out", default="")
    rec = sub.add_parser("record")
    rec.add_argument("--name", required=True)
    rec.add_argument("--before-log", default="")
    rec.add_argument("--after-log", default="")
    rec.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "smoke":
        report = _cmd_smoke(args)
    else:
        report = build_record(args.name, before_log=args.before_log or None, after_log=args.after_log or None)
        write_record(args.out, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
