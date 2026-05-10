#!/usr/bin/env python3
"""Summarize V14 fine-tune logs into comparable experiment rows."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


KEYS = (
    "wr2",
    "wr4",
    "reward",
    "postkl",
    "dlogit",
    "clip",
    "gn",
    "agn",
    "cgn",
    "adv",
    "ret",
    "bc",
    "lr",
    "T",
)


def _extract(line: str, key: str) -> str:
    match = re.search(rf"{re.escape(key)}=([^ ]+)", line)
    return match.group(1) if match else ""


def summarize(path: Path) -> dict[str, str]:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        if " b" not in line or "wr=" not in line:
            continue
        rows.append({key: _extract(line, key) for key in KEYS})
    if not rows:
        return {"file": path.name, "batches": "0"}
    result = {"file": path.name, "batches": str(len(rows))}
    result.update(rows[-1])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    rows = [summarize(path) for path in args.logs]
    columns = ("file", "batches", *KEYS)
    print("\t".join(columns))
    for row in rows:
        print("\t".join(row.get(col, "") for col in columns))


if __name__ == "__main__":
    main()
