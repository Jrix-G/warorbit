from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load_recent_jsonl(path: str | Path, window: int) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: List[Dict[str, Any]] = []
    for line in lines[-max(1, int(window)) :]:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float(default)


def _rank_from_record(record: Dict[str, Any]) -> float | None:
    if "rank" in record:
        return float(record["rank"])
    if "rank_mean" in record:
        return float(record["rank_mean"])
    return None


def analyze_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    window = len(records)
    entropy_mean = _mean((float(r.get("policy_entropy", 0.0)) for r in records), default=0.0)
    ranks = [rank for rank in (_rank_from_record(r) for r in records) if rank is not None]
    rank_mean = _mean(ranks, default=4.0)
    wins = [
        1.0 if int(r.get("winner", -1)) == int(r.get("our_index", -2)) else 0.0
        for r in records
        if "winner" in r and "our_index" in r
    ]
    winrate = _mean(wins, default=0.0)
    do_nothing_rate = _mean((float(r.get("do_nothing_rate", 0.0)) for r in records), default=0.0)

    reason = None
    if entropy_mean < 0.3:
        reason = "entropy_mean < 0.3"
    elif rank_mean > 3.7:
        reason = "rank_mean > 3.7"
    elif do_nothing_rate > 0.80:
        reason = "do_nothing_rate > 0.80"
    elif winrate == 0.0 and window >= 20:
        reason = "winrate == 0.0 and window >= 20"

    return {
        "status": "unhealthy" if reason else "healthy",
        "reason": reason,
        "entropy_mean": entropy_mean,
        "rank_mean": rank_mean,
        "winrate": winrate,
        "do_nothing_rate": do_nothing_rate,
        "window": window,
    }


def analyze_log(path: str | Path, window: int) -> Dict[str, Any]:
    return analyze_records(_load_recent_jsonl(path, window))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check health of notebook 4p training logs.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()
    report = analyze_log(args.log, args.window)
    print(json.dumps(report, sort_keys=True))
    return 1 if report["status"] == "unhealthy" else 0


if __name__ == "__main__":
    raise SystemExit(main())
