from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from .health_check import analyze_log
except ImportError:
    from health_check import analyze_log  # type: ignore


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_if_changed(config: Dict[str, Any], key: str, value: Any, changes: List[Tuple[str, Any, Any]]) -> None:
    before = config.get(key)
    if before != value:
        config[key] = value
        changes.append((key, before, value))


def apply_autocorrect(config: Dict[str, Any], health: Dict[str, Any]) -> tuple[Dict[str, Any], List[Tuple[str, Any, Any]]]:
    corrected = dict(config)
    changes: List[Tuple[str, Any, Any]] = []
    train_steps = corrected.get("train_steps")

    entropy_mean = float(health.get("entropy_mean", 0.0))
    rank_mean = float(health.get("rank_mean", 4.0))
    do_nothing_rate = float(health.get("do_nothing_rate", 0.0))
    winrate = float(health.get("winrate", 0.0))
    window = int(health.get("window", 0))

    if entropy_mean < 0.3:
        next_entropy = min(float(corrected.get("entropy_coef_start", 0.05)) * 2.0, 0.20)
        _set_if_changed(corrected, "entropy_coef_start", next_entropy, changes)
        _set_if_changed(corrected, "temperature_start", 1.5, changes)

    if rank_mean > 3.7 and do_nothing_rate > 0.80:
        next_entropy = min(float(corrected.get("entropy_coef_start", 0.05)) * 2.0, 0.20)
        next_lr = max(float(corrected.get("learning_rate", 0.0003)) * 0.5, 1e-5)
        _set_if_changed(corrected, "entropy_coef_start", next_entropy, changes)
        _set_if_changed(corrected, "learning_rate", next_lr, changes)

    if winrate == 0.0 and window >= 20:
        _set_if_changed(corrected, "notebook_pool_limit", 2, changes)
        _set_if_changed(corrected, "entropy_coef_start", 0.08, changes)

    if train_steps is not None:
        corrected["train_steps"] = train_steps
    return corrected, changes


def _write_changes(log_path: Path, health: Dict[str, Any], changes: List[Tuple[str, Any, Any]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": _timestamp(),
        "health": health,
        "changes": [
            {"key": key, "before": before, "after": after}
            for key, before, after in changes
        ],
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply targeted autocorrections to training config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--output-config", default=None)
    parser.add_argument("--autocorrect-log", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    health = analyze_log(args.log, args.window)
    corrected, changes = apply_autocorrect(config, health)
    if not changes:
        print(json.dumps({"status": "unchanged", "health": health, "changes": []}, sort_keys=True))
        return 2

    output_path = Path(args.output_config) if args.output_config else config_path.parent / "autocorrected_config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(corrected, indent=2, sort_keys=True), encoding="utf-8")

    base_dir = config_path.parent.parent
    autocorrect_log = Path(args.autocorrect_log) if args.autocorrect_log else base_dir / "logs" / "autocorrect.log"
    _write_changes(autocorrect_log, health, changes)
    print(json.dumps({"status": "corrected", "output_config": str(output_path), "health": health, "changes": changes}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
