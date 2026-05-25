"""V22 combo dataset format."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterable


FIELDS = ("state", "combos", "chosen", "baseline", "episode_id", "player", "n_players", "source")


class V22SampleError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def normalize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(sample, dict):
        raise V22SampleError("sample must be a dict")
    missing = [field for field in FIELDS if field not in sample]
    if missing:
        raise V22SampleError(f"missing fields: {', '.join(missing)}")
    state = _jsonable(sample["state"], "state")
    combos = sample["combos"]
    if not isinstance(combos, list) or not combos:
        raise V22SampleError("combos must be a non-empty list")
    combos = [_combo(row) for row in combos]
    chosen = _int(sample["chosen"], "chosen")
    if chosen < 0 or chosen >= len(combos):
        raise V22SampleError("chosen out of range")
    baseline = _finite(sample["baseline"], "baseline")
    episode_id = str(sample["episode_id"])
    if not episode_id:
        raise V22SampleError("episode_id must be non-empty")
    player = _int(sample["player"], "player")
    n_players = _int(sample["n_players"], "n_players")
    if n_players < 2 or player < 0 or player >= n_players:
        raise V22SampleError("invalid player/n_players")
    source = str(sample["source"])
    if not source:
        raise V22SampleError("source must be non-empty")
    return {
        "state": state,
        "combos": combos,
        "chosen": chosen,
        "baseline": baseline,
        "episode_id": episode_id,
        "player": player,
        "n_players": n_players,
        "source": source,
    }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(normalize_sample(json.loads(line)))
            except Exception as exc:
                raise V22SampleError(f"{path}:{line_no}: {exc}") from exc
    return out


def write_jsonl(path: str | Path, samples: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for sample in samples:
            fh.write(canonical_json(normalize_sample(sample)) + "\n")
            count += 1
    return count


def split_by_episode(samples: Iterable[dict[str, Any]], val_fraction: float = 0.2, seed: int = 0):
    rows = [normalize_sample(sample) for sample in samples]
    episodes = sorted({row["episode_id"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(episodes)
    n_val = int(round(len(episodes) * val_fraction))
    if val_fraction > 0.0 and episodes and n_val == 0:
        n_val = 1
    val_ids = set(episodes[:n_val])
    train = [row for row in rows if row["episode_id"] not in val_ids]
    val = [row for row in rows if row["episode_id"] in val_ids]
    return train, val


def _combo(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise V22SampleError("combo must be a dict")
    shots = row.get("shots")
    if not isinstance(shots, list):
        raise V22SampleError("combo shots must be a list")
    features = row.get("features")
    if not isinstance(features, list) or not features:
        raise V22SampleError("combo features must be a non-empty list")
    return {
        "shots": _jsonable(shots, "shots"),
        "features": [_finite(x, "feature") for x in features],
        "score": _finite(row.get("score", 0.0), "score"),
        "passive_score": _finite(row.get("passive_score", 0.0), "passive_score"),
        "det_score": _finite(row.get("det_score", 0.0), "det_score"),
        "target_weight": max(0.0, _finite(row.get("target_weight", 0.0), "target_weight")),
    }


def _jsonable(value: Any, field: str) -> Any:
    try:
        json.dumps(value, sort_keys=True, separators=(",", ":"))
    except Exception as exc:
        raise V22SampleError(f"{field} must be JSON serializable") from exc
    return value


def _finite(value: Any, field: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise V22SampleError(f"{field} must be a number") from exc
    if not math.isfinite(out):
        raise V22SampleError(f"{field} must be finite")
    return out


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise V22SampleError(f"{field} must be int")
    return int(value)
