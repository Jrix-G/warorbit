from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Iterator


V21_FIELDS = (
    "state",
    "candidates",
    "chosen",
    "outcome",
    "esc",
    "episode_id",
    "player",
    "n_players",
    "source",
)


class V21SampleError(ValueError):
    pass


def _jsonable(value: Any, field: str) -> Any:
    try:
        json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise V21SampleError(f"{field} must be JSON serializable") from exc
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_equal(left: Any, right: Any) -> bool:
    return canonical_json(left) == canonical_json(right)


def normalize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical V21 sample dict after defensive validation."""
    if not isinstance(sample, dict):
        raise V21SampleError("sample must be a dict")

    missing = [field for field in V21_FIELDS if field not in sample]
    if missing:
        raise V21SampleError(f"missing fields: {', '.join(missing)}")

    state = _jsonable(sample["state"], "state")
    if state is None:
        raise V21SampleError("state must not be None")

    candidates = sample["candidates"]
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, list):
        raise V21SampleError("candidates must be a non-empty list")
    if not candidates:
        raise V21SampleError("candidates must be non-empty")
    candidates = [_jsonable(candidate, "candidate") for candidate in candidates]

    chosen = _jsonable(sample["chosen"], "chosen")
    if not any(_stable_equal(chosen, candidate) for candidate in candidates):
        raise V21SampleError("chosen must match one candidate")

    outcome = _finite_float(sample["outcome"], "outcome")
    esc = _finite_float(sample["esc"], "esc")

    episode_id = sample["episode_id"]
    if episode_id is None or str(episode_id) == "":
        raise V21SampleError("episode_id must be non-empty")
    episode_id = str(episode_id)

    player = _int_field(sample["player"], "player")
    n_players = _int_field(sample["n_players"], "n_players")
    if n_players < 2:
        raise V21SampleError("n_players must be >= 2")
    if player < 0 or player >= n_players:
        raise V21SampleError("player must satisfy 0 <= player < n_players")

    source = sample["source"]
    if source is None or str(source) == "":
        raise V21SampleError("source must be non-empty")
    source = str(source)

    return {
        "state": state,
        "candidates": candidates,
        "chosen": chosen,
        "outcome": outcome,
        "esc": esc,
        "episode_id": episode_id,
        "player": player,
        "n_players": n_players,
        "source": source,
    }


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise V21SampleError(f"{field} must be a finite number")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise V21SampleError(f"{field} must be a finite number") from exc
    if not math.isfinite(out):
        raise V21SampleError(f"{field} must be finite")
    return out


def _int_field(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise V21SampleError(f"{field} must be an int")
    return value


def sample_key(sample: dict[str, Any]) -> str:
    canonical = normalize_sample(sample)
    payload = {field: canonical[field] for field in V21_FIELDS if field != "source"}
    raw = canonical_json(payload).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def dedupe_samples(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample in samples:
        canonical = normalize_sample(sample)
        key = sample_key(canonical)
        if key in seen:
            continue
        seen.add(key)
        out.append(canonical)
    return out


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise V21SampleError(f"{path}:{line_no}: invalid JSON") from exc
            try:
                yield normalize_sample(raw)
            except V21SampleError as exc:
                raise V21SampleError(f"{path}:{line_no}: {exc}") from exc


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, samples: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for sample in samples:
            canonical = normalize_sample(sample)
            fh.write(canonical_json(canonical) + "\n")
            count += 1
    return count


def split_by_episode(
    samples: Iterable[dict[str, Any]],
    val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError("val_fraction must be in [0, 1]")

    canonical = [normalize_sample(sample) for sample in samples]
    episode_ids = sorted({sample["episode_id"] for sample in canonical})
    rng = random.Random(seed)
    rng.shuffle(episode_ids)

    n_val = int(round(len(episode_ids) * val_fraction))
    if val_fraction > 0.0 and episode_ids and n_val == 0:
        n_val = 1
    val_ids = set(episode_ids[:n_val])

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for sample in canonical:
        if sample["episode_id"] in val_ids:
            val.append(sample)
        else:
            train.append(sample)
    return train, val


def mix_offline_online(
    offline: Iterable[dict[str, Any]],
    online: Iterable[dict[str, Any]],
    online_fraction: float = 0.25,
    seed: int = 0,
) -> list[dict[str, Any]]:
    if not 0.0 <= online_fraction <= 1.0:
        raise ValueError("online_fraction must be in [0, 1]")

    offline_samples = dedupe_samples(offline)
    online_samples = dedupe_samples(online)
    rng = random.Random(seed)
    rng.shuffle(online_samples)

    if online_fraction == 0.0:
        selected_online: list[dict[str, Any]] = []
    elif online_fraction == 1.0:
        selected_online = online_samples
    else:
        max_online = int((online_fraction * len(offline_samples)) / (1.0 - online_fraction))
        selected_online = online_samples[:max_online]

    mixed = dedupe_samples([*offline_samples, *selected_online])
    rng.shuffle(mixed)
    return mixed
