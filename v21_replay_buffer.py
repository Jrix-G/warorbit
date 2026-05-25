from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Iterator

from v21_dataset import (
    V21SampleError,
    canonical_json,
    mix_offline_online,
    normalize_sample,
    sample_key,
)


class V21ReplayBuffer:
    """Append-only JSONL replay buffer with canonical V21 sample dedupe."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def iter_samples(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise V21SampleError(f"{self.path}:{line_no}: invalid JSON") from exc
                try:
                    yield normalize_sample(raw)
                except V21SampleError as exc:
                    raise V21SampleError(f"{self.path}:{line_no}: {exc}") from exc

    def load(self) -> list[dict]:
        return list(self.iter_samples())

    def keys(self) -> set[str]:
        return {sample_key(sample) for sample in self.iter_samples()}

    def append(self, sample: dict) -> bool:
        return self.extend([sample]) == 1

    def extend(self, samples: Iterable[dict]) -> int:
        canonical_samples = [normalize_sample(sample) for sample in samples]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        seen = self.keys()
        appended = 0
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            for canonical in canonical_samples:
                key = sample_key(canonical)
                if key in seen:
                    continue
                fh.write(canonical_json(canonical) + "\n")
                seen.add(key)
                appended += 1
        return appended

    def sample(self, n: int, seed: int | None = None) -> list[dict]:
        if n < 0:
            raise ValueError("n must be >= 0")
        samples = self.load()
        rng = random.Random(seed)
        rng.shuffle(samples)
        return samples[:n]

    def mix_with_offline(
        self,
        offline: Iterable[dict],
        online_fraction: float = 0.25,
        seed: int = 0,
    ) -> list[dict]:
        return mix_offline_online(
            offline=offline,
            online=self.iter_samples(),
            online_fraction=online_fraction,
            seed=seed,
        )

    def __len__(self) -> int:
        return sum(1 for _ in self.iter_samples())
