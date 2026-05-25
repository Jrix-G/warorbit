"""V22 linear combo model.

V22 scores complete launch combinations, not isolated shots.  The model is
kept NPZ-compatible so it can be loaded inside a Kaggle submission without a
training framework.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class LinearComboRanker:
    def __init__(self, w: np.ndarray | None = None, b: float = 0.0) -> None:
        self.w = np.asarray(w, dtype=np.float64).reshape(-1) if w is not None else np.zeros(0, dtype=np.float64)
        self.b = float(b)

    @classmethod
    def load(cls, path: str | Path | None) -> "LinearComboRanker":
        if path is None or str(path).strip() == "":
            return cls()
        p = Path(path)
        if not p.exists():
            return cls()
        data = np.load(p, allow_pickle=False)
        return cls(w=data["w"], b=float(data["b"]) if "b" in data.files else 0.0)

    def ready_for(self, dim: int) -> bool:
        return self.w.shape[0] == int(dim)

    def score(self, features: np.ndarray) -> float:
        x = np.asarray(features, dtype=np.float64).reshape(-1)
        if self.w.shape[0] != x.shape[0]:
            return 0.0
        return float(x @ self.w + self.b)


def save(path: str | Path, model: LinearComboRanker, metadata: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        w=model.w.astype(np.float32),
        b=np.asarray(model.b, dtype=np.float32),
        metadata=np.asarray(str(metadata)),
    )
