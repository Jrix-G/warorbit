from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict
import json
import numpy as np


def save_checkpoint(path: str | Path, state: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        np.savez_compressed(f, **state, metadata=json.dumps(metadata))


def load_checkpoint(path: str | Path) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    data = np.load(Path(path), allow_pickle=False)
    state = {k: data[k] for k in data.files if k != "metadata"}
    metadata = json.loads(str(data["metadata"]))
    return state, metadata


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
