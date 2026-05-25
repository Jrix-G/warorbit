"""Data diagnostics for V21 datasets and buffers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import v21_dataset


def probe_path(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    suffixes = "".join(p.suffixes).lower()
    if suffixes.endswith(".npz"):
        return probe_npz(p)
    if suffixes.endswith(".jsonl"):
        return probe_jsonl(p)
    raise ValueError(f"unsupported data file: {p}")


def probe_npz(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    z = np.load(p, allow_pickle=False)
    arrays: dict[str, Any] = {}
    for key in z.files:
        arr = z[key]
        arrays[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    report: dict[str, Any] = {"path": str(p), "kind": "npz", "arrays": arrays}
    if "EP" in z.files:
        ep = np.asarray(z["EP"])
        report["episodes"] = int(len(set(int(x) for x in ep.reshape(-1).tolist())))
    if "POL" in z.files and "MASK" in z.files:
        pol = np.asarray(z["POL"])
        mask = np.asarray(z["MASK"]).astype(bool)
        report["policy"] = _policy_stats(pol, mask)
    return report


def probe_jsonl(path: str | Path) -> dict[str, Any]:
    try:
        samples = v21_dataset.load_jsonl(path)
    except v21_dataset.V21SampleError:
        return probe_generic_jsonl(path)
    episodes = {sample["episode_id"] for sample in samples}
    candidates = [len(sample["candidates"]) for sample in samples]
    sources: dict[str, int] = {}
    for sample in samples:
        source = str(sample["source"])
        sources[source] = sources.get(source, 0) + 1
    return {
        "path": str(path),
        "kind": "jsonl",
        "samples": len(samples),
        "episodes": len(episodes),
        "avg_candidates": float(sum(candidates) / len(candidates)) if candidates else 0.0,
        "sources": sources,
        "leakage_risk": "high" if samples and len(episodes) < max(2, len(samples) // 4) else "normal",
    }


def probe_generic_jsonl(path: str | Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    rows = 0
    games = 0
    modes: dict[str, int] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows += 1
            typ = str(row.get("type", "unknown")) if isinstance(row, dict) else "unknown"
            counts[typ] = counts.get(typ, 0) + 1
            if typ == "game":
                games += 1
                mode = str(row.get("mode", row.get("n_players", "unknown")))
                modes[mode] = modes.get(mode, 0) + 1
    return {
        "path": str(path),
        "kind": "jsonl-generic",
        "rows": rows,
        "types": counts,
        "games": games,
        "modes": modes,
    }


def _policy_stats(pol: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if pol.ndim != 3 or mask.ndim != 2:
        return {"valid_rows": 0.0, "pass_rate": 0.0, "action_rate": 0.0}
    valid = mask.reshape(-1)
    if not np.any(valid):
        return {"valid_rows": 0.0, "pass_rate": 0.0, "action_rate": 0.0}
    labels = np.argmax(pol.reshape(-1, pol.shape[-1])[valid], axis=1)
    pass_rate = float(np.mean(labels == 0))
    return {"valid_rows": float(labels.shape[0]), "pass_rate": pass_rate, "action_rate": 1.0 - pass_rate}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe V21/WO datasets")
    parser.add_argument("paths", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps([probe_path(path) for path in args.paths], sort_keys=True))


if __name__ == "__main__":
    main()
