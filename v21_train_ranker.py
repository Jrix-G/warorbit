"""Train and evaluate V21 linear candidate rankers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import v21_dataset
import v21_policy_ranker


def train_ranker(
    samples: Iterable[dict[str, Any]],
    *,
    epochs: int = 80,
    lr: float = 0.2,
    l2: float = 1.0e-4,
) -> tuple[v21_policy_ranker.LinearCandidateRanker, dict[str, float]]:
    data = [v21_dataset.normalize_sample(sample) for sample in samples]
    if not data:
        raise ValueError("at least one sample is required")
    dim = len(v21_policy_ranker.FEATURE_NAMES)
    w = np.zeros(dim, dtype=np.float64)
    b = 0.0
    for _epoch in range(max(1, int(epochs))):
        grad_w = np.zeros(dim, dtype=np.float64)
        grad_b = 0.0
        denom = 0
        for sample in data:
            x, _label, target = _sample_matrix(sample, dim)
            logits = x @ w + b
            probs = _softmax(logits)
            diff = probs - target
            grad_w += x.T @ diff
            grad_b += float(diff.sum())
            denom += 1
        grad_w = grad_w / max(1, denom) + float(l2) * w
        grad_b = grad_b / max(1, denom)
        w -= float(lr) * grad_w
        b -= float(lr) * grad_b
    ranker = v21_policy_ranker.LinearCandidateRanker(w=w, b=b)
    metrics = evaluate_ranker(data, ranker)
    return ranker, metrics


def evaluate_ranker(
    samples: Iterable[dict[str, Any]],
    ranker: v21_policy_ranker.LinearCandidateRanker,
) -> dict[str, float]:
    data = [v21_dataset.normalize_sample(sample) for sample in samples]
    dim = len(v21_policy_ranker.FEATURE_NAMES)
    total = 0
    top1 = 0
    losses: list[float] = []
    for sample in data:
        x, label, target = _sample_matrix(sample, dim)
        logits = x @ ranker.w + ranker.b
        probs = _softmax(logits)
        losses.append(float(-np.sum(target * np.log(np.maximum(1.0e-12, probs)))))
        top1 += int(np.argmax(logits) == label)
        total += 1
    return {
        "samples": float(total),
        "top1": top1 / max(1, total),
        "loss": float(np.mean(losses)) if losses else 0.0,
    }


def save_ranker(
    path: str | Path,
    ranker: v21_policy_ranker.LinearCandidateRanker,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps(metadata or {}, sort_keys=True)
    np.savez_compressed(path, w=ranker.w.astype(np.float32), b=np.asarray(ranker.b, dtype=np.float32), metadata=np.asarray(meta))


def train_from_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
    epochs: int = 80,
    lr: float = 0.2,
) -> dict[str, Any]:
    samples = v21_dataset.load_jsonl(input_path)
    train, val = v21_dataset.split_by_episode(samples, val_fraction=val_fraction, seed=seed)
    if not train:
        train = samples
    ranker, train_metrics = train_ranker(train, epochs=epochs, lr=lr)
    val_metrics = evaluate_ranker(val or train, ranker)
    report = {"train": train_metrics, "val": val_metrics, "train_samples": len(train), "val_samples": len(val)}
    save_ranker(output_path, ranker, metadata=report)
    return report


def _sample_matrix(sample: dict[str, Any], dim: int) -> tuple[np.ndarray, int, np.ndarray]:
    candidates = sample["candidates"]
    chosen = sample["chosen"]
    rows = []
    target_weights: list[float] = []
    label = -1
    chosen_json = v21_dataset.canonical_json(chosen)
    for idx, candidate in enumerate(candidates):
        features = candidate.get("features") if isinstance(candidate, dict) else None
        if features is None:
            raise ValueError("candidate missing features")
        arr = np.asarray(features, dtype=np.float64).reshape(-1)
        if arr.shape[0] != dim:
            raise ValueError(f"candidate feature size {arr.shape[0]} != {dim}")
        rows.append(arr)
        target_weights.append(_candidate_target_weight(candidate))
        if v21_dataset.canonical_json(candidate) == chosen_json:
            label = idx
    if label < 0:
        raise ValueError("chosen candidate not found")
    target = np.asarray(target_weights, dtype=np.float64)
    if float(target.sum()) <= 0.0:
        target = np.zeros(len(rows), dtype=np.float64)
        target[label] = 1.0
    else:
        target = target / max(1.0e-12, float(target.sum()))
    return np.vstack(rows), label, target


def _candidate_target_weight(candidate: Any) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    for key in ("target_weight", "oracle_target_weight"):
        if key in candidate:
            try:
                value = float(candidate[key])
            except (TypeError, ValueError):
                return 0.0
            if not np.isfinite(value):
                return 0.0
            return max(0.0, value)
    return 0.0


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - float(np.max(z))
    exp = np.exp(z)
    return exp / max(1.0e-12, float(np.sum(exp)))


def _cmd_smoke(args: argparse.Namespace) -> dict[str, Any]:
    dim = len(v21_policy_ranker.FEATURE_NAMES)
    good = [1.0] + [0.0] * (dim - 1)
    bad = [0.0] * dim
    samples = []
    for i in range(12):
        chosen = {"shot": [0, 0.0, 10], "features": good}
        other = {"shot": [0, 1.0, 2], "features": bad}
        samples.append(
            {
                "state": {"i": i},
                "candidates": [chosen, other],
                "chosen": chosen,
                "outcome": 1.0,
                "esc": 0.0,
                "episode_id": f"ep-{i}",
                "player": 0,
                "n_players": 2,
                "source": "smoke",
            }
        )
    ranker, metrics = train_ranker(samples, epochs=40, lr=0.3)
    if args.out:
        save_ranker(args.out, ranker, metadata={"smoke": metrics})
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V21 linear candidate ranker")
    sub = parser.add_subparsers(dest="cmd", required=True)
    train = sub.add_parser("train")
    train.add_argument("--input", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--lr", type=float, default=0.2)
    train.add_argument("--val-fraction", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=0)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "train":
        report = train_from_jsonl(args.input, args.output, val_fraction=args.val_fraction, seed=args.seed, epochs=args.epochs, lr=args.lr)
    else:
        report = _cmd_smoke(args)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
