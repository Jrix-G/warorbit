"""Train V22 linear combo rankers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import v22_dataset
import v22_features
import v22_model


def train_combo_ranker(
    samples: Iterable[dict[str, Any]],
    *,
    epochs: int = 100,
    lr: float = 0.12,
    l2: float = 1.0e-4,
) -> tuple[v22_model.LinearComboRanker, dict[str, float]]:
    data = [v22_dataset.normalize_sample(sample) for sample in samples]
    if not data:
        raise ValueError("at least one sample is required")
    dim = len(v22_features.FEATURE_NAMES)
    w = np.zeros(dim, dtype=np.float64)
    b = 0.0
    for _ in range(max(1, int(epochs))):
        grad_w = np.zeros(dim, dtype=np.float64)
        grad_b = 0.0
        denom = 0
        for sample in data:
            x, target = _matrix(sample, dim)
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
    model = v22_model.LinearComboRanker(w=w, b=b)
    return model, evaluate(data, model)


def evaluate(samples: Iterable[dict[str, Any]], model: v22_model.LinearComboRanker) -> dict[str, float]:
    data = [v22_dataset.normalize_sample(sample) for sample in samples]
    dim = len(v22_features.FEATURE_NAMES)
    total = 0
    top1 = 0
    losses: list[float] = []
    for sample in data:
        x, target = _matrix(sample, dim)
        logits = x @ model.w + model.b if model.ready_for(dim) else np.zeros(len(x), dtype=np.float64)
        probs = _softmax(logits)
        label = int(np.argmax(target))
        top1 += int(np.argmax(logits) == label)
        losses.append(float(-np.sum(target * np.log(np.maximum(1.0e-12, probs)))))
        total += 1
    return {"samples": float(total), "top1": top1 / max(1, total), "loss": float(np.mean(losses)) if losses else 0.0}


def train_from_jsonl(input_path: str | Path, output_path: str | Path, *, val_fraction: float, seed: int, epochs: int, lr: float):
    samples = v22_dataset.load_jsonl(input_path)
    train, val = v22_dataset.split_by_episode(samples, val_fraction=val_fraction, seed=seed)
    if not train:
        train = samples
    model, train_metrics = train_combo_ranker(train, epochs=epochs, lr=lr)
    val_metrics = evaluate(val or train, model)
    report = {"train": train_metrics, "val": val_metrics, "train_samples": len(train), "val_samples": len(val)}
    v22_model.save(output_path, model, metadata=json.dumps(report, sort_keys=True))
    return report


def _matrix(sample: dict[str, Any], dim: int) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    weights = []
    for combo in sample["combos"]:
        arr = np.asarray(combo["features"], dtype=np.float64).reshape(-1)
        if arr.shape[0] != dim:
            raise ValueError(f"combo feature size {arr.shape[0]} != {dim}")
        rows.append(arr)
        weights.append(max(0.0, float(combo.get("target_weight", 0.0))))
    target = np.asarray(weights, dtype=np.float64)
    if float(target.sum()) <= 0.0:
        target = np.zeros(len(rows), dtype=np.float64)
        target[int(sample["chosen"])] = 1.0
    else:
        target = target / max(1.0e-12, float(target.sum()))
    return np.vstack(rows), target


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - float(np.max(z))
    exp = np.exp(z)
    return exp / max(1.0e-12, float(exp.sum()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V22 combo ranker")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.12)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_from_jsonl(
        args.input,
        args.output,
        val_fraction=args.val_fraction,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
