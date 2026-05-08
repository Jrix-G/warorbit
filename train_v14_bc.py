#!/usr/bin/env python3
"""Train the V14 candidate ranker by behavioral cloning."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import v14_core


def _backward(model: v14_core.V14Scorer, cache: dict[str, np.ndarray], grad_logits: np.ndarray):
    g3 = grad_logits.reshape(-1, 1)
    h2 = cache["h2"]
    h1 = cache["h1"]
    x = cache["x"]
    z2 = cache["z2"]
    z1 = cache["z1"]
    dW3 = h2.T @ g3
    db3 = g3.sum(axis=0)
    dh2 = g3 @ model.W3.T
    dz2 = dh2 * (z2 > 0.0)
    dW2 = h1.T @ dz2
    db2 = dz2.sum(axis=0)
    dh1 = dz2 @ model.W2.T
    dz1 = dh1 * (z1 > 0.0)
    dW1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


class Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float):
        self.lr = lr
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k in params:
            g = grads[k].astype(np.float32)
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * (g * g)
            mh = self.m[k] / (1.0 - 0.9 ** self.t)
            vh = self.v[k] / (1.0 - 0.999 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def _batch_loss_and_grads(model: v14_core.V14Scorer, X, mask, y, l2: float):
    params = model.to_dict()
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    losses = []
    correct = 0
    used = 0
    for feats, valid, label in zip(X, mask, y):
        k = int(valid.sum())
        if k <= 0 or label >= k:
            continue
        scores, cache = model.forward_with_cache(feats[:k])
        probs = v14_core.softmax(scores)
        losses.append(float(-np.log(probs[int(label)] + 1e-12)))
        correct += int(np.argmax(probs) == int(label))
        used += 1
        grad_logits = probs
        grad_logits[int(label)] -= 1.0
        grad_logits /= max(1, len(X))
        sample_grads = _backward(model, cache, grad_logits)
        for key in grads:
            grads[key] += sample_grads[key]
    if l2 > 0:
        for key in ("W1", "W2", "W3"):
            grads[key] += l2 * params[key]
    return float(np.mean(losses) if losses else 0.0), correct / max(1, used), grads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("replay_dataset/v14_bc_top1.npz"))
    parser.add_argument("--out", type=Path, default=Path("evaluations/scorer_v14.npz"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()

    data = np.load(args.data)
    X = data["X"].astype(np.float32)
    mask = data["mask"].astype(np.float32)
    y = data["y"].astype(np.int64)
    rng = np.random.default_rng(args.seed)
    model = v14_core.V14Scorer(seed=args.seed)
    opt = Adam(model.to_dict(), lr=args.lr)

    n = len(y)
    split = max(1, int(n * 0.9))
    order = rng.permutation(n)
    train_idx = order[:split]
    val_idx = order[split:] if split < n else order[: min(n, 256)]
    best_val = 1e9
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        rng.shuffle(train_idx)
        losses = []
        accs = []
        for start in range(0, len(train_idx), args.batch_size):
            idx = train_idx[start:start + args.batch_size]
            loss, acc, grads = _batch_loss_and_grads(model, X[idx], mask[idx], y[idx], args.l2)
            params = model.to_dict()
            opt.step(params, grads)
            model.W1, model.b1 = params["W1"], params["b1"]
            model.W2, model.b2 = params["W2"], params["b2"]
            model.W3, model.b3 = params["W3"], params["b3"]
            losses.append(loss)
            accs.append(acc)
        val_loss, val_acc, _ = _batch_loss_and_grads(model, X[val_idx], mask[val_idx], y[val_idx], 0.0)
        print(
            f"epoch={epoch + 1:03d} train_loss={np.mean(losses):.4f} "
            f"train_acc={np.mean(accs):.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}",
            flush=True,
        )
        if val_loss < best_val:
            best_val = val_loss
            np.savez(args.out, **model.to_dict())
            print(f"  saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
