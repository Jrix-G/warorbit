"""Phase 2 — fit a calibrated 4p win-probability value function.

Logistic regression (pure numpy) on the dataset from build_value_dataset.py.
V(state) = sigmoid( ((features - mean)/std) . w + b ) ~ P(player finishes #1).

Reports log-loss, AUC and a calibration table; saves weights for the bot.

Run:
    python train_value_fn.py
"""

from __future__ import annotations

import numpy as np

DATASET = "analysis/v15_value_dataset_4p.npz"
OUT = "analysis/v15_value_fn_4p.npz"
L2 = 1e-4
LR = 0.5
ITERS = 4000
SEED = 0


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _auc(y, p):
    order = np.argsort(p)
    y = y[order]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.arange(1, len(y) + 1)
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    d = np.load(DATASET, allow_pickle=True)
    X, y = d["X"], d["y"]
    names = list(d["feature_names"])
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    cut = int(0.8 * len(X))
    Xtr, ytr = X[:cut], y[:cut]
    Xva, yva = X[cut:], y[cut:]

    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std < 1e-8] = 1.0
    Ztr = (Xtr - mean) / std
    Zva = (Xva - mean) / std

    n, dft = Ztr.shape
    w = np.zeros(dft)
    b = 0.0
    for it in range(ITERS):
        p = _sigmoid(Ztr @ w + b)
        g = p - ytr
        w -= LR * (Ztr.T @ g / n + L2 * w)
        b -= LR * g.mean()

    def logloss(Z, yy):
        p = np.clip(_sigmoid(Z @ w + b), 1e-7, 1 - 1e-7)
        return -(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean()

    pva = _sigmoid(Zva @ w + b)
    print(f"train logloss={logloss(Ztr, ytr):.4f}  val logloss={logloss(Zva, yva):.4f}")
    print(f"val AUC={_auc(yva, pva):.4f}  (0.5=random, base positive rate {yva.mean():.3f})")
    print("\ncoefficients (standardized):")
    for nm, wv in sorted(zip(names, w), key=lambda kv: -abs(kv[1])):
        print(f"  {nm:22s} {wv:+.3f}")
    print("\ncalibration (val):")
    for lo in np.arange(0, 1.0, 0.2):
        m = (pva >= lo) & (pva < lo + 0.2)
        if m.sum():
            print(f"  pred[{lo:.1f}-{lo+0.2:.1f}): n={m.sum():6d} "
                  f"predicted~{pva[m].mean():.3f} actual={yva[m].mean():.3f}")

    np.savez(OUT, w=w, b=b, mean=mean, std=std,
             feature_names=np.array(names))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
