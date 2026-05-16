"""V15.4 SP2 — fit a value function from self-play data.

Input  : a self-play sample file (analysis/sp_gen<N>.npz) with arrays
         n_players, X (features [.,5]), y (win label).
Output : an EvalWeights npz — separate 2p and 4p logistic value functions,
         same five features as the ESC, but with weights and per-feature
         standardisation fitted to self-play outcomes.

A logistic fit maximises P(win) calibration; RCC only needs the ranking, so
the fitted (w, mean, std) are stored directly into an EvalWeights set. 2p and
4p are fitted separately because their dynamics differ.

Run:
    python train_vf_selfplay.py analysis/sp_gen1.npz analysis/vf_gen1.npz
"""

from __future__ import annotations

import sys

import numpy as np

import v15_eval

L2 = 1e-3
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


def _fit(X, y):
    """Fit one standardised logistic model; return (w, b, mean, std, auc)."""
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    cut = max(1, int(0.8 * len(X)))
    Xtr, ytr, Xva, yva = X[:cut], y[:cut], X[cut:], y[cut:]

    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std < 1e-8] = 1.0
    Ztr = (Xtr - mean) / std
    Zva = (Xva - mean) / std

    w = np.zeros(X.shape[1])
    b = 0.0
    n = len(Ztr)
    for _ in range(ITERS):
        p = _sigmoid(Ztr @ w + b)
        g = p - ytr
        w -= LR * (Ztr.T @ g / n + L2 * w)
        b -= LR * g.mean()

    auc = _auc(yva, _sigmoid(Zva @ w + b)) if len(Xva) else float("nan")
    return w, b, mean, std, auc


def main():
    if len(sys.argv) < 3:
        print("usage: train_vf_selfplay.py <samples.npz> <out_weights.npz>")
        sys.exit(1)
    src, out = sys.argv[1], sys.argv[2]
    d = np.load(src, allow_pickle=True)
    nps, X, y = d["n_players"], d["X"], d["y"]

    res = {}
    for mode, npv in (("2p", 2), ("4p", 4)):
        m = nps == npv
        if m.sum() < 100:
            print(f"{mode}: only {int(m.sum())} samples — skipping (using ESC)")
            res[mode] = None
            continue
        w, b, mean, std, auc = _fit(X[m], y[m])
        print(f"{mode}: n={int(m.sum())} positive_rate={y[m].mean():.3f} "
              f"val_AUC={auc:.4f}")
        names = ["ship", "prod", "planet", "domin", "pmarg", "fleet",
                 "elim", "topprod", "conc", "step", "efleet"]
        for nm, wv in sorted(zip(names, w), key=lambda kv: -abs(kv[1])):
            print(f"    {nm:9s} {wv:+.3f}")
        res[mode] = (w, mean, std)

    # fall back to ESC for any mode without enough data
    esc = v15_eval.ESC
    w2, m2, s2 = res["2p"] if res["2p"] else (esc.w2p, esc.mean2p, esc.std2p)
    w4, m4, s4 = res["4p"] if res["4p"] else (esc.w4p, esc.mean4p, esc.std4p)
    ew = v15_eval.EvalWeights(w2p=w2, w4p=w4, mean2p=m2, std2p=s2,
                              mean4p=m4, std4p=s4, tag=out)
    ew.save(out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
