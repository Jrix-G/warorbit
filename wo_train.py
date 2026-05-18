"""wo_train — supervised training of the V15++ value network.

Fits WOValueNet to predict a state's eventual game result, by plain
regression (MSE) on real outcomes from replay datasets (wo_dataset.py). This
is curve-fitting on a fixed dataset: a convex, convergent objective — no
policy loop, no self-play, no collapse. The net is later used only to score
leaf positions inside V15's search.

Train/val split is BY EPISODE (whole games held out): consecutive states of a
game are near-duplicates, so a random split would leak and inflate the score.

Every run is benchmarked against the predict-the-mean baseline — if the net
cannot beat a constant, it learned nothing and the result is reported as such.

Run:
    python -u wo_train.py --data analysis/wo_all_data.npz --out analysis/wo_value.pt
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time

import numpy as np
import torch

from wo_net import WOValueNet


def _load(paths):
    """Concatenate datasets; episode ids are made globally unique across files
    so the by-episode split never mixes one game's states across the split."""
    PF, GF, MASK, VAL, EP = [], [], [], [], []
    ep_off = 0
    for p in paths:
        z = np.load(p)
        val = z["VAL"]
        ep = z["EP"] if "EP" in z.files else np.zeros(len(val), np.int32)
        PF.append(z["PF"]); GF.append(z["GF"]); MASK.append(z["MASK"])
        VAL.append(val); EP.append(ep.astype(np.int64) + ep_off)
        ep_off += (int(ep.max()) + 1) if len(ep) else 0
        print(f"  {p}: {len(val)} samples, {ep_off} episodes cumulative")
    return (np.concatenate(PF), np.concatenate(GF), np.concatenate(MASK),
            np.concatenate(VAL), np.concatenate(EP))


def _split(ep, val_frac, seed):
    """Hold out whole episodes for validation."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(ep)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_eps = set(uniq[:n_val].tolist())
    is_val = np.fromiter((e in val_eps for e in ep), dtype=bool, count=len(ep))
    return ~is_val, is_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True,
                    help="one or more wo_dataset .npz files")
    ap.add_argument("--out", default="analysis/wo_value.pt")
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.manual_seed(args.seed)

    PF, GF, MASK, VAL, EP = _load(args.data)
    tr, va = _split(EP, args.val_frac, args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[wo_train] {len(VAL)} samples  train={int(tr.sum())} "
          f"val={int(va.sum())}  episodes={len(np.unique(EP))}  device={dev}")

    def _t(a, idx, dt=torch.float32):
        return torch.as_tensor(a[idx], dtype=dt)

    PFtr, GFtr = _t(PF, tr), _t(GF, tr)
    MKtr, VLtr = _t(MASK, tr, torch.bool), _t(VAL, tr)
    PFva, GFva = _t(PF, va).to(dev), _t(GF, va).to(dev)
    MKva, VLva = _t(MASK, va, torch.bool).to(dev), _t(VAL, va).to(dev)

    net = WOValueNet(d=args.d).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=args.wd)
    n = len(VLtr)
    # trivial baseline: predict the (train) mean for every val state.
    base = float(((VLva - VLtr.mean().to(dev)) ** 2).mean())

    best_mse, best_state, bad = 1e9, None, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            pred = net(PFtr[idx].to(dev), GFtr[idx].to(dev),
                       MKtr[idx].to(dev))
            loss = ((pred - VLtr[idx].to(dev)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            pv = net(PFva, GFva, MKva)
            vmse = float(((pv - VLva) ** 2).mean())
            sign = float(((pv > 0) == (VLva > 0)).float().mean())
        print(f"[wo_train] ep{ep:3d}  train_mse={tot / n:.4f}  "
              f"val_mse={vmse:.4f}  val_sign_acc={sign:.3f}")
        if vmse < best_mse - 1e-4:
            best_mse, bad = vmse, 0
            best_state = {k: v.cpu().clone()
                          for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[wo_train] early stop ({bad} epochs no val gain)")
                break

    torch.save({"state_dict": best_state, "d": args.d,
                "val_mse": best_mse, "baseline_mse": base},
               args.out)
    gain = (base - best_mse) / base * 100.0 if base > 0 else 0.0
    print(f"[wo_train] done ({(time.time() - t0) / 60:.1f} min)  "
          f"best val_mse={best_mse:.4f}  baseline(predict-mean)={base:.4f}")
    if best_mse < base - 1e-3:
        print(f"[wo_train] net beats baseline by {gain:.1f}% -> real signal "
              f"learned.  saved -> {args.out}")
    else:
        print("[wo_train] WARNING: net does not beat predict-the-mean — "
              "no usable signal (need more/cleaner data).")


if __name__ == "__main__":
    main()
