"""wo_policy_train — imitation learning for the policy net.

Cross-entropy loss on owned-planet rows only. POL[S,N,N+1] is one-hot
per owned planet (0=pass, j+1=target planet j). Non-owned planets are
excluded from the loss via the mask.

Run:
    python -u wo_policy_train.py --data analysis/wo_all_data.npz \
        --out analysis/wo_policy.pt
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from wo_policy_net import WOPolicyNet


def _load(paths):
    PF, GF, POL, MASK, EP = [], [], [], [], []
    ep_off = 0
    for p in paths:
        z = np.load(p)
        ep = z["EP"] if "EP" in z.files else np.zeros(len(z["VAL"]), np.int32)
        PF.append(z["PF"]); GF.append(z["GF"]); POL.append(z["POL"])
        MASK.append(z["MASK"]); EP.append(ep.astype(np.int64) + ep_off)
        ep_off += (int(ep.max()) + 1) if len(ep) else 0
        print(f"  {p}: {len(z['VAL'])} samples")
    return (np.concatenate(PF), np.concatenate(GF),
            np.concatenate(POL), np.concatenate(MASK), np.concatenate(EP))


def _split(ep, val_frac, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(ep)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_eps = set(uniq[:n_val].tolist())
    is_val = np.fromiter((e in val_eps for e in ep), dtype=bool, count=len(ep))
    return ~is_val, is_val


def _policy_loss(logits, pol_target, mask):
    """Cross-entropy over owned planets only.

    logits:     [B, N, N+1]
    pol_target: [B, N, N+1]  one-hot
    mask:       [B, N]  bool (True = valid planet)
    """
    B, N, A = logits.shape
    # owned = has a non-zero row in pol_target (pass+launch sum > 0)
    owned = (pol_target.sum(dim=-1) > 0.5) & mask    # [B,N]
    if not owned.any():
        return logits.sum() * 0.0

    logits_flat = logits[owned]          # [K, N+1]
    target_flat = pol_target[owned]      # [K, N+1]
    label = target_flat.argmax(dim=-1)   # [K]
    return F.cross_entropy(logits_flat, label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default="analysis/wo_policy.pt")
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.manual_seed(args.seed)

    PF, GF, POL, MASK, EP = _load(args.data)
    tr, va = _split(EP, args.val_frac, args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_tr = int(tr.sum())
    print(f"[pol_train] {len(EP)} samples  train={n_tr}  "
          f"val={int(va.sum())}  eps={len(np.unique(EP))}  device={dev}")

    def _t(a, idx, dt=torch.float32):
        return torch.as_tensor(a[idx], dtype=dt)

    PFtr = _t(PF, tr); GFtr = _t(GF, tr)
    MKtr = _t(MASK, tr, torch.bool); PLtr = _t(POL, tr)
    PFva = _t(PF, va).to(dev); GFva = _t(GF, va).to(dev)
    MKva = _t(MASK, va, torch.bool).to(dev); PLva = _t(POL, va).to(dev)

    net = WOPolicyNet(d=args.d).to(dev)
    print(f"[pol_train] {net.n_params()} parameters")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

    best_loss, best_state, bad = 1e9, None, 0
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(n_tr)
        tot = 0.0; cnt = 0
        for i in range(0, n_tr, args.batch):
            idx = perm[i:i + args.batch]
            logits = net(PFtr[idx].to(dev), GFtr[idx].to(dev), MKtr[idx].to(dev))
            loss = _policy_loss(logits, PLtr[idx].to(dev), MKtr[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx); cnt += len(idx)

        net.eval()
        with torch.no_grad():
            vl = net(PFva, GFva, MKva)
            vloss = _policy_loss(vl, PLva, MKva).item()
            # top-1 accuracy on owned planets
            owned_va = (PLva.sum(dim=-1) > 0.5) & MKva
            pred_label = vl[owned_va].argmax(dim=-1)
            true_label = PLva[owned_va].argmax(dim=-1)
            acc = (pred_label == true_label).float().mean().item()

        print(f"[pol_train] ep{ep:3d}  train_ce={tot/cnt:.4f}  "
              f"val_ce={vloss:.4f}  val_acc={acc:.3f}")

        if vloss < best_loss - 1e-4:
            best_loss, bad = vloss, 0
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[pol_train] early stop ({bad} epochs no gain)")
                break

    torch.save({"state_dict": best_state, "d": args.d, "val_ce": best_loss},
               args.out)
    print(f"[pol_train] done ({(time.time()-t0)/60:.1f} min)  "
          f"best val_ce={best_loss:.4f}  saved -> {args.out}")


if __name__ == "__main__":
    main()
