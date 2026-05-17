"""v17_warmstart — clone V15 into the V17 network (behavioural cloning).

A randomly-initialised AlphaZero net plays terribly and would spend weeks of
self-play just climbing back to V15's level. Instead we warm-start: generate
V15 (RCC+V7) self-play games, and train the network to imitate V15's move
(policy) and predict the game outcome (value). The net then begins the
self-play loop at ~V15 strength.

Phase-0 gate: the cloned net, played greedily, must perform close to V15.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE python -u v17_warmstart.py --games 240
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

import v14_core
import v15_fast_sim as fsim
import v15_search
import v17_encode as enc
from v17_net import V17Net
from local_simulator.official_fast import OfficialFastGame

N_MAX = 48                  # planet-slot padding
EPISODE = 260
SAMPLE_EVERY = 2
ANGLE_TOL = 0.5


def _move_to_labels(fs, player, move):
    """V15's move -> per-planet target label: 0=pass, j+1=target planet j."""
    n = len(fs.planets)
    ids = {int(fs.planets[i, enc.ID]): i for i in range(n)}
    label = np.full(n, -100, dtype=np.int64)       # -100 = ignore (not owned)
    for i in range(n):
        if int(fs.planets[i, enc.OWNER]) == player:
            label[i] = 0                            # owned -> default pass
    for mv in (move or []):
        if not (isinstance(mv, list) and len(mv) == 3):
            continue
        si = ids.get(int(mv[0]))
        if si is None or int(fs.planets[si, enc.OWNER]) != player:
            continue
        sx, sy = fs.planets[si, enc.X], fs.planets[si, enc.Y]
        best_j, best_d = -1, 9.9
        for j in range(n):
            if j == si:
                continue
            b = math.atan2(fs.planets[j, enc.Y] - sy,
                           fs.planets[j, enc.X] - sx)
            d = abs((b - mv[1] + math.pi) % (2 * math.pi) - math.pi)
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d < ANGLE_TOL:
            label[si] = best_j + 1
    return label


def _play_v15_game(task):
    """One V15-vs-V15 game; return per-(step,player) cloning samples."""
    n_players, seed = task
    g = OfficialFastGame(n_players, seed=seed, episode_steps=EPISODE,
                         use_c_accel=False)
    obs0 = v14_core.obs_as_dict(g.observation(0))
    fs = fsim.from_obs(obs0, n_players=n_players, episode_steps=EPISODE)
    fs.n_players = n_players
    raw = []
    t = 0
    while not fs.done:
        moves = []
        for p in range(n_players):
            o = v15_search.state_to_obs(fs, p)
            m = v15_search.search(o, None)
            m = m if isinstance(m, list) else []
            moves.append(m)
            if t % SAMPLE_EVERY == 0 and 6 <= t < EPISODE - 8:
                pf, gf = enc.encode(fs, p)
                raw.append((pf, gf, _move_to_labels(fs, p, m), p))
        fs = fsim.step(fs, moves)
        t += 1
    sc = fsim.scores(fs)
    best = max(sc) if sc else 0
    winners = [p for p in range(n_players) if sc[p] == best and best > 0]
    out = {}
    for p in range(n_players):
        out[p] = 1.0 if (len(winners) == 1 and winners[0] == p) else (
            0.0 if p in winners else -1.0)
    return [(pf, gf, lab, out[p]) for (pf, gf, lab, p) in raw]


def _pad(pf, lab):
    """Pad a sample's planet axis to N_MAX; return (pf, mask, label)."""
    n = pf.shape[0]
    pfp = np.zeros((N_MAX, enc.P_DIM), dtype=np.float32)
    mask = np.zeros(N_MAX, dtype=bool)
    labp = np.full(N_MAX, -100, dtype=np.int64)
    k = min(n, N_MAX)
    pfp[:k] = pf[:k]
    mask[:k] = True
    labp[:k] = np.where(lab[:k] >= 0, np.minimum(lab[:k], N_MAX), lab[:k])
    return pfp, mask, labp


def generate(games, workers):
    tasks = [(2, 400000 + i) for i in range(games // 2)]
    tasks += [(4, 500000 + i) for i in range(games - games // 2)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_play_v15_game, tasks))
    PF, GF, LAB, VAL = [], [], [], []
    for samples in results:
        for (pf, gf, lab, val) in samples:
            pfp, mask, labp = _pad(pf, lab)
            PF.append(pfp)
            GF.append(gf)
            LAB.append(labp)
            VAL.append(val)
    print(f"[warmstart] {games} V15 games -> {len(PF)} samples "
          f"({(time.time()-t0)/60:.1f} min)")
    return (np.stack(PF), np.stack(GF).astype(np.float32),
            np.stack(LAB), np.array(VAL, dtype=np.float32))


def train(PF, GF, LAB, VAL, epochs, d, lr, device):
    net = V17Net(d=d).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(PF)
    pf = torch.tensor(PF, device=device)
    gf = torch.tensor(GF, device=device)
    # build planet mask from is_mine|is_enemy|is_neutral (any owner-flag set)
    mask = (pf[:, :, 0] + pf[:, :, 1] + pf[:, :, 2]) > 0.5
    lab = torch.tensor(LAB, device=device)
    val = torch.tensor(VAL, device=device)
    idx = np.arange(n)
    bs = 256
    for ep in range(epochs):
        np.random.shuffle(idx)
        tot_p = tot_v = correct = counted = atk_correct = atk_counted = 0.0
        for s in range(0, n, bs):
            b = idx[s:s + bs]
            bi = torch.tensor(b, device=device)
            logits, value = net(pf[bi], gf[bi], mask[bi])
            B, N, A = logits.shape
            lab_flat = lab[bi].reshape(B * N)
            logits_flat = logits.reshape(B * N, A)
            # upweight attack moves 8x to counter pass-class dominance
            ce = F.cross_entropy(logits_flat, lab_flat,
                                 ignore_index=-100, reduction='none')
            owned_mask = lab_flat != -100
            w = torch.ones(B * N, device=device)
            w[lab_flat > 0] = 8.0
            pl = (ce * w)[owned_mask].mean()
            vl = F.mse_loss(value, val[bi])
            loss = pl + vl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_p += pl.item() * len(b)
            tot_v += vl.item() * len(b)
            with torch.no_grad():
                lb = lab_flat
                pr = logits_flat.argmax(-1)
                m = lb != -100
                correct += (pr[m] == lb[m]).sum().item()
                counted += m.sum().item()
                atk = m & (lb > 0)
                atk_correct += (pr[atk] == lb[atk]).sum().item()
                atk_counted += atk.sum().item()
        print(f"[warmstart] epoch {ep+1}/{epochs}: policy_ce={tot_p/n:.4f} "
              f"value_mse={tot_v/n:.4f} move_acc={correct/counted:.3f} "
              f"attack_acc={atk_correct/max(atk_counted,1):.3f}")
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=240)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    os.makedirs("analysis", exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    PF, GF, LAB, VAL = generate(args.games, args.workers)
    net = train(PF, GF, LAB, VAL, args.epochs, args.d, args.lr, dev)
    torch.save({"state_dict": net.state_dict(), "d": args.d},
               "analysis/v17_warmstart.pt")
    print("[warmstart] -> analysis/v17_warmstart.pt")


if __name__ == "__main__":
    main()
