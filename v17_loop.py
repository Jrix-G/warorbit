"""v17_loop — AlphaZero policy/value iteration orchestrator.

Each iteration:
  1. SELF-PLAY — the current net + MCTS plays games (CPU multiprocessing);
     records (state, MCTS policy, outcome).
  2. TRAIN — the net is trained toward the MCTS policy and the outcomes.
  3. The improved net drives the next iteration's self-play.

This is the ratchet: MCTS amplifies the net, training distils the amplified
policy back, repeat. Warm-started from the V15 clone.

Checkpoint/resume every iteration — a multi-day run survives Ctrl-C.

Run (resumable):
    python -u v17_loop.py --iterations 18 --games 200 --n-sims 100
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from v17_net import V17Net
from v17_selfplay import play_game
from v17_train import train_net

CKPT = "analysis/v17_loop.pt"
WARMSTART = "analysis/v17_warmstart.pt"


def _stack(buffer):
    PF = np.stack([s[0] for s in buffer])
    GF = np.stack([s[1] for s in buffer]).astype(np.float32)
    POL = np.stack([s[2] for s in buffer])
    MASK = np.stack([s[3] for s in buffer])
    VAL = np.array([s[4] for s in buffer], dtype=np.float32)
    return PF, GF, POL, MASK, VAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=18)
    ap.add_argument("--games", type=int, default=200, help="games/iteration")
    ap.add_argument("--n-sims", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--buffer", type=int, default=160000, help="max samples")
    ap.add_argument("--mode", type=int, default=2, choices=(2, 4),
                    help="2 = 2p only (Phase 1/2), 4 = include 4p")
    args = ap.parse_args()
    os.makedirs("analysis", exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(CKPT):
        c = torch.load(CKPT, map_location="cpu")
        d = c["d"]
        net = V17Net(d=d)
        net.load_state_dict(c["state_dict"])
        start = c["iter"] + 1
        print(f"[loop] resumed at iteration {start}")
    else:
        w = torch.load(WARMSTART, map_location="cpu")
        d = w["d"]
        net = V17Net(d=d)
        net.load_state_dict(w["state_dict"])
        start = 1
        print(f"[loop] warm-started from {WARMSTART}")

    buffer: list = []
    for it in range(start, args.iterations + 1):
        t0 = time.time()
        sd = {k: v.cpu() for k, v in net.state_dict().items()}
        tasks = []
        if args.mode == 2:
            for i in range(args.games):
                tasks.append((sd, d, 2, 30000 + it * 10000 + i, args.n_sims))
        else:
            half = args.games // 2
            for i in range(half):
                tasks.append((sd, d, 2, 30000 + it * 10000 + i, args.n_sims))
            for i in range(args.games - half):
                tasks.append((sd, d, 4, 60000 + it * 10000 + i, args.n_sims))

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(play_game, tasks))
        new = [s for game in results for s in game]
        buffer.extend(new)
        if len(buffer) > args.buffer:
            buffer = buffer[-args.buffer:]

        PF, GF, POL, MASK, VAL = _stack(buffer)
        stats = train_net(net, PF, GF, POL, MASK, VAL,
                          epochs=args.epochs, lr=args.lr, device=dev)
        torch.save({"state_dict": net.state_dict(), "d": d, "iter": it},
                   CKPT)
        torch.save({"state_dict": net.state_dict(), "d": d, "iter": it},
                   f"analysis/v17_iter{it}.pt")
        pl, vl = stats[-1]
        print(f"[loop] iter {it}/{args.iterations}: {len(new)} new samples "
              f"(buffer {len(buffer)}) policy_ce={pl:.4f} value_mse={vl:.4f} "
              f"win_avg={VAL.mean():+.3f} ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
