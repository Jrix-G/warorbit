"""wo_league — generate self-play games (V15 vs V15) for residual net training.

Plays N games of raw V15 vs itself, saves every (step, player) sample as an
.npz compatible with wo_dataset / wo_train.  The distribution of visited states
matches what V15's search actually evaluates — fixing the training-distribution
mismatch that limits the value net trained only on top-1 human replays.

Run:
    python -u wo_league.py --games 100 --out analysis/wo_league_data.npz
    python -u wo_league.py --games 100 --modes 2,4 --workers 11 --out analysis/wo_league_data.npz
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import v14_core
import v15_eval
import v15_fast_sim as fsim
import v15_search
import v17_encode as enc
from local_simulator.official_fast import OfficialFastGame
from wo_dataset import N_MAX, _pad, _policy_target, _episode_values

_BUDGET = 0.5


def _init(budget):
    global _BUDGET
    import torch
    torch.set_num_threads(1)
    _BUDGET = budget


def _play_game(task):
    """Play one full V15 vs V15 game, return list of (pf,gf,mask,val,esc) rows."""
    n_players, seed = task
    episode_steps = 250
    g = OfficialFastGame(n_players, seed=seed, episode_steps=episode_steps,
                         use_c_accel=False)
    fs = fsim.from_obs(v14_core.obs_as_dict(g.observation(0)),
                       n_players=n_players, episode_steps=episode_steps)
    fs.n_players = n_players

    # record (state_snapshot, actions) at each step for post-hoc value labeling
    history = []        # list of (fs_snapshot, actions_list)
    while not fs.done:
        actions = []
        for p in range(n_players):
            o = v15_search.state_to_obs(fs, p)
            m = v15_search.search(o, None, time_budget=_BUDGET)
            actions.append(m if isinstance(m, list) else [])
        history.append((fs, actions))
        fs = fsim.step(fs, actions)

    # final scores -> value labels
    sc = fsim.scores(fs)
    best = max(sc)
    win_set = [i for i, s in enumerate(sc) if s == best and best > 0]
    vals = [1.0 if (len(win_set) == 1 and p in win_set) else -1.0
            for p in range(n_players)]

    rows = []
    for t, (state, acts) in enumerate(history):
        if len(state.planets) == 0:
            continue
        for p in range(n_players):
            pf, gf = enc.encode(state, p)
            n = pf.shape[0]
            # build policy target from actual action taken
            targets = enc.action_to_targets(state, p, acts[p])
            pol = _policy_target(state, p, targets)
            pfp, polp, mask = _pad(pf, pol)
            esc = v15_eval.evaluate(state, p, v15_eval.ESC)
            rows.append((pfp, gf.astype(np.float32), mask, float(vals[p]), float(esc)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--modes", default="2", help="player counts, comma-separated")
    ap.add_argument("--out", default="analysis/wo_league_data.npz")
    ap.add_argument("--budget", type=float, default=0.5,
                    help="per-move time budget for V15 (seconds)")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--seed-offset", type=int, default=9_000_000)
    ap.add_argument("--save-every", type=int, default=0,
                    help="flush intermediate .npz every N games (0=off)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    modes = [int(x) for x in args.modes.split(",") if x.strip()]
    tasks = [(npl, args.seed_offset + i)
             for npl in modes
             for i in range(args.games)]

    PF, GF, MASK, VAL, ESC_ARR, EP = [], [], [], [], [], []
    ep_idx = 0
    done_games = 0
    total_samples = 0
    t0 = time.time()

    def _flush(path):
        if not PF:
            return
        np.savez_compressed(path,
                            PF=np.stack(PF), GF=np.stack(GF), MASK=np.stack(MASK),
                            VAL=np.array(VAL, dtype=np.float32),
                            ESC=np.array(ESC_ARR, dtype=np.float32),
                            EP=np.array(EP, dtype=np.int32))
        print(f"[league] flushed {len(PF)} samples -> {path}", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.budget,)) as pool:
        futs = {pool.submit(_play_game, t): t for t in tasks}
        for fut in as_completed(futs):
            npl, seed = futs[fut]
            rows = fut.result()
            for pfp, gf, mask, val, esc in rows:
                PF.append(pfp); GF.append(gf); MASK.append(mask)
                VAL.append(val); ESC_ARR.append(esc); EP.append(ep_idx)
            ep_idx += 1
            done_games += 1
            total_samples += len(rows)
            elapsed = time.time() - t0
            print(f"[league] game {done_games}/{len(tasks)}  "
                  f"{npl}p seed={seed}  +{len(rows)} samples  "
                  f"total={total_samples}  {elapsed:.0f}s", flush=True)
            if args.save_every and done_games % args.save_every == 0:
                stem, ext = os.path.splitext(args.out)
                _flush(f"{stem}_ckpt{done_games}{ext}")

    PF = np.stack(PF)
    GF = np.stack(GF)
    MASK = np.stack(MASK)
    VAL = np.array(VAL, dtype=np.float32)
    ESC_ARR = np.array(ESC_ARR, dtype=np.float32)
    EP = np.array(EP, dtype=np.int32)
    np.savez_compressed(args.out, PF=PF, GF=GF, MASK=MASK,
                        VAL=VAL, ESC=ESC_ARR, EP=EP)
    elapsed = time.time() - t0
    print(f"\n[league] {ep_idx} games -> {len(PF)} samples in {elapsed/60:.1f} min")
    print(f"  VAL mean={VAL.mean():+.3f}  ESC mean={ESC_ARR.mean():.3f}")
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
