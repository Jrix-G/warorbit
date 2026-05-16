"""V15.4 SP3 — one generation of the self-play value-iteration loop.

For generation N:
  1. GENERATE  — play `--games` RCC self-play games with generation N-1's
                 weights; collect labelled position samples.
  2. TRAIN     — fit a value function on those samples -> analysis/vf_genN.npz
  3. BENCHMARK — paired head-to-head: RCC+genN vs RCC+gen(N-1). Antithetic
                 (sides/slots swapped on the same map) to cancel map luck.
  4. VERDICT   — kill-switch: report whether genN significantly beats gen N-1.

Generation 0 weights = the hand-tuned ESC (v15_eval.ESC).

Run:
    KMP_DUPLICATE_LIB_OK=TRUE python -u run_generation.py --gen 1 \
        --games 300 --bench 60 --workers 8 --episode-steps 500
"""

from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import v15_eval
from train_vf_selfplay import _fit
from v15_selfplay import play_game

ANALYSIS = "analysis"


def _wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (p, c - m, c + m)


def _prev_weights(gen: int) -> v15_eval.EvalWeights:
    if gen <= 1:
        return v15_eval.ESC
    path = os.path.join(ANALYSIS, f"vf_gen{gen-1}.npz")
    return v15_eval.EvalWeights.load(path)


# --- generation workers -----------------------------------------------------

def _gen_worker(task):
    n_players, seed, weights, episode_steps = task
    samples, _ = play_game(n_players, seed, [weights] * n_players,
                           episode_steps=episode_steps)
    return samples


def _bench_worker(task):
    """Play one paired benchmark game. weight_ids[p] in {N, P}; returns the
    list of slot owners that won (so the caller can tally N vs P)."""
    n_players, seed, slot_is_new, wN, wP, episode_steps = task
    wbp = [wN if slot_is_new[p] else wP for p in range(n_players)]
    _, sc = play_game(n_players, seed, wbp, episode_steps=episode_steps)
    best = max(sc) if sc else 0
    winners = [p for p in range(n_players) if sc[p] == best and best > 0]
    if len(winners) != 1:
        return None                       # tie / dead game -> no count
    return slot_is_new[winners[0]]        # True if genN won


# --- stages -----------------------------------------------------------------

def generate(gen, games, workers, episode_steps):
    prev = _prev_weights(gen)
    tasks = []
    half = games // 2
    for i in range(half):
        tasks.append((2, 600000 + gen * 100000 + i, prev, episode_steps))
    for i in range(games - half):
        tasks.append((4, 700000 + gen * 100000 + i, prev, episode_steps))

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_gen_worker, tasks))
    dt = time.time() - t0

    nps, X, y = [], [], []
    for samples in results:
        for (n, feat, win) in samples:
            nps.append(n)
            X.append(feat)
            y.append(win)
    nps = np.array(nps, dtype=np.int64)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    out = os.path.join(ANALYSIS, f"sp_gen{gen}.npz")
    np.savez(out, n_players=nps, X=X, y=y)
    print(f"[generate] {games} games in {dt/60:.1f} min -> "
          f"{len(y)} samples ({out})")
    return out


def train(gen, samples_path):
    d = np.load(samples_path, allow_pickle=True)
    nps, X, y = d["n_players"], d["X"], d["y"]
    esc = v15_eval.ESC
    res = {}
    for mode, npv in (("2p", 2), ("4p", 4)):
        m = nps == npv
        if m.sum() < 100:
            print(f"[train] {mode}: {int(m.sum())} samples — keep ESC")
            res[mode] = None
            continue
        w, b, mean, std, auc = _fit(X[m], y[m])
        print(f"[train] {mode}: n={int(m.sum())} "
              f"pos={y[m].mean():.3f} val_AUC={auc:.4f}")
        names = ["ship", "prod", "planet", "domin", "pmarg", "fleet",
                 "elim", "topprod", "conc", "step", "efleet"]
        top = sorted(zip(names, w), key=lambda kv: -abs(kv[1]))[:5]
        print("    top: " + "  ".join(f"{nm}{wv:+.2f}" for nm, wv in top))
        res[mode] = (w, mean, std)
    w2, m2, s2 = res["2p"] or (esc.w2p, esc.mean2p, esc.std2p)
    w4, m4, s4 = res["4p"] or (esc.w4p, esc.mean4p, esc.std4p)
    ew = v15_eval.EvalWeights(w2p=w2, w4p=w4, mean2p=m2, std2p=s2,
                              mean4p=m4, std4p=s4, tag=f"vf_gen{gen}")
    out = os.path.join(ANALYSIS, f"vf_gen{gen}.npz")
    ew.save(out)
    print(f"[train] -> {out}")
    return ew


def benchmark(gen, wN, wP, bench, workers, episode_steps):
    tasks = []
    base = 800000 + gen * 100000
    # 2p paired: genN in slot0, then swapped
    for i in range(bench):
        s = base + i
        tasks.append((2, s, [True, False], wN, wP, episode_steps))
        tasks.append((2, s, [False, True], wN, wP, episode_steps))
    # 4p 2v2 paired: genN in slots {0,1}, then {2,3}
    for i in range(bench):
        s = base + 50000 + i
        tasks.append((4, s, [True, True, False, False], wN, wP, episode_steps))
        tasks.append((4, s, [False, False, True, True], wN, wP, episode_steps))

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(_bench_worker, tasks))
    dt = time.time() - t0

    out = {}
    for mode, npv in (("2p", 2), ("4p", 4)):
        sub = [r for r, t in zip(res, tasks) if t[0] == npv and r is not None]
        w = sum(1 for r in sub if r)
        n = len(sub)
        p, lo, hi = _wilson(w, n)
        out[mode] = (w, n, p, lo, hi)
    print(f"[benchmark] {dt/60:.1f} min")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--bench", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--episode-steps", type=int, default=500)
    args = ap.parse_args()

    os.makedirs(ANALYSIS, exist_ok=True)
    print(f"=== Generation {args.gen} "
          f"(prev = {'ESC' if args.gen <= 1 else f'vf_gen{args.gen-1}'}) ===")

    samples_path = generate(args.gen, args.games, args.workers,
                            args.episode_steps)
    wN = train(args.gen, samples_path)
    wP = _prev_weights(args.gen)
    res = benchmark(args.gen, wN, wP, args.bench, args.workers,
                    args.episode_steps)

    print(f"\n=== VERDICT — gen{args.gen} vs "
          f"{'ESC' if args.gen <= 1 else f'gen{args.gen-1}'} ===")
    agg_w = agg_n = 0
    for mode in ("2p", "4p"):
        w, n, p, lo, hi = res[mode]
        agg_w += w
        agg_n += n
        print(f"  {mode}: genN W={w}/{n} WR={p:.3f} 95%CI=[{lo:.3f},{hi:.3f}]")
    p, lo, hi = _wilson(agg_w, agg_n)
    verdict = ("IMPROVEMENT — keep going" if lo > 0.5 else
               "REGRESSION — stop, diagnose" if hi < 0.5 else
               "INCONCLUSIVE — more games or kill-switch")
    print(f"  AGG: W={agg_w}/{agg_n} WR={p:.3f} 95%CI=[{lo:.3f},{hi:.3f}]")
    print(f"  -> {verdict}")

    with open(os.path.join(ANALYSIS, f"GEN{args.gen}_RESULT.txt"), "w") as f:
        f.write(f"gen{args.gen}: AGG W={agg_w}/{agg_n} WR={p:.3f} "
                f"CI=[{lo:.3f},{hi:.3f}] -> {verdict}\n")


if __name__ == "__main__":
    main()
