"""V15.4 SP6 — one generation of the self-play value-iteration loop, on GPU.

Same three stages as run_generation.py (generate -> train -> paired
benchmark -> kill-switch verdict), but games are played in big batches on
v15_gpu_sim + v15_gpu_search with torch.compile. Generation 0 weights = ESC.

Run:
    python -u run_generation_gpu.py --gen 1 --games 1200 --bench 96 --chunk 384
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

import v15_eval
from train_vf_selfplay import _fit
import v15_gpu_selfplay as sp

ANALYSIS = "analysis"


def _wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (p, c - m, c + m)


def _prev_weights(gen):
    if gen <= 1:
        return v15_eval.ESC
    return v15_eval.EvalWeights.load(os.path.join(ANALYSIS, f"vf_gen{gen-1}.npz"))


def _chunks(n_games, chunk):
    """Split n_games into chunk-sized pieces (last one padded up to chunk)."""
    out = []
    done = 0
    while done < n_games:
        out.append(min(chunk, n_games - done))
        done += chunk
    return out


def generate(gen, games, chunk):
    prev = _prev_weights(gen)
    nps, X, y = [], [], []
    t0 = time.time()
    for n_players, seed_base in ((2, 1_000_000 + gen * 100000),
                                 (4, 2_000_000 + gen * 100000)):
        half = games // 2
        off = seed_base
        for c, csz in enumerate(_chunks(half, chunk)):
            states = sp.initial_states(n_players, chunk, off)
            off += chunk
            samples, _ = sp.play_batch(states, [prev] * n_players,
                                       explore=0.20)
            # keep only the first csz games' samples (chunk may be padded)
            keep = csz * 10**9   # samples are not per-game ordered; keep all
            for (np_, feat, win) in samples:
                nps.append(np_)
                X.append(feat)
                y.append(win)
            print(f"  [{n_players}p chunk {c}] {len(samples)} samples "
                  f"({(time.time()-t0)/60:.1f} min)")
    nps = np.array(nps, dtype=np.int64)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    out = os.path.join(ANALYSIS, f"sp_gen{gen}.npz")
    np.savez(out, n_players=nps, X=X, y=y)
    print(f"[generate] {len(y)} samples in {(time.time()-t0)/60:.1f} min "
          f"-> {out}")
    return out


def train(gen, samples_path):
    d = np.load(samples_path, allow_pickle=True)
    nps, X, y = d["n_players"], d["X"], d["y"]
    esc = v15_eval.ESC
    res = {}
    names = ["ship", "prod", "planet", "domin", "pmarg", "fleet",
             "elim", "topprod", "conc", "step", "efleet"]
    for mode, npv in (("2p", 2), ("4p", 4)):
        m = nps == npv
        if m.sum() < 200:
            print(f"[train] {mode}: {int(m.sum())} samples — keep ESC")
            res[mode] = None
            continue
        w, b, mean, std, auc = _fit(X[m], y[m])
        top = sorted(zip(names, w), key=lambda kv: -abs(kv[1]))[:5]
        print(f"[train] {mode}: n={int(m.sum())} pos={y[m].mean():.3f} "
              f"val_AUC={auc:.4f} | "
              + " ".join(f"{nm}{wv:+.2f}" for nm, wv in top))
        res[mode] = (w, mean, std)
    w2, m2, s2 = res["2p"] or (esc.w2p, esc.mean2p, esc.std2p)
    w4, m4, s4 = res["4p"] or (esc.w4p, esc.mean4p, esc.std4p)
    ew = v15_eval.EvalWeights(w2p=w2, w4p=w4, mean2p=m2, std2p=s2,
                              mean4p=m4, std4p=s4, tag=f"vf_gen{gen}")
    out = os.path.join(ANALYSIS, f"vf_gen{gen}.npz")
    ew.save(out)
    print(f"[train] -> {out}")
    return ew


def _bench_mode(n_players, bench, chunk, wN, wP, seed_base):
    """Paired benchmark for one mode; returns (wins_genN, n)."""
    # arrangement A: genN in the first half of slots, P in the rest
    half = n_players // 2
    arrA = [wN] * half + [wP] * (n_players - half)
    arrB = [wP] * half + [wN] * (n_players - half)
    new_in_A = [True] * half + [False] * (n_players - half)
    wins = n = 0
    for arr, new_in in ((arrA, new_in_A), (arrB, [not x for x in new_in_A])):
        off = seed_base
        for csz in _chunks(bench, chunk):
            states = sp.initial_states(n_players, chunk, off)
            off += chunk
            _, sc = sp.play_batch(states, arr, collect=False)
            for b in range(csz):
                best = sc[b].max()
                winners = [p for p in range(n_players)
                           if sc[b, p] == best and best > 0]
                if len(winners) != 1:
                    continue
                n += 1
                if new_in[winners[0]]:
                    wins += 1
        seed_base += 100000
    return wins, n


def benchmark(gen, wN, wP, bench, chunk):
    t0 = time.time()
    res = {}
    for n_players in (2, 4):
        w, n = _bench_mode(n_players, bench, chunk, wN, wP,
                           5_000_000 + gen * 100000 + n_players * 10000)
        p, lo, hi = _wilson(w, n)
        res[f"{n_players}p"] = (w, n, p, lo, hi)
    print(f"[benchmark] {(time.time()-t0)/60:.1f} min")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("--games", type=int, default=1200)
    ap.add_argument("--bench", type=int, default=96)
    ap.add_argument("--chunk", type=int, default=384)
    args = ap.parse_args()

    os.makedirs(ANALYSIS, exist_ok=True)
    print(f"=== Generation {args.gen} (GPU) "
          f"prev={'ESC' if args.gen <= 1 else f'vf_gen{args.gen-1}'} ===")

    sp_path = generate(args.gen, args.games, args.chunk)
    wN = train(args.gen, sp_path)
    wP = _prev_weights(args.gen)
    res = benchmark(args.gen, wN, wP, args.bench, args.chunk)

    print(f"\n=== VERDICT gen{args.gen} vs "
          f"{'ESC' if args.gen <= 1 else f'gen{args.gen-1}'} ===")
    agg_w = agg_n = 0
    for mode in ("2p", "4p"):
        w, n, p, lo, hi = res[mode]
        agg_w += w
        agg_n += n
        print(f"  {mode}: genN W={w}/{n} WR={p:.3f} CI=[{lo:.3f},{hi:.3f}]")
    p, lo, hi = _wilson(agg_w, agg_n)
    verdict = ("IMPROVEMENT — keep going" if lo > 0.5 else
               "REGRESSION — stop, diagnose" if hi < 0.5 else
               "INCONCLUSIVE — more games")
    print(f"  AGG: W={agg_w}/{agg_n} WR={p:.3f} CI=[{lo:.3f},{hi:.3f}]")
    print(f"  -> {verdict}")
    with open(os.path.join(ANALYSIS, f"GEN{args.gen}_GPU_RESULT.txt"), "w") as f:
        f.write(f"gen{args.gen}: AGG W={agg_w}/{agg_n} WR={p:.3f} "
                f"CI=[{lo:.3f},{hi:.3f}] -> {verdict}\n")


if __name__ == "__main__":
    main()
