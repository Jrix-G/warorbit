"""V15.4 SP6 — one generation of the self-play value-iteration loop, on GPU.

Stages: generate -> train -> paired benchmark -> kill-switch verdict.
Games run in big torch.compile'd batches on v15_gpu_sim + v15_gpu_search.
Generation 0 weights = ESC.

CHECKPOINT / RESUME
  Every chunk of games is written to analysis/gen<N>/ as soon as it finishes.
  Ctrl-C is safe: at worst the in-progress chunk is lost. Re-launching the
  SAME command skips every chunk already on disk and continues. The trained
  VF and each benchmark chunk are checkpointed the same way.

Run (resumable):
    python -u run_generation_gpu.py --gen 1 --games 1500 --bench 120 --chunk 384
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np

import v15_eval
from train_vf_selfplay import _fit
import v15_gpu_selfplay as sp

ANALYSIS = "analysis"
EXPLORE = 0.20            # self-play exploration for data generation


def _wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (p, c - m, c + m)


def _ckpt_dir(gen):
    d = os.path.join(ANALYSIS, f"gen{gen}")
    os.makedirs(d, exist_ok=True)
    return d


def _prev_weights(gen):
    if gen <= 1:
        return v15_eval.ESC
    return v15_eval.EvalWeights.load(
        os.path.join(ANALYSIS, f"vf_gen{gen-1}.npz"))


def _winner(sc_row, n_players):
    best = sc_row.max()
    w = [p for p in range(n_players) if sc_row[p] == best and best > 0]
    return w[0] if len(w) == 1 else -1


# --- stage 1: generate ------------------------------------------------------

def generate(gen, games, chunk):
    """Play self-play games in fixed-size chunks; checkpoint each chunk."""
    d = _ckpt_dir(gen)
    prev = _prev_weights(gen)
    half = games // 2
    t0 = time.time()
    for n_players, seed_base in ((2, 1_000_000 + gen * 100000),
                                 (4, 2_000_000 + gen * 100000)):
        n_chunks = max(1, math.ceil(half / chunk))
        for i in range(n_chunks):
            path = os.path.join(d, f"sp_{n_players}p_{i}.npz")
            if os.path.exists(path):
                print(f"  [resume] {n_players}p chunk {i}/{n_chunks} cached")
                continue
            states = sp.initial_states(n_players, chunk,
                                       seed_base + i * chunk)
            samples, _ = sp.play_batch(states, [prev] * n_players,
                                       explore=EXPLORE)
            nps = np.array([s[0] for s in samples], dtype=np.int64)
            X = np.array([s[1] for s in samples], dtype=np.float64)
            y = np.array([s[2] for s in samples], dtype=np.float64)
            tmp = path + ".tmp"
            np.savez(tmp, n_players=nps, X=X, y=y)
            os.replace(tmp, path)            # atomic — Ctrl-C safe
            print(f"  [{n_players}p chunk {i}/{n_chunks}] {len(y)} samples "
                  f"pos={y.mean():.3f} ({(time.time()-t0)/60:.1f} min)")
    return d


def _load_samples(d):
    nps, X, y = [], [], []
    for f in sorted(glob.glob(os.path.join(d, "sp_*.npz"))):
        dd = np.load(f)
        nps.append(dd["n_players"])
        X.append(dd["X"])
        y.append(dd["y"])
    return (np.concatenate(nps), np.concatenate(X), np.concatenate(y))


# --- stage 2: train ---------------------------------------------------------

def train(gen, d):
    out = os.path.join(ANALYSIS, f"vf_gen{gen}.npz")
    if os.path.exists(out):
        print(f"[train] {out} cached")
        return v15_eval.EvalWeights.load(out)
    nps, X, y = _load_samples(d)
    esc = v15_eval.ESC
    names = ["ship", "prod", "planet", "domin", "pmarg", "fleet",
             "elim", "topprod", "conc", "step", "efleet"]
    res = {}
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
    tmp = out + ".tmp"
    ew.save(tmp)
    os.replace(tmp, out)
    print(f"[train] -> {out}")
    return ew


# --- stage 3: paired benchmark ----------------------------------------------

def benchmark(gen, wN, wP, bench, chunk):
    """Antithetic paired benchmark; each chunk checkpointed in bench.json."""
    d = _ckpt_dir(gen)
    bpath = os.path.join(d, "bench.json")
    prog = json.load(open(bpath)) if os.path.exists(bpath) else {}
    t0 = time.time()

    for n_players in (2, 4):
        half = n_players // 2
        arr = {
            "A": [wN] * half + [wP] * (n_players - half),
            "B": [wP] * half + [wN] * (n_players - half),
        }
        new_in = {
            "A": [True] * half + [False] * (n_players - half),
            "B": [False] * half + [True] * (n_players - half),
        }
        n_chunks = max(1, math.ceil(bench / chunk))
        for an in ("A", "B"):
            for ci in range(n_chunks):
                key = f"{n_players}p_{an}_{ci}"
                if key in prog:
                    continue
                seed = (6_000_000 + gen * 100000 + n_players * 10000
                        + (0 if an == "A" else 5000) + ci * chunk)
                states = sp.initial_states(n_players, chunk, seed)
                _, sc = sp.play_batch(states, arr[an], collect=False)
                wins = n = 0
                for b in range(chunk):
                    win = _winner(sc[b], n_players)
                    if win < 0:
                        continue
                    n += 1
                    if new_in[an][win]:
                        wins += 1
                prog[key] = [wins, n]
                tmp = bpath + ".tmp"
                json.dump(prog, open(tmp, "w"))
                os.replace(tmp, bpath)
                print(f"  [bench {key}] genN {wins}/{n} "
                      f"({(time.time()-t0)/60:.1f} min)")

    res = {}
    for n_players in (2, 4):
        w = sum(prog[k][0] for k in prog if k.startswith(f"{n_players}p_"))
        n = sum(prog[k][1] for k in prog if k.startswith(f"{n_players}p_"))
        res[f"{n_players}p"] = (w, n) + _wilson(w, n)[1:]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("--games", type=int, default=1500)
    ap.add_argument("--bench", type=int, default=120)
    ap.add_argument("--chunk", type=int, default=384)
    args = ap.parse_args()

    os.makedirs(ANALYSIS, exist_ok=True)
    print(f"=== Generation {args.gen} (GPU) "
          f"prev={'ESC' if args.gen <= 1 else f'vf_gen{args.gen-1}'} ===")

    d = generate(args.gen, args.games, args.chunk)
    wN = train(args.gen, d)
    wP = _prev_weights(args.gen)
    res = benchmark(args.gen, wN, wP, args.bench, args.chunk)

    print(f"\n=== VERDICT gen{args.gen} vs "
          f"{'ESC' if args.gen <= 1 else f'gen{args.gen-1}'} ===")
    agg_w = agg_n = 0
    for mode in ("2p", "4p"):
        w, n, lo, hi = res[mode]
        agg_w += w
        agg_n += n
        p = w / n if n else 0.0
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
