"""v16_es — Evolution-Strategies optimisation of the RCC evaluator.

The V15.4 self-play value-function loop failed by the optimizer's curse: a
win-PREDICTOR maximised inside an argmax search gets Goodharted. The fix is to
optimise the deployed pipeline's WIN-RATE directly — the objective IS the
metric, so there is nothing to Goodhart and (with elitism) no regression.

This module implements OpenAI-ES (Salimans et al. 2017) over the evaluator
weights:
  * antithetic Gaussian perturbations of theta,
  * fitness = win-rate of RCC(theta) vs a fixed opponent league, played on
    the GPU batched engine,
  * rank-based fitness shaping (robust to noisy win-rate / outliers),
  * gradient estimate -> Adam-free step,
  * the best theta on a held-out seed set is kept (never deploy worse).

Phase 1 (this file's default): the LINEAR evaluator, 11 weights for one mode.
A de-risking step — it proves the ES loop is correct and monotone before the
non-linear evaluator and the Standing-Conditioned-Risk search are added.
"""

from __future__ import annotations

# torch + numpy each ship an OpenMP runtime; allow the duplicate (Windows).
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time

import numpy as np

import v15_eval
import v15_gpu_selfplay as sp


def _weights_for(mode_np: int, w: np.ndarray) -> v15_eval.EvalWeights:
    """Build an EvalWeights whose `mode_np`-player linear weights are `w`
    (no standardisation); the other mode keeps ESC."""
    esc = v15_eval.ESC
    z = np.zeros(v15_eval.N_FEATURES)
    o = np.ones(v15_eval.N_FEATURES)
    if mode_np == 4:
        return v15_eval.EvalWeights(w2p=esc.w2p.copy(), w4p=w.copy(),
                                    mean2p=z.copy(), std2p=o.copy(),
                                    mean4p=z.copy(), std4p=o.copy(),
                                    tag="es")
    return v15_eval.EvalWeights(w2p=w.copy(), w4p=esc.w4p.copy(),
                                mean2p=z.copy(), std2p=o.copy(),
                                mean4p=z.copy(), std4p=o.copy(), tag="es")


def fitness(w: np.ndarray, mode_np: int, opponent: v15_eval.EvalWeights,
            n_games: int, seed: int, horizon: int,
            max_steps: int = 500) -> float:
    """Win-rate of RCC(w) against `opponent` filling the other seats.

    The candidate occupies seat 0; opponents fill 1..mode_np-1. A win is a
    strict 1st place. Seeds are fixed per call so two candidates are compared
    on the SAME maps (paired / common-random-numbers variance reduction)."""
    cand = _weights_for(mode_np, w)
    wbp = [cand] + [opponent] * (mode_np - 1)
    states = sp.initial_states(mode_np, n_games, seed)
    _, sc = sp.play_batch(states, wbp, collect=False, horizon=horizon,
                          max_steps=max_steps)
    # Tie-aware expected reward: the prize (+1) is split among k joint
    # leaders, so a k-way tie scores 1/k. Without this, identical
    # deterministic policies tie on symmetric maps and the signal collapses
    # (an all-ESC game would score ~0, not the correct 0.25).
    score = 0.0
    counted = 0
    for b in range(n_games):
        best = sc[b].max()
        if best <= 0:
            continue                       # dead game — no winner
        leaders = [p for p in range(mode_np) if sc[b, p] == best]
        counted += 1
        if 0 in leaders:
            score += 1.0 / len(leaders)
    return score / counted if counted else 0.0


def _rank_shape(F: np.ndarray) -> np.ndarray:
    """Centered rank transform in [-0.5, 0.5] — standard ES fitness shaping,
    robust to the scale and outliers of a noisy win-rate signal."""
    order = np.argsort(F)
    ranks = np.empty(len(F))
    ranks[order] = np.arange(len(F))
    return ranks / (len(F) - 1) - 0.5


def run_es(mode_np, generations, pop_pairs, sigma, lr, n_games,
           horizon, seed0, max_steps=500):
    """OpenAI-ES on the linear evaluator weights for `mode_np` players."""
    esc = v15_eval.ESC
    theta = (esc.w4p if mode_np == 4 else esc.w2p).astype(np.float64).copy()
    opponent = esc                          # league = {RCC(ESC)} for phase 1
    rng = np.random.default_rng(seed0)

    base = fitness(theta, mode_np, opponent, n_games, seed0, horizon,
                   max_steps)
    best_theta, best_fit = theta.copy(), base
    print(f"[es] {mode_np}p start: ESC fitness vs ESC = {base:.3f} "
          f"(n={n_games})")

    for g in range(1, generations + 1):
        t0 = time.time()
        eps = rng.standard_normal((pop_pairs, len(theta)))
        F = np.zeros(2 * pop_pairs)
        # common-random-numbers: every candidate this generation is scored
        # on the same maps, so differences reflect theta, not luck.
        eval_seed = seed0 + g * 100000
        for i in range(pop_pairs):
            F[i] = fitness(theta + sigma * eps[i], mode_np, opponent,
                           n_games, eval_seed, horizon, max_steps)
            F[pop_pairs + i] = fitness(theta - sigma * eps[i], mode_np,
                                       opponent, n_games, eval_seed, horizon,
                                       max_steps)
        shaped = _rank_shape(F)
        sp_plus, sp_minus = shaped[:pop_pairs], shaped[pop_pairs:]
        grad = ((sp_plus - sp_minus)[:, None] * eps).sum(axis=0) \
            / (pop_pairs * sigma)
        theta = theta + lr * grad

        fit = fitness(theta, mode_np, opponent, n_games, eval_seed, horizon,
                      max_steps)
        if fit > best_fit:
            best_fit, best_theta = fit, theta.copy()
        print(f"[es] gen {g}/{generations}: fitness={fit:.3f} "
              f"best={best_fit:.3f} ({(time.time()-t0)/60:.1f} min)")

    out = _weights_for(mode_np, best_theta)
    out.save(f"analysis/es_{mode_np}p.npz")
    print(f"[es] {mode_np}p done: best fitness {best_fit:.3f} "
          f"(ESC baseline {base:.3f}) -> analysis/es_{mode_np}p.npz")
    return best_theta, best_fit, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=int, default=4, choices=(2, 4))
    ap.add_argument("--generations", type=int, default=15)
    ap.add_argument("--pop-pairs", type=int, default=6)   # 2x = population
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-games", type=int, default=192)
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7_000_000)
    args = ap.parse_args()
    os.makedirs("analysis", exist_ok=True)
    run_es(args.mode, args.generations, args.pop_pairs, args.sigma,
           args.lr, args.n_games, args.horizon, args.seed, args.max_steps)


if __name__ == "__main__":
    main()
