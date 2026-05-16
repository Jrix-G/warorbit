"""v16_es — Evolution-Strategies training of the V16 (MLP + SCR) evaluator.

OpenAI-ES (Salimans et al. 2017) over the v16 parameter vector theta (MLP
weights + spread head + z_gain). Fitness = win-rate of the v16 search vs a
league, played on the GPU batched engine. The objective IS the metric, so
there is no Goodhart; elitism guarantees no regression (theta_0 = ESC).

Key properties:
  * fitness blends 2p and 4p so 2p cannot be sacrificed for 4p,
  * common random numbers — every candidate in a generation is scored on the
    SAME maps, so the rank signal ES needs is far cleaner than the absolute
    win-rate noise,
  * tie-aware scoring (a k-way tie = 1/k of the prize),
  * a league that grows with past champions (fictitious self-play) — rank-
    aware champions naturally gang up on the leader, which is the signal that
    teaches coalition-safe 4p play,
  * CHECKPOINT/RESUME — the ES state is saved every generation; re-launching
    the same command resumes (an ES run takes hours).

Run (resumable):
    python -u v16_es.py --generations 24 --hidden 6 --horizon 10
"""

from __future__ import annotations

# torch + numpy each ship an OpenMP runtime; allow the duplicate (Windows).
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import pickle
import time

import numpy as np

import v16_eval
from v16_selfplay import initial_states, play_batch_v16

CKPT = "analysis/v16_es_ckpt.pkl"
BEST = "analysis/v16_best.npy"
W_2P, W_4P = 0.40, 0.60        # fitness blend weight (4p is the weak mode)


def _tie_score(sc, n_players):
    """Tie-aware win-rate for seat 0: a k-way tie for 1st scores 1/k."""
    score = 0.0
    counted = 0
    for b in range(len(sc)):
        best = sc[b].max()
        if best <= 0:
            continue
        leaders = [p for p in range(n_players) if sc[b, p] == best]
        counted += 1
        if 0 in leaders:
            score += 1.0 / len(leaders)
    return score / counted if counted else 0.0


def fitness(theta, hidden, league, n2, n4, seed, horizon, max_steps):
    """Blended 2p+4p win-rate of v16(theta) vs the league.

    Candidate is seat 0; the other seats cycle through `league`. Seeds are
    fixed per call (common random numbers)."""
    res = {}
    for n_players, n_games, base in ((2, n2, 0), (4, n4, 500000)):
        if n_games <= 0:
            res[n_players] = 0.0
            continue
        opps = [league[i % len(league)] for i in range(n_players - 1)]
        tbp = [theta] + opps
        states = initial_states(n_players, n_games, seed + base)
        sc = play_batch_v16(states, tbp, hidden, horizon=horizon,
                            max_steps=max_steps)
        res[n_players] = _tie_score(sc, n_players)
    return W_2P * res[2] + W_4P * res[4], res[2], res[4]


def _rank_shape(F):
    """Centered rank transform in [-0.5, 0.5] — robust ES fitness shaping."""
    order = np.argsort(F)
    ranks = np.empty(len(F))
    ranks[order] = np.arange(len(F))
    return ranks / (len(F) - 1) - 0.5


def run_es(hidden, generations, pop_pairs, sigma, lr, n2, n4,
           horizon, max_steps, seed0, league_every):
    dim = v16_eval.n_params(hidden)

    # --- resume from checkpoint if present ---
    if os.path.exists(CKPT):
        with open(CKPT, "rb") as f:
            st = pickle.load(f)
        theta, best_theta, best_fit = st["theta"], st["best"], st["best_fit"]
        league, start_gen = st["league"], st["gen"] + 1
        rng = np.random.default_rng()
        rng.bit_generator.state = st["rng"]
        print(f"[es] resumed at generation {start_gen} (best {best_fit:.3f})")
    else:
        theta = v16_eval.initial_theta(hidden)        # == ESC
        league = [theta.copy()]                       # league starts with ESC
        rng = np.random.default_rng(seed0)
        f0, f2, f4 = fitness(theta, hidden, league, n2, n4, seed0,
                             horizon, max_steps)
        best_theta, best_fit, start_gen = theta.copy(), f0, 1
        print(f"[es] start: ESC fitness={f0:.3f} (2p={f2:.3f} 4p={f4:.3f}) "
              f"dim={dim}")

    for g in range(start_gen, generations + 1):
        t0 = time.time()
        eps = rng.standard_normal((pop_pairs, dim))
        F = np.zeros(2 * pop_pairs)
        eval_seed = seed0 + g * 100000               # CRN: same maps for all
        for i in range(pop_pairs):
            F[i] = fitness(theta + sigma * eps[i], hidden, league,
                           n2, n4, eval_seed, horizon, max_steps)[0]
            F[pop_pairs + i] = fitness(theta - sigma * eps[i], hidden, league,
                                       n2, n4, eval_seed, horizon,
                                       max_steps)[0]
        shaped = _rank_shape(F)
        grad = ((shaped[:pop_pairs] - shaped[pop_pairs:])[:, None] * eps
                ).sum(axis=0) / (pop_pairs * sigma)
        theta = theta + lr * grad

        f0, f2, f4 = fitness(theta, hidden, league, n2, n4, eval_seed,
                             horizon, max_steps)
        if f0 > best_fit:
            best_fit, best_theta = f0, theta.copy()
            np.save(BEST, best_theta)
        # fictitious self-play: periodically add the champion to the league
        if g % league_every == 0 and best_fit > 0:
            league.append(best_theta.copy())

        with open(CKPT, "wb") as f:
            pickle.dump({"theta": theta, "best": best_theta,
                         "best_fit": best_fit, "league": league, "gen": g,
                         "rng": rng.bit_generator.state}, f)
        print(f"[es] gen {g}/{generations}: fitness={f0:.3f} "
              f"(2p={f2:.3f} 4p={f4:.3f}) best={best_fit:.3f} "
              f"league={len(league)} ({(time.time()-t0)/60:.1f} min)")

    np.save(BEST, best_theta)
    print(f"[es] done: best fitness {best_fit:.3f} -> {BEST}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=6)
    ap.add_argument("--generations", type=int, default=24)
    ap.add_argument("--pop-pairs", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.06)
    ap.add_argument("--lr", type=float, default=0.04)
    ap.add_argument("--games-2p", type=int, default=48)
    ap.add_argument("--games-4p", type=int, default=80)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--league-every", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7_000_000)
    args = ap.parse_args()
    os.makedirs("analysis", exist_ok=True)
    run_es(args.hidden, args.generations, args.pop_pairs, args.sigma,
           args.lr, args.games_2p, args.games_4p, args.horizon,
           args.max_steps, args.seed, args.league_every)


if __name__ == "__main__":
    main()
