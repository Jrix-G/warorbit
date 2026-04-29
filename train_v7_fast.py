"""ES training of bot_v7 — unified scorer + heuristic constants.

Design (post-mortem of previous failed runs):
  - Antithetic sampling with SHARED opp+seeds → variance reduction is real.
  - Rank-shaped fitness (Wierstra 2014, OpenAI ES 2017) → robust to outliers.
  - Adam-lite momentum on ES update → smoother convergence.
  - Per-parameter (base, scale) decoding → ES sees N(0,1) for everything.
  - Sigma annealing (exponential decay).
  - All evaluations parallelised through the worker pool.
  - Best checkpoint saved on REAL fixed-game eval only — never on noisy avg_r.
  - CSV logging of every generation for post-hoc analysis.

Usage:
    python train_v7_fast.py --minutes 20            # sanity test
    python train_v7_fast.py --minutes 840 --workers 8   # overnight
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ─── Parameter space ────────────────────────────────────────────────────────
# (name, base, scale, kind)
# Each ES parameter is decoded as: actual = base + clip(p, -2, 2) * scale
# kind="scorer" → entry of the linear log-multiplier (uses scale only)
# kind="heur"   → overrides bot_v7 module attribute

SCORER_DIM = 15  # bot_v7.MISSION_FEATURE_DIM
SCORER_SCALE = 0.5  # log-mult uses 0.5 * w @ features (kept in stable range)

HEURISTIC_SPECS: List[Tuple[str, float, float]] = [
    # name, base value, ES scale (one σ ≈ this much movement)
    ("HOSTILE_MARGIN_BASE",                3.0,  1.5),
    ("HOSTILE_MARGIN_CAP",                12.0,  4.0),
    ("HOSTILE_MARGIN_PROD_WEIGHT",         2.0,  1.0),
    ("NEUTRAL_MARGIN_BASE",                2.0,  1.0),
    ("TWO_PLAYER_HOSTILE_AGGRESSION_BOOST", 1.35, 0.30),
    ("ATTACK_COST_TURN_WEIGHT",            0.55, 0.20),
    ("STATIC_TARGET_MARGIN",               4.0,  1.5),
]

DIM = SCORER_DIM + len(HEURISTIC_SPECS)


def decode(params: np.ndarray):
    """Map ES vector → (scorer_w_15, heuristic_dict)."""
    scorer_w = params[:SCORER_DIM].astype(np.float64) * SCORER_SCALE
    heur = {}
    for i, (name, base, scale) in enumerate(HEURISTIC_SPECS):
        p = float(params[SCORER_DIM + i])
        p = max(-2.0, min(2.0, p))
        val = base + p * scale
        heur[name] = max(0.01, val)  # never go negative/zero
    return scorer_w, heur


# ─── Opponent pool ──────────────────────────────────────────────────────────

NOTEBOOK_OPPONENTS = [
    "notebook_orbitbotnext",
    "notebook_distance_prioritized",
    "notebook_physics_accurate",
    "notebook_tactical_heuristic",
    "notebook_debugendless_orbit_wars_sun_dodging_baseline",
    "notebook_djenkivanov_orbit_wars_optimized_nearest_planet_sniper",
    "notebook_johnjanson_lb_max_score_1000_agi_is_here",
    "notebook_mdmahfuzsumon_how_my_ai_wins_space_wars",
    "notebook_pascalledesma_orbitbotnext",
    "notebook_pascalledesma_orbitwork_v14",
    "notebook_romantamrazov_orbit_star_wars_lb_max_1224",
    "notebook_sigmaborov_lb_928_7_physics_accurate_planner",
    "notebook_sigmaborov_lb_958_1_orbit_wars_2026_reinforce",
    "notebook_sigmaborov_orbit_wars_2026_starter",
    "notebook_sigmaborov_orbit_wars_2026_tactical_heuristic",
]

# Fixed eval set — uses ALL training opponents (broader signal than a subset).
# Trade-off: more games = slower eval, but lower variance.
EVAL_OPPONENTS = NOTEBOOK_OPPONENTS  # all 15
EVAL_GAMES_PER_OPP = 4  # kept as the "full" fixed eval size
MATCH_4P_RATIO = 0.70


# ─── Worker (subprocess) ────────────────────────────────────────────────────

def _silenced_imports():
    """Import bot_v7 / SimGame / ZOO with stdout silenced (notebooks chatter)."""
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    try:
        import bot_v7  # noqa: F401
        from SimGame import run_match  # noqa: F401
        from opponents import ZOO  # noqa: F401
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def _build_match_schedule(opponents, total_games, four_player_ratio, seed_base):
    """Build a deterministic mix of 2p and 4p matches.

    2p matches use one opponent.
    4p matches use three opponents plus our agent.
    """
    rng = np.random.RandomState(seed_base)
    pool = list(opponents)
    if not pool:
        return []

    shuffled = pool[:]
    rng.shuffle(shuffled)
    cursor = 0
    schedule = []
    for i in range(total_games):
        use_4p = len(shuffled) >= 3 and rng.rand() < float(four_player_ratio)
        if use_4p:
            if cursor + 3 > len(shuffled):
                rng.shuffle(shuffled)
                cursor = 0
            opp_names = tuple(shuffled[cursor:cursor + 3])
            cursor += 3
            our_index = int(rng.randint(0, 4))
            mode = "4p"
        else:
            if cursor >= len(shuffled):
                rng.shuffle(shuffled)
                cursor = 0
            opp_names = (shuffled[cursor],)
            cursor += 1
            our_index = int(rng.randint(0, 2))
            mode = "2p"
        schedule.append((mode, opp_names, our_index, seed_base + i))
    return schedule


def _eval_worker(args):
    """Play one game with a given parameter vector. Returns +1 win / 0 loss/draw.

    args = (params_array, opp_names, seed, our_index, overage_time)
    our_index ∈ {0,1,2,3} — which player slot we take (alternation handled by caller).
    """
    try:
        params, opp_names, seed, our_index, overage_time = args

        _silenced_imports()
        import bot_v7
        from SimGame import run_match
        from opponents import ZOO

        opp_names = tuple(opp_names or ())
        if not opp_names:
            return 0.0

        opp_agents = []
        for name in opp_names:
            if name not in ZOO:
                return 0.0
            opp_agents.append(ZOO[name])

        scorer_w, heur = decode(np.asarray(params, dtype=np.float32))
        bot_v7.reset_heuristic_params()
        bot_v7.set_heuristic_params(heur)
        bot_v7.set_scorer(lambda f: float(scorer_w @ f.astype(np.float64)),
                          noise_std=0.0, log_player=-1)
        bot_v7.reset_episode_log()

        try:
            n_players = len(opp_agents) + 1
            if our_index < 0 or our_index >= n_players:
                our_index = 0
            agents = []
            opp_iter = iter(opp_agents)
            for slot in range(n_players):
                if slot == our_index:
                    agents.append(bot_v7.agent)
                else:
                    agents.append(next(opp_iter))
            r = run_match(agents, seed=seed, n_players=n_players, overage_time=overage_time)
            won = (r.get("winner") == our_index)
        except Exception:
            won = False
        finally:
            bot_v7.set_scorer(None)
            bot_v7.reset_heuristic_params()

        return 1.0 if won else 0.0
    except Exception:
        return 0.0  # never let a worker crash the whole training


# ─── Fixed evaluation ───────────────────────────────────────────────────────

def evaluate_fixed(pool, params, opponents, games_per_opp, overage_time, match_4p_ratio, seed_base=10000):
    """Evaluate a single param vector on a fixed game set. Returns (wr, summary_dict)."""
    total_games = len(opponents) * games_per_opp
    schedule = _build_match_schedule(opponents, total_games, match_4p_ratio, seed_base)
    tasks = []
    meta = []
    for mode, opp_names, our_index, seed in schedule:
        tasks.append((params, opp_names, seed, our_index, overage_time))
        meta.append(mode)

    results = pool.map(_eval_worker, tasks)
    per_mode = {}
    for mode, r in zip(meta, results):
        per_mode.setdefault(mode, []).append(int(r))
    summary = {mode: (sum(v), len(v)) for mode, v in per_mode.items()}
    total_w = sum(s[0] for s in summary.values())
    total_n = sum(s[1] for s in summary.values())
    return total_w / max(1, total_n), summary


# ─── ES generation ──────────────────────────────────────────────────────────

def es_generation(pool, params, sigma, opponents, n_pairs, games_per_eval,
                  overage_time, seed_counter, match_4p_ratio):
    """One generation of antithetic ES with rank-shaped fitness.

    Each antithetic pair (+eps, -eps) is evaluated on the SAME opponents and
    SAME seeds, so the difference (r+ - r-) is purely due to the perturbation.
    """
    rng = np.random.RandomState((seed_counter * 9176091 + int(time.time() * 1000)) & 0x7FFFFFFF)
    epsilons = [rng.randn(DIM).astype(np.float32) for _ in range(n_pairs)]
    schedule = _build_match_schedule(
        opponents,
        n_pairs * games_per_eval,
        match_4p_ratio,
        seed_counter * 101 + 7,
    )

    tasks = []
    for i, eps in enumerate(epsilons):
        # Each "eval" plays games_per_eval games — alternating our_index
        for g in range(games_per_eval):
            mode, opp_names, our_index, seed = schedule[i * games_per_eval + g]
            tasks.append(((params + sigma * eps).astype(np.float32),
                          opp_names, seed, our_index, overage_time))
            tasks.append(((params - sigma * eps).astype(np.float32),
                          opp_names, seed, our_index, overage_time))

    results = pool.map(_eval_worker, tasks)

    # Aggregate: 2 * n_pairs * games_per_eval results, ordered as
    # for i in pairs: for g in games: (+, -)
    raw_rewards = np.array(results, dtype=np.float64)

    # Reduce to per-perturbation reward (mean over games_per_eval)
    # Layout: idx = i * (2 * games_per_eval) + g * 2 + sign
    per_pert_pos = np.zeros(n_pairs)
    per_pert_neg = np.zeros(n_pairs)
    for i in range(n_pairs):
        for g in range(games_per_eval):
            base = i * (2 * games_per_eval) + g * 2
            per_pert_pos[i] += raw_rewards[base]
            per_pert_neg[i] += raw_rewards[base + 1]
    per_pert_pos /= games_per_eval
    per_pert_neg /= games_per_eval

    # Rank-shaped fitness over all 2N values.
    # Add tiny jitter so ties (very common with binary rewards) break uniformly
    # rather than by memory order — otherwise argsort introduces positional bias.
    all_r = np.concatenate([per_pert_pos, per_pert_neg])
    jitter = rng.randn(len(all_r)) * 1e-6
    ranks = (all_r + jitter).argsort().argsort().astype(np.float64)
    shaped = ranks / max(1, len(ranks) - 1) - 0.5  # in [-0.5, 0.5]
    shaped_pos = shaped[:n_pairs]
    shaped_neg = shaped[n_pairs:]

    # ES gradient estimate (antithetic)
    update = np.zeros(DIM, dtype=np.float64)
    for i, eps in enumerate(epsilons):
        update += (shaped_pos[i] - shaped_neg[i]) * eps.astype(np.float64)
    update /= (n_pairs * sigma)

    avg_raw = float(raw_rewards.mean())
    return update, avg_raw, raw_rewards


# ─── Main ───────────────────────────────────────────────────────────────────

def format_per_opp(summary):
    parts = []
    for k, (w, n) in summary.items():
        parts.append(f"{k.replace('notebook_','')[:18]}={w}/{n}")
    return "  ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--sigma-init", type=float, default=0.30)
    ap.add_argument("--sigma-min", type=float, default=0.05)
    ap.add_argument("--sigma-half-life-min", type=float, default=60.0,
                    help="Sigma halves every N minutes (annealing)")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--pairs", type=int, default=8,
                    help="Antithetic pairs per generation")
    ap.add_argument("--games-per-eval", type=int, default=2,
                    help="Games per perturbation eval (rank-shape across all)")
    ap.add_argument("--eval-games-per-opp", type=int, default=1,
                    help="Games per opponent for the fixed eval set")
    ap.add_argument("--match-4p-ratio", type=float, default=MATCH_4P_RATIO,
                    help="Fraction of matches to play as 4-player games")
    ap.add_argument("--eval-match-4p-ratio", type=float, default=None,
                    help="Override 4p ratio for fixed eval set (defaults to match-4p-ratio)")
    ap.add_argument("--overage", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=5,
                    help="Run fixed eval every N generations")
    ap.add_argument("--load", type=str, default=None,
                    help="Resume from .npz checkpoint")
    ap.add_argument("--out", type=str, default="evaluations/scorer_v7_unified")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_path.with_suffix(".csv")

    # ── Initial params ───────────────────────────────────────────────────
    if args.load and Path(args.load).exists():
        ckpt = np.load(args.load)
        params = ckpt["params"].astype(np.float32)
        momentum = ckpt["momentum"].astype(np.float64)
        gen_offset = int(ckpt.get("generation", 0))
        print(f"Resumed from {args.load}  gen={gen_offset}", flush=True)
    else:
        params = np.zeros(DIM, dtype=np.float32)
        # Warm-start scorer block from existing scorer_es.npy if available
        es_path = ROOT / "evaluations" / "scorer_es.npy"
        if es_path.exists():
            try:
                w_es = np.load(es_path).astype(np.float32)
                if w_es.shape == (SCORER_DIM,):
                    # Convert to ES-space: actual_w = SCORER_SCALE * params
                    params[:SCORER_DIM] = w_es / SCORER_SCALE
                    print(f"Warm-start scorer block from {es_path}", flush=True)
            except Exception:
                pass
        momentum = np.zeros(DIM, dtype=np.float64)
        gen_offset = 0

    # ── Discover opponents ──────────────────────────────────────────────
    print("Loading ZOO ...", flush=True)
    _silenced_imports()
    from opponents import ZOO

    available = [n for n in NOTEBOOK_OPPONENTS if n in ZOO]
    eval_opps = [n for n in EVAL_OPPONENTS if n in ZOO]
    print(f"Pool: {len(available)} opponents | Eval set: {len(eval_opps)} × {EVAL_GAMES_PER_OPP} games",
          flush=True)
    if not available or not eval_opps:
        print("No opponents available — aborting.")
        return

    # ── CSV header ──────────────────────────────────────────────────────
    if not csv_path.exists() or args.load is None:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gen", "elapsed_s", "sigma", "lr", "avg_raw_r",
                        "norm_params", "eval_wr", "eval_detail"])

    deadline = time.time() + args.minutes * 60.0
    t_start = time.time()
    best_score = -1.0
    best_params = params.copy()
    last_eval_wr = None
    last_eval_detail = ""
    generation = gen_offset
    eval_4p_ratio = args.match_4p_ratio if args.eval_match_4p_ratio is None else args.eval_match_4p_ratio

    print(f"\nES training | DIM={DIM} | {args.minutes:.0f}min | "
          f"{args.workers} workers | pairs={args.pairs} | "
          f"sigma={args.sigma_init}→{args.sigma_min} | lr={args.lr} momentum={args.momentum}",
          flush=True)
    print(f"Training mix: 4p_ratio={args.match_4p_ratio:.2f} | "
          f"Eval mix: 4p_ratio={eval_4p_ratio:.2f} | eval_games_per_opp={args.eval_games_per_opp}",
          flush=True)
    print(f"Out: {out_path}.npz   CSV: {csv_path}\n", flush=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:

        # ── Baseline eval ───────────────────────────────────────────────
        print("=== Baseline eval ===", flush=True)
        t0 = time.time()
        wr, summary = evaluate_fixed(pool, params, eval_opps,
                                     args.eval_games_per_opp, args.overage, eval_4p_ratio)
        last_eval_wr = wr
        last_eval_detail = format_per_opp(summary)
        print(f"WR={wr:.0%}  {last_eval_detail}  ({time.time()-t0:.0f}s)", flush=True)
        best_score = wr
        best_params = params.copy()
        no_improve_evals = 0
        sigma_boost = 1.0  # multiplied into sigma; reset+shrink on stagnation
        np.savez(str(out_path) + ".npz",
                 params=best_params, momentum=momentum, generation=generation,
                 wr=best_score)
        print(f"Initial checkpoint saved.\n", flush=True)

        # ── ES loop ─────────────────────────────────────────────────────
        while time.time() < deadline:
            elapsed_min = (time.time() - t_start) / 60.0

            # Sigma annealing (exponential) × ratchet boost
            decay = 0.5 ** (elapsed_min / max(1e-3, args.sigma_half_life_min))
            sigma = max(args.sigma_min, args.sigma_init * decay * sigma_boost)

            t_gen = time.time()
            update, avg_r, _ = es_generation(
                pool, params, sigma, available,
                args.pairs, args.games_per_eval, args.overage, generation, args.match_4p_ratio,
            )

            # Adam-lite momentum
            momentum = args.momentum * momentum + (1.0 - args.momentum) * update
            params = (params + args.lr * momentum).astype(np.float32)

            # Soft norm clip on ES space (each component already clipped at decode).
            # Reset momentum on clip — otherwise it keeps pushing into the wall.
            n = float(np.linalg.norm(params))
            if n > 6.0:
                params = (params * (6.0 / n)).astype(np.float32)
                momentum = np.zeros_like(momentum)

            generation += 1
            gen_t = time.time() - t_gen
            eval_wr_str = ""
            eval_detail = last_eval_detail

            # ── Periodic fixed eval ────────────────────────────────────
            if generation % args.eval_every == 0:
                t_e = time.time()
                wr, summary = evaluate_fixed(pool, params, eval_opps,
                                             args.eval_games_per_opp, args.overage, eval_4p_ratio)
                last_eval_wr = wr
                last_eval_detail = format_per_opp(summary)
                star = ""
                if wr > best_score:
                    best_score = wr
                    best_params = params.copy()
                    no_improve_evals = 0
                    sigma_boost = 1.0
                    np.savez(str(out_path) + ".npz",
                             params=best_params, momentum=momentum,
                             generation=generation, wr=best_score)
                    star = " ★"
                else:
                    no_improve_evals += 1
                    # Ratchet: after 4 stagnant evals, restart from best with
                    # reduced sigma to refine instead of drift.
                    if no_improve_evals >= 4:
                        params = best_params.copy()
                        momentum = np.zeros_like(momentum)
                        sigma_boost *= 0.7
                        no_improve_evals = 0
                        star = " ↻ratchet"
                print(f"gen {generation:4d} | σ={sigma:.3f} | avg_r={avg_r:+.3f} | "
                      f"|p|={np.linalg.norm(params):.2f} | gen_t={gen_t:.0f}s | "
                      f"EVAL={wr:.0%} {last_eval_detail} ({time.time()-t_e:.0f}s){star}",
                      flush=True)
                eval_wr_str = f"{wr:.4f}"
                eval_detail = last_eval_detail
            else:
                print(f"gen {generation:4d} | σ={sigma:.3f} | avg_r={avg_r:+.3f} | "
                      f"|p|={np.linalg.norm(params):.2f} | gen_t={gen_t:.0f}s",
                      flush=True)

            # CSV row
            with open(csv_path, "a", newline="") as f:
                cw = csv.writer(f)
                cw.writerow([generation, f"{time.time()-t_start:.1f}",
                             f"{sigma:.4f}", f"{args.lr:.4f}",
                             f"{avg_r:.4f}", f"{np.linalg.norm(params):.4f}",
                             eval_wr_str, eval_detail])

            # Always save "latest" checkpoint (separate from best)
            np.savez(str(out_path) + "_latest.npz",
                     params=params, momentum=momentum,
                     generation=generation, wr=last_eval_wr or 0.0)

            if time.time() >= deadline:
                break

        # ── Final eval ──────────────────────────────────────────────────
        print("\n=== Final eval (best params) ===", flush=True)
        t0 = time.time()
        wr, summary = evaluate_fixed(pool, best_params, eval_opps,
                                     args.eval_games_per_opp, args.overage, eval_4p_ratio)
        print(f"WR={wr:.0%}  {format_per_opp(summary)}  ({time.time()-t0:.0f}s)",
              flush=True)

    # Persist also human-readable scorer + heur dump
    scorer_w, heur = decode(best_params)
    np.save(str(out_path) + "_scorer.npy", scorer_w.astype(np.float32))
    with open(str(out_path) + "_heur.txt", "w") as f:
        for k, v in heur.items():
            f.write(f"{k} = {v:.4f}\n")

    print(f"\nDone. {generation} generations.", flush=True)
    print(f"Best WR: {best_score:.0%}", flush=True)
    print(f"Saved:", flush=True)
    print(f"  {out_path}.npz           (full ES state — for resume)", flush=True)
    print(f"  {out_path}_scorer.npy    (decoded scorer weights)", flush=True)
    print(f"  {out_path}_heur.txt      (decoded heuristic constants)", flush=True)
    print(f"  {csv_path}               (per-gen metrics)", flush=True)


if __name__ == "__main__":
    main()
