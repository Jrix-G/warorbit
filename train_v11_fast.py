#!/usr/bin/env python3
"""ES training for V11 heuristic constants and linear mission scorer.

This trains the V11 knobs directly, not the V10 proxy model. Promotion is based
on fixed evaluation games, with a 4p-heavy default schedule.
"""

from __future__ import annotations

import argparse
import csv
import io
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SCORER_DIM = 15
SCORER_SCALE = 0.5

HEURISTIC_SPECS: List[Tuple[str, float, float]] = []

DIM = SCORER_DIM + len(HEURISTIC_SPECS)

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


def decode(params: np.ndarray):
    scorer_w = params[:SCORER_DIM].astype(np.float64) * SCORER_SCALE
    heur = {}
    for i, (name, base, scale) in enumerate(HEURISTIC_SPECS):
        p = max(-2.0, min(2.0, float(params[SCORER_DIM + i])))
        value = max(0.01, base + p * scale)
        heur[name] = value
    return scorer_w, heur


def _silent_imports():
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    try:
        import bot_v11  # noqa: F401
        from local_simulator.official_fast import run_fast_game  # noqa: F401
        from opponents import ZOO  # noqa: F401
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def _schedule(opponents: Sequence[str], total_games: int, four_player_ratio: float, seed_base: int):
    rng = np.random.RandomState(seed_base)
    pool = list(opponents)
    schedule = []
    cursor = 0
    rng.shuffle(pool)
    for i in range(total_games):
        use_4p = len(pool) >= 3 and rng.rand() < four_player_ratio
        if use_4p:
            if cursor + 3 > len(pool):
                rng.shuffle(pool)
                cursor = 0
            opps = tuple(pool[cursor:cursor + 3])
            cursor += 3
            our_idx = int(rng.randint(0, 4))
        else:
            if cursor >= len(pool):
                rng.shuffle(pool)
                cursor = 0
            opps = (pool[cursor],)
            cursor += 1
            our_idx = int(rng.randint(0, 2))
        schedule.append((opps, our_idx, seed_base + i))
    return schedule


def _worker(args):
    params, opp_names, our_idx, seed, max_steps, overage = args
    try:
        _silent_imports()
        import bot_v11
        from local_simulator.official_fast import run_fast_game
        from opponents import ZOO

        scorer_w, heur = decode(np.asarray(params, dtype=np.float32))
        bot_v11.reset_heuristic_params()
        bot_v11.set_heuristic_params(heur)
        bot_v11.set_scorer(lambda feat: float(scorer_w @ feat.astype(np.float64)), noise_std=0.0, log_player=-1)
        bot_v11.reset_episode_log()

        opp_agents = [ZOO[name] for name in opp_names if name in ZOO]
        if len(opp_agents) != len(opp_names):
            return 0.0
        n_players = len(opp_agents) + 1
        agents = []
        opp_iter = iter(opp_agents)
        for slot in range(n_players):
            agents.append(bot_v11.agent if slot == our_idx else next(opp_iter))
        result = run_fast_game(agents, seed=seed, n_players=n_players, max_steps=max_steps, overage_time=overage, use_c_accel=True)
        return 1.0 if int(result.get("winner", -1)) == our_idx else 0.0
    except Exception:
        return 0.0
    finally:
        try:
            import bot_v11
            bot_v11.set_scorer(None)
            bot_v11.reset_heuristic_params()
        except Exception:
            pass


def evaluate(pool, params, opponents, games, four_player_ratio, seed_base, max_steps, overage):
    sched = _schedule(opponents, games, four_player_ratio, seed_base)
    tasks = [(params, opps, our_idx, seed, max_steps, overage) for opps, our_idx, seed in sched]
    results = pool.map(_worker, tasks)
    return float(np.mean(results)) if results else 0.0, int(sum(results)), len(results)


def es_generation(pool, params, sigma, opponents, pairs, games_per_eval, four_player_ratio, seed_base, max_steps, overage):
    rng = np.random.RandomState((seed_base * 1103515245 + 12345) & 0x7fffffff)
    eps = [rng.randn(DIM).astype(np.float32) for _ in range(pairs)]
    sched = _schedule(opponents, pairs * games_per_eval, four_player_ratio, seed_base)
    tasks = []
    for i, e in enumerate(eps):
        for g in range(games_per_eval):
            opps, our_idx, seed = sched[i * games_per_eval + g]
            tasks.append(((params + sigma * e).astype(np.float32), opps, our_idx, seed, max_steps, overage))
            tasks.append(((params - sigma * e).astype(np.float32), opps, our_idx, seed, max_steps, overage))
    raw = np.asarray(pool.map(_worker, tasks), dtype=np.float64)
    pos = np.zeros(pairs)
    neg = np.zeros(pairs)
    for i in range(pairs):
        for g in range(games_per_eval):
            base = i * (2 * games_per_eval) + g * 2
            pos[i] += raw[base]
            neg[i] += raw[base + 1]
    pos /= max(1, games_per_eval)
    neg /= max(1, games_per_eval)

    all_r = np.concatenate([pos, neg])
    ranks = (all_r + rng.randn(len(all_r)) * 1e-6).argsort().argsort().astype(np.float64)
    shaped = ranks / max(1, len(ranks) - 1) - 0.5
    update = np.zeros(DIM, dtype=np.float64)
    for i, e in enumerate(eps):
        update += (shaped[i] - shaped[pairs + i]) * e.astype(np.float64)
    update /= max(1e-9, pairs * sigma)
    return update, float(raw.mean())


def _save(path: Path, params, momentum, generation, wr):
    np.savez(str(path), params=params.astype(np.float32), momentum=momentum.astype(np.float64),
             generation=np.array(generation, dtype=np.int32), wr=np.array(wr, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 4))
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--games-per-eval", type=int, default=2)
    parser.add_argument("--eval-games", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--match-4p-ratio", type=float, default=0.90)
    parser.add_argument("--eval-4p-ratio", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--overage", type=float, default=60.0)
    parser.add_argument("--lr", type=float, default=0.045)
    parser.add_argument("--momentum", type=float, default=0.90)
    parser.add_argument("--sigma-init", type=float, default=0.28)
    parser.add_argument("--sigma-min", type=float, default=0.06)
    parser.add_argument("--out", default="evaluations/scorer_v11_kaggle")
    parser.add_argument("--load", default=None)
    parser.add_argument("--skip-baseline-eval", action="store_true")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    _silent_imports()
    from opponents import ZOO

    opponents = [name for name in NOTEBOOK_OPPONENTS if name in ZOO]
    if not opponents:
        raise SystemExit("No notebook opponents found.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    latest = out.with_name(out.name + "_latest.npz")
    best = out.with_suffix(".npz")
    csv_path = out.with_suffix(".csv")

    if args.load and Path(args.load).exists():
        ckpt = np.load(args.load)
        params = ckpt["params"].astype(np.float32)
        mom = ckpt["momentum"].astype(np.float64) if "momentum" in ckpt else np.zeros(DIM, dtype=np.float64)
        generation = int(ckpt["generation"]) if "generation" in ckpt else 0
        best_wr = float(ckpt["wr"]) if "wr" in ckpt else -1.0
        print(f"Resumed {args.load} gen={generation} wr={best_wr:.3f}", flush=True)
    else:
        params = np.zeros(DIM, dtype=np.float32)
        mom = np.zeros(DIM, dtype=np.float64)
        generation = 0
        best_wr = -1.0

    if not csv_path.exists() or not args.load:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["gen", "elapsed_s", "sigma", "avg_r", "eval_wr", "best_wr", "param_norm"])

    print(
        f"V11 ES train | minutes={args.minutes:.1f} workers={args.workers} pairs={args.pairs} "
        f"games_per_eval={args.games_per_eval} eval_games={args.eval_games} 4p={args.match_4p_ratio:.2f}",
        flush=True,
    )
    print(f"opponents={len(opponents)} dim={DIM} out={best}", flush=True)

    deadline = time.time() + args.minutes * 60.0
    started = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        if not args.skip_baseline_eval:
            wr, w, n = evaluate(pool, params, opponents, args.eval_games, args.eval_4p_ratio, 50000, args.max_steps, args.overage)
            best_wr = wr
            _save(best, params, mom, generation, best_wr)
            print(f"baseline eval={wr:.3f} ({w}/{n})", flush=True)

        while time.time() < deadline:
            elapsed_min = (time.time() - started) / 60.0
            sigma = max(args.sigma_min, args.sigma_init * (0.5 ** (elapsed_min / 45.0)))
            update, avg_r = es_generation(
                pool, params, sigma, opponents, args.pairs, args.games_per_eval,
                args.match_4p_ratio, generation * 1009 + 17, args.max_steps, args.overage,
            )
            mom = args.momentum * mom + (1.0 - args.momentum) * update
            params = (params + args.lr * mom).astype(np.float32)
            norm = float(np.linalg.norm(params))
            if norm > 7.0:
                params = (params * (7.0 / norm)).astype(np.float32)
                mom = np.zeros_like(mom)
                norm = 7.0
            generation += 1

            eval_wr = ""
            if generation % args.eval_every == 0:
                wr, w, n = evaluate(
                    pool, params, opponents, args.eval_games, args.eval_4p_ratio,
                    60000 + generation * 31, args.max_steps, args.overage,
                )
                eval_wr = f"{wr:.4f}"
                if wr >= best_wr:
                    best_wr = wr
                    _save(best, params, mom, generation, best_wr)
                    mark = " *"
                else:
                    mark = ""
                print(f"gen={generation:04d} avg_r={avg_r:.3f} eval={wr:.3f} ({w}/{n}) best={best_wr:.3f} sigma={sigma:.3f} norm={norm:.2f}{mark}", flush=True)
            else:
                print(f"gen={generation:04d} avg_r={avg_r:.3f} best={best_wr:.3f} sigma={sigma:.3f} norm={norm:.2f}", flush=True)

            _save(latest, params, mom, generation, best_wr)
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([generation, f"{time.time() - started:.1f}", f"{sigma:.4f}",
                                        f"{avg_r:.4f}", eval_wr, f"{best_wr:.4f}", f"{norm:.4f}"])

    scorer, heur = decode(params)
    np.save(out.with_name(out.name + "_scorer.npy"), scorer.astype(np.float32))
    with open(out.with_name(out.name + "_heur.txt"), "w") as f:
        for key, value in heur.items():
            f.write(f"{key} = {value:.4f}\n")
    print(f"Done gen={generation} best={best_wr:.3f} saved={best} latest={latest}", flush=True)


if __name__ == "__main__":
    main()
