"""Kaggle-environments ES smoke trainer for bot_v7.

This is intentionally small: it mirrors the fast ES smoke loop, but runs games
through kaggle_environments.make("orbit_wars") instead of the local SimGame.
Use it to compare wall-clock speed and behavior against the local simulator.
"""

from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train_v7_fast import (  # noqa: E402
    DIM,
    EVAL_OPPONENTS,
    MATCH_4P_RATIO,
    NOTEBOOK_OPPONENTS,
    _build_match_schedule,
    decode,
)


def _silenced_imports():
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    try:
        import bot_v7  # noqa: F401
        from kaggle_environments import make  # noqa: F401
        from opponents import ZOO  # noqa: F401
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def _winner_from_env(env, n_players: int) -> int:
    final = env.steps[-1]
    rewards = [final[i].reward for i in range(n_players)]
    best = max(rewards)
    if rewards.count(best) != 1:
        return -1
    return rewards.index(best)


def _eval_worker(args):
    params, opp_names, seed, our_index, episode_steps, debug = args
    try:
        _silenced_imports()
        import bot_v7
        from kaggle_environments import make
        from opponents import ZOO

        opp_agents = []
        for name in tuple(opp_names or ()):
            if name not in ZOO:
                return 0.0
            opp_agents.append(ZOO[name])
        if not opp_agents:
            return 0.0

        scorer_w, heur = decode(np.asarray(params, dtype=np.float32))
        bot_v7.reset_heuristic_params()
        bot_v7.set_heuristic_params(heur)
        bot_v7.set_scorer(lambda f: float(scorer_w @ f.astype(np.float64)),
                          noise_std=0.0, log_player=-1)
        bot_v7.reset_episode_log()

        try:
            n_players = len(opp_agents) + 1
            our_index = int(our_index)
            if our_index < 0 or our_index >= n_players:
                our_index = 0

            agents = []
            opp_iter = iter(opp_agents)
            for slot in range(n_players):
                agents.append(bot_v7.agent if slot == our_index else next(opp_iter))

            env = make(
                "orbit_wars",
                configuration={"episodeSteps": int(episode_steps)},
                debug=bool(debug),
            )
            env.run(agents)
            won = _winner_from_env(env, n_players) == our_index
        except Exception:
            won = False
        finally:
            bot_v7.set_scorer(None)
            bot_v7.reset_heuristic_params()

        return 1.0 if won else 0.0
    except Exception:
        return 0.0


def evaluate_fixed(pool, params, opponents, games_per_opp, match_4p_ratio,
                   episode_steps, debug, seed_base=10000):
    total_games = len(opponents) * games_per_opp
    schedule = _build_match_schedule(opponents, total_games, match_4p_ratio, seed_base)
    tasks = []
    meta = []
    for mode, opp_names, our_index, seed in schedule:
        tasks.append((params, opp_names, seed, our_index, episode_steps, debug))
        meta.append(mode)

    results = pool.map(_eval_worker, tasks)
    per_mode = {}
    for mode, result in zip(meta, results):
        per_mode.setdefault(mode, []).append(int(result))
    summary = {mode: (sum(values), len(values)) for mode, values in per_mode.items()}
    total_w = sum(w for w, _ in summary.values())
    total_n = sum(n for _, n in summary.values())
    return total_w / max(1, total_n), summary


def es_generation(pool, params, sigma, opponents, n_pairs, games_per_eval,
                  seed_counter, match_4p_ratio, episode_steps, debug):
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
        for g in range(games_per_eval):
            mode, opp_names, our_index, seed = schedule[i * games_per_eval + g]
            tasks.append(((params + sigma * eps).astype(np.float32),
                          opp_names, seed, our_index, episode_steps, debug))
            tasks.append(((params - sigma * eps).astype(np.float32),
                          opp_names, seed, our_index, episode_steps, debug))

    raw_rewards = np.array(pool.map(_eval_worker, tasks), dtype=np.float64)
    per_pos = np.zeros(n_pairs)
    per_neg = np.zeros(n_pairs)
    for i in range(n_pairs):
        for g in range(games_per_eval):
            base = i * (2 * games_per_eval) + g * 2
            per_pos[i] += raw_rewards[base]
            per_neg[i] += raw_rewards[base + 1]
    per_pos /= max(1, games_per_eval)
    per_neg /= max(1, games_per_eval)

    all_r = np.concatenate([per_pos, per_neg])
    ranks = (all_r + rng.randn(len(all_r)) * 1e-6).argsort().argsort().astype(np.float64)
    shaped = ranks / max(1, len(ranks) - 1) - 0.5
    shaped_pos = shaped[:n_pairs]
    shaped_neg = shaped[n_pairs:]

    update = np.zeros(DIM, dtype=np.float64)
    for i, eps in enumerate(epsilons):
        update += (shaped_pos[i] - shaped_neg[i]) * eps.astype(np.float64)
    update /= max(1e-9, n_pairs * sigma)
    return update, float(raw_rewards.mean())


def format_summary(summary):
    return "  ".join(f"{mode}={w}/{n}" for mode, (w, n) in summary.items())


class SerialPool:
    def map(self, func, items):
        return [func(item) for item in items]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--games-per-eval", type=int, default=1)
    parser.add_argument("--eval-games-per-opp", type=int, default=1)
    parser.add_argument("--match-4p-ratio", type=float, default=MATCH_4P_RATIO)
    parser.add_argument("--eval-match-4p-ratio", type=float, default=None)
    parser.add_argument("--episode-steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--sigma-init", type=float, default=0.30)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--eval-every", type=int, default=999)
    parser.add_argument("--skip-baseline-eval", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--out", type=str, default="evaluations/scorer_v7_kaggle")
    parser.add_argument("--load", type=str, default=None,
                        help="Resume from a specific .npz checkpoint")
    parser.add_argument("--auto-resume", action="store_true", default=True,
                        help="Auto-resume from an existing checkpoint if present")
    parser.add_argument("--no-auto-resume", action="store_false", dest="auto_resume",
                        help="Disable auto-resume and start fresh")
    parser.add_argument("--resume-source", choices=("best", "latest"), default="best",
                        help="Auto-resume training from best or latest checkpoint")
    parser.add_argument("--rollback-on-bad-eval", action="store_true",
                        help="Rollback to best params when fixed eval is clearly worse")
    parser.add_argument("--rollback-margin", type=float, default=0.05,
                        help="Rollback if eval WR is more than this below best WR")
    parser.add_argument("--sigma-decay-on-rollback", type=float, default=0.7,
                        help="Multiply sigma by this after rollback")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    _silenced_imports()
    from opponents import ZOO

    available = [name for name in NOTEBOOK_OPPONENTS if name in ZOO]
    eval_opps = [name for name in EVAL_OPPONENTS if name in ZOO]
    eval_4p_ratio = args.match_4p_ratio if args.eval_match_4p_ratio is None else args.eval_match_4p_ratio

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path = Path(str(out_path) + "_latest.npz")
    best_path = Path(str(out_path) + ".npz")

    params = np.zeros(DIM, dtype=np.float32)
    momentum = np.zeros(DIM, dtype=np.float64)
    best_params = params.copy()
    best_score = -1.0
    saw_eval = False
    gen_offset = 0

    resume_candidates = []
    if args.load:
        resume_candidates.append(Path(args.load))
    elif args.auto_resume:
        if args.resume_source == "latest":
            resume_candidates.extend([latest_path, best_path])
        else:
            resume_candidates.extend([best_path, latest_path])

    resumed = False
    for candidate in resume_candidates:
        if candidate.exists():
            try:
                ckpt = np.load(candidate)
                params = ckpt["params"].astype(np.float32)
                momentum = ckpt["momentum"].astype(np.float64) if "momentum" in ckpt else np.zeros(DIM, dtype=np.float64)
                gen_offset = int(ckpt["generation"]) if "generation" in ckpt else 0
                best_score = float(ckpt["wr"]) if "wr" in ckpt else -1.0
                best_params = params.copy()
                resumed = True
                print(f"Resumed from {candidate} (gen={gen_offset}, wr={best_score:.3f})", flush=True)
                break
            except Exception:
                continue

    if best_path.exists():
        try:
            best_ckpt = np.load(best_path)
            saved_best_score = float(best_ckpt["wr"]) if "wr" in best_ckpt else -1.0
            if saved_best_score >= best_score:
                best_params = best_ckpt["params"].astype(np.float32)
                best_score = saved_best_score
        except Exception:
            pass

    if not resumed:
        scorer_path = ROOT / "evaluations" / "scorer_es.npy"
        if scorer_path.exists():
            try:
                w_es = np.load(scorer_path).astype(np.float32)
                if w_es.shape == (15,):
                    params[:15] = w_es / 0.5
                    best_params = params.copy()
                    print(f"Warm-start scorer block from {scorer_path}", flush=True)
            except Exception:
                pass

    print("Kaggle ES training", flush=True)
    print(f"Pool: {len(available)} opponents | Eval set: {len(eval_opps)}", flush=True)
    print(f"minutes={args.minutes} workers={args.workers} pairs={args.pairs} "
          f"episode_steps={args.episode_steps} 4p_ratio={args.match_4p_ratio}", flush=True)
    print(f"resume_source={args.resume_source} rollback_on_bad_eval={args.rollback_on_bad_eval}",
          flush=True)

    deadline = time.time() + args.minutes * 60.0
    started = time.time()
    generation = gen_offset
    sigma_scale = 1.0

    pool_factory = SerialPool if args.workers <= 1 else None
    ctx = mp.get_context("spawn") if args.workers > 1 else None
    pool_context = pool_factory() if pool_factory else ctx.Pool(processes=args.workers)
    with pool_context as pool:
        if not args.skip_baseline_eval:
            t0 = time.time()
            wr, summary = evaluate_fixed(pool, params, eval_opps, args.eval_games_per_opp,
                                         eval_4p_ratio, args.episode_steps, args.debug)
            best_score = wr
            best_params = params.copy()
            saw_eval = True
            np.savez(str(best_path), params=best_params, momentum=momentum,
                     generation=generation, wr=best_score)
            print(f"baseline WR={wr:.0%} {format_summary(summary)} ({time.time() - t0:.1f}s)",
                  flush=True)
        else:
            print("baseline skipped", flush=True)

        while time.time() < deadline:
            sigma = max(args.sigma_min, args.sigma_init * sigma_scale)
            t0 = time.time()
            update, avg_r = es_generation(
                pool, params, sigma, available, args.pairs, args.games_per_eval,
                generation, args.match_4p_ratio, args.episode_steps, args.debug,
            )
            momentum = args.momentum * momentum + (1.0 - args.momentum) * update
            params = (params + args.lr * momentum).astype(np.float32)
            generation += 1

            np.savez(str(latest_path), params=params, momentum=momentum,
                     generation=generation, wr=best_score if best_score >= 0 else 0.0)

            print(f"gen {generation:4d} | avg_r={avg_r:+.3f} | "
                  f"|p|={np.linalg.norm(params):.2f} | gen_t={time.time() - t0:.1f}s",
                  flush=True)

            if generation % args.eval_every == 0:
                t1 = time.time()
                wr, summary = evaluate_fixed(pool, params, eval_opps, args.eval_games_per_opp,
                                             eval_4p_ratio, args.episode_steps, args.debug)
                saw_eval = True
                if wr > best_score:
                    best_score = wr
                    best_params = params.copy()
                    np.savez(str(best_path), params=best_params, momentum=momentum,
                             generation=generation, wr=best_score)
                    eval_note = " best"
                elif (
                    args.rollback_on_bad_eval
                    and best_score >= 0.0
                    and wr < best_score - args.rollback_margin
                ):
                    params = best_params.copy()
                    momentum = np.zeros(DIM, dtype=np.float64)
                    sigma_scale = max(0.25, sigma_scale * args.sigma_decay_on_rollback)
                    eval_note = f" rollback sigma_scale={sigma_scale:.2f}"
                    np.savez(str(latest_path), params=params, momentum=momentum,
                             generation=generation, wr=best_score)
                else:
                    eval_note = ""
                print(f"eval WR={wr:.0%} {format_summary(summary)} ({time.time() - t1:.1f}s){eval_note}",
                      flush=True)

        if args.skip_final_eval:
            best_params = params.copy()
            print("final eval skipped", flush=True)
        else:
            if not saw_eval:
                best_params = params.copy()
            t0 = time.time()
            wr, summary = evaluate_fixed(pool, best_params, eval_opps, args.eval_games_per_opp,
                                         eval_4p_ratio, args.episode_steps, args.debug)
            if not saw_eval:
                best_score = wr
            np.savez(str(best_path), params=best_params, momentum=momentum,
                     generation=generation, wr=best_score)
            print(f"final WR={wr:.0%} {format_summary(summary)} ({time.time() - t0:.1f}s)",
                  flush=True)

    if args.skip_final_eval and best_score < 0.0:
        np.savez(str(best_path), params=best_params, momentum=momentum,
                 generation=generation, wr=best_score)
    np.savez(str(latest_path), params=params, momentum=momentum,
             generation=generation, wr=best_score if best_score >= 0 else 0.0)
    scorer_w, heur = decode(best_params)
    np.save(str(out_path) + "_scorer.npy", scorer_w.astype(np.float32))
    with open(str(out_path) + "_heur.txt", "w") as f:
        for key, value in heur.items():
            f.write(f"{key} = {value:.4f}\n")

    print(f"Done. {generation} generations in {time.time() - started:.1f}s", flush=True)
    if best_score >= 0.0:
        print(f"Best WR: {best_score:.0%}", flush=True)
    else:
        print("Best WR: n/a (all fixed evals skipped)", flush=True)


if __name__ == "__main__":
    main()
