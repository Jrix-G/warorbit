#!/usr/bin/env python3
"""Long-run ES trainer for the bot_v8_2 candidate-plan ranker.

Optimizes only the lightweight V8.2 ranker/value weights — the planner grammar
is hand-authored. Training learns when to pick each candidate plan.

Key design points (vs the original draft):

* Persistent multiprocessing pool. Notebook agents are 3000+ lines each; we
  pay the import cost once per worker via `initializer`.
* Single batched pool.map per generation. Each task carries its own params
  vector, so positive/negative mirrors are scheduled together with the
  periodic eval.
* Fixed eval seeds across generations. Generation-N and generation-(N+5) eval
  scores must be on the same opponent draws or "best" promotion is noise.
* 2p and 4p winrate tracked separately. Best promotion requires non-regression
  on the worst mode, not just average WR.
* Train games can use shorter --train-max-steps than --eval-max-steps. ES does
  not need 500-turn games to estimate plan ordering.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


DEFAULT_TRAIN_OPPONENTS = [
    "notebook_orbitbotnext",
    "notebook_distance_prioritized",
    "notebook_physics_accurate",
    "notebook_tactical_heuristic",
]


# ---------------------------------------------------------------------------
# Worker process: imports done once, weights set per task.
# ---------------------------------------------------------------------------

_WORKER_BOT = None
_WORKER_RUN = None
_WORKER_ZOO = None


def _silent_import_bot():
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    try:
        import bot_v8_2  # noqa: F401
        from SimGame import run_match  # noqa: F401
        from opponents import ZOO  # noqa: F401
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return bot_v8_2, run_match, ZOO


def _worker_init():
    global _WORKER_BOT, _WORKER_RUN, _WORKER_ZOO
    _WORKER_BOT, _WORKER_RUN, _WORKER_ZOO = _silent_import_bot()


def _worker_play(task):
    """Play one match with the given flat params vector.

    task = (params, opponent_names, our_index, seed, max_steps).
    Returns (raw_reward_in_[0,1], mode, winner_index, our_score, top_other_score).
    """
    params, opponent_names, our_index, seed, max_steps = task
    if _WORKER_BOT is None:
        _worker_init()
    bot = _WORKER_BOT
    run_match = _WORKER_RUN
    ZOO = _WORKER_ZOO

    _set_flat_weights(bot, params)

    agents = []
    opp_iter = iter(opponent_names)
    for slot in range(len(opponent_names) + 1):
        if slot == our_index:
            agents.append(bot.agent)
        else:
            name = next(opp_iter)
            agents.append(ZOO[name])
    n_players = len(agents)
    result = run_match(agents, seed=seed, n_players=n_players, max_steps=max_steps)
    winner = int(result.get("winner", -1))
    scores = list(result.get("scores", []))
    if winner == our_index:
        base = 1.0
    elif winner < 0:
        base = 0.5
    else:
        base = 0.0
    if 0 <= our_index < len(scores):
        our = float(scores[our_index])
        other = max((float(s) for i, s in enumerate(scores) if i != our_index), default=our)
        margin = math.tanh((our - other) / max(100.0, abs(our) + abs(other)))
    else:
        our = other = 0.0
        margin = 0.0
    reward = float(np.clip(base + 0.05 * margin, 0.0, 1.0))
    mode = "2p" if n_players == 2 else "4p"
    return reward, mode, winner, our, other


# ---------------------------------------------------------------------------
# Flat-vector helpers (can be called from main or worker).
# ---------------------------------------------------------------------------

def _flatten_weights(bot) -> np.ndarray:
    state_w, cand_w, value_w, value_b = bot.get_ranker_weights()
    return np.concatenate([
        state_w.ravel(),
        cand_w.ravel(),
        value_w.ravel(),
        np.array([value_b], dtype=np.float32),
    ]).astype(np.float32)


def _set_flat_weights(bot, vec: np.ndarray) -> None:
    vec = np.asarray(vec, dtype=np.float32).ravel()
    n_state = bot.N_PLANS_MAX * bot.N_STATE_FEATURES
    n_cand = bot.N_CANDIDATE_FEATURES
    n_value = bot.N_STATE_FEATURES
    expected = n_state + n_cand + n_value + 1
    if vec.size != expected:
        raise ValueError(f"weight vector size {vec.size} != {expected}")
    off = 0
    state_w = vec[off:off + n_state].reshape(bot.N_PLANS_MAX, bot.N_STATE_FEATURES)
    off += n_state
    cand_w = vec[off:off + n_cand]
    off += n_cand
    value_w = vec[off:off + n_value]
    off += n_value
    value_b = float(vec[off])
    bot.set_ranker_weights(state_w=state_w, candidate_w=cand_w, value_w=value_w, value_b=value_b)


# ---------------------------------------------------------------------------
# Schedule generation.
# ---------------------------------------------------------------------------

def _schedule(opponents, n_games, four_player_ratio, seed, max_steps):
    rng = np.random.default_rng(seed)
    pool = list(opponents)
    if not pool:
        return []
    out = []
    for i in range(n_games):
        is_4p = len(pool) >= 3 and float(rng.random()) < four_player_ratio
        if is_4p:
            opps = list(rng.choice(pool, size=3, replace=len(pool) < 3))
            our_index = int(rng.integers(0, 4))
        else:
            opps = [str(rng.choice(pool))]
            our_index = int(rng.integers(0, 2))
        out.append((opps, our_index, int(seed * 100003 + i), int(max_steps)))
    return out


def _eval_schedule(opponents, eval_games, four_player_ratio, eval_seed_base, max_steps):
    """Stable evaluation schedule — same seeds across generations."""
    return _schedule(opponents, eval_games, four_player_ratio, eval_seed_base, max_steps)


# ---------------------------------------------------------------------------
# ES utilities.
# ---------------------------------------------------------------------------

def _rank_shape(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x))
    shaped = order.astype(np.float32) / max(1, len(x) - 1)
    return shaped - np.mean(shaped)


def _save_flat(path: Path, params, best_score, generation, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        params=np.asarray(params, dtype=np.float32),
        best_score=np.array(best_score, dtype=np.float32),
        generation=np.array(generation, dtype=np.int32),
        meta_json=np.array(json.dumps(meta, sort_keys=True, default=str)),
    )


def _load_flat(path: Path):
    data = np.load(str(path))
    params = np.asarray(data["params"], dtype=np.float32)
    best_score = float(data["best_score"]) if "best_score" in data else -1.0
    generation = int(data["generation"]) if "generation" in data else 0
    return params, best_score, generation


# ---------------------------------------------------------------------------
# Per-mode summarising of pool results.
# ---------------------------------------------------------------------------

def _summarise(rewards: List[Tuple[float, str, int, float, float]]):
    """Return (mean_reward, wr_2p, wr_4p, n_2p, n_4p)."""
    if not rewards:
        return 0.0, 0.0, 0.0, 0, 0
    r2 = [r for r, m, *_ in rewards if m == "2p"]
    r4 = [r for r, m, *_ in rewards if m == "4p"]
    mean = float(np.mean([r for r, *_ in rewards]))
    wr2 = float(np.mean(r2)) if r2 else 0.0
    wr4 = float(np.mean(r4)) if r4 else 0.0
    return mean, wr2, wr4, len(r2), len(r4)


# ---------------------------------------------------------------------------
# Main training loop.
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train bot_v8_2 ranker with antithetic ES.")
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--games-per-eval", type=int, default=2)
    parser.add_argument("--eval-games", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=220,
                        help="Training game cap (kept small for ES throughput).")
    parser.add_argument("--eval-max-steps", type=int, default=None,
                        help="Eval game cap. Defaults to --max-steps; set 500 for fidelity eval.")
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=0.0005)
    parser.add_argument("--four-player-ratio", type=float, default=0.70)
    parser.add_argument("--eval-four-player-ratio", type=float, default=None,
                        help="Eval 4p ratio (defaults to --four-player-ratio).")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=8202)
    parser.add_argument("--checkpoint", default="evaluations/v8_2_ranker_train_latest.npz")
    parser.add_argument("--best-checkpoint", default="evaluations/v8_2_ranker_train_best.npz")
    parser.add_argument("--export-bot-checkpoint", default=None,
                        help="Defaults to bot_v8_2.DEFAULT_CHECKPOINT.")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--pool-limit", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Run a fixed-seed eval every N generations.")
    parser.add_argument("--min-improvement", type=float, default=0.01,
                        help="Best-checkpoint promotion requires this much WR gain.")
    parser.add_argument("--min-mode-floor", type=float, default=0.05,
                        help="Best-checkpoint promotion requires per-mode WR drop ≤ this from previous best.")
    parser.add_argument("--log-jsonl", default="evaluations/v8_2_train.jsonl")
    args = parser.parse_args()

    eval_max_steps = args.eval_max_steps if args.eval_max_steps is not None else args.max_steps
    eval_4p_ratio = args.eval_four_player_ratio if args.eval_four_player_ratio is not None else args.four_player_ratio

    # Lazy imports so worker initializer mirrors main process imports.
    bot_v8_2, run_match, ZOO_main = _silent_import_bot()
    from opponents import training_pool

    if args.export_bot_checkpoint is None:
        args.export_bot_checkpoint = str(bot_v8_2.DEFAULT_CHECKPOINT)

    opponents = [name for name in training_pool(args.pool_limit) if name in ZOO_main]
    if not opponents:
        opponents = [name for name in DEFAULT_TRAIN_OPPONENTS if name in ZOO_main]
    if not opponents:
        raise SystemExit("No opponents available for training.")

    latest_path = Path(args.checkpoint)
    best_path = Path(args.best_checkpoint)
    if args.resume and latest_path.exists():
        params, best_score, generation = _load_flat(latest_path)
        print(f"Resumed {latest_path} generation={generation} best_score={best_score:.4f}", flush=True)
    else:
        params = _flatten_weights(bot_v8_2)
        best_score = -1.0
        generation = 0
        print(f"Starting fresh dim={params.size}", flush=True)

    best_wr_2p = -1.0
    best_wr_4p = -1.0
    if args.resume and best_path.exists():
        try:
            data = np.load(str(best_path))
            meta = json.loads(str(data["meta_json"])) if "meta_json" in data else {}
            best_wr_2p = float(meta.get("best_wr_2p", -1.0))
            best_wr_4p = float(meta.get("best_wr_4p", -1.0))
        except Exception:
            pass

    rng = np.random.default_rng(args.seed + generation)
    start = time.time()
    deadline = start + max(1.0, args.minutes * 60.0)
    velocity = np.zeros_like(params)

    log_path = Path(args.log_jsonl)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Persistent pool (spawn — avoids fork-with-numpy quirks on Linux).
    if args.workers > 1:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=args.workers, initializer=_worker_init)
        map_fn = pool.map
    else:
        pool = None
        _worker_init()
        def map_fn(fn, items):
            return [fn(i) for i in items]

    try:
        while time.time() < deadline:
            generation += 1

            # Build training schedule (one slice per pair, shared by pos and neg).
            train_sched = _schedule(
                opponents, args.pairs * args.games_per_eval,
                args.four_player_ratio,
                args.seed + generation, args.max_steps,
            )
            if not train_sched:
                print("[warn] empty training schedule — aborting", flush=True)
                break

            eps = rng.normal(size=(args.pairs, params.size)).astype(np.float32)

            # Build a flat task list: per-pair pos + neg interleaved.
            tasks = []
            task_meta = []  # (pair_idx, sign)  sign in {+1, -1}
            for i in range(args.pairs):
                pos = (params + args.sigma * eps[i]).astype(np.float32)
                neg = (params - args.sigma * eps[i]).astype(np.float32)
                slice_ = train_sched[i * args.games_per_eval:(i + 1) * args.games_per_eval]
                for opps, our_idx, seed, max_steps in slice_:
                    tasks.append((pos, opps, our_idx, seed, max_steps))
                    task_meta.append((i, +1))
                for opps, our_idx, seed, max_steps in slice_:
                    tasks.append((neg, opps, our_idx, seed, max_steps))
                    task_meta.append((i, -1))

            results = list(map_fn(_worker_play, tasks))

            rewards_pos = [[] for _ in range(args.pairs)]
            rewards_neg = [[] for _ in range(args.pairs)]
            for (pair_idx, sign), res in zip(task_meta, results):
                bucket = rewards_pos if sign > 0 else rewards_neg
                bucket[pair_idx].append(res[0])

            r_pos = np.array([np.mean(b) if b else 0.0 for b in rewards_pos], dtype=np.float32)
            r_neg = np.array([np.mean(b) if b else 0.0 for b in rewards_neg], dtype=np.float32)
            shaped = _rank_shape(np.concatenate([r_pos, r_neg]))
            shaped_pos = shaped[:args.pairs]
            shaped_neg = shaped[args.pairs:]
            grad = np.mean((shaped_pos - shaped_neg)[:, None] * eps, axis=0) / max(args.sigma, 1e-6)
            grad -= args.l2 * params
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm > 10.0:
                grad *= 10.0 / grad_norm
            velocity = 0.9 * velocity + 0.1 * grad
            params = (params + args.lr * velocity).astype(np.float32)

            train_mean, train_wr_2p, train_wr_4p, n_train_2p, n_train_4p = _summarise(results)

            eval_mean = train_mean
            eval_wr_2p = train_wr_2p
            eval_wr_4p = train_wr_4p
            improved = False
            promoted = False
            if generation == 1 or generation % args.eval_every == 0:
                eval_sched = _eval_schedule(
                    opponents, args.eval_games,
                    eval_4p_ratio,
                    args.seed + 50000, eval_max_steps,
                )
                eval_tasks = [(params, opps, our_idx, seed, max_steps)
                              for opps, our_idx, seed, max_steps in eval_sched]
                eval_results = list(map_fn(_worker_play, eval_tasks))
                eval_mean, eval_wr_2p, eval_wr_4p, n_eval_2p, n_eval_4p = _summarise(eval_results)

                # Floor checks only kick in once we have a real best (best_score >= 0).
                if best_score < 0:
                    regress_floor_2p = -1.0
                    regress_floor_4p = -1.0
                else:
                    regress_floor_2p = best_wr_2p - args.min_mode_floor if best_wr_2p > 0 else -1.0
                    regress_floor_4p = best_wr_4p - args.min_mode_floor if best_wr_4p > 0 else -1.0

                gain = eval_mean - best_score
                non_regressing = (eval_wr_2p >= regress_floor_2p) and (eval_wr_4p >= regress_floor_4p)
                if gain >= args.min_improvement and non_regressing:
                    best_score = eval_mean
                    best_wr_2p = eval_wr_2p
                    best_wr_4p = eval_wr_4p
                    improved = True
                    promoted = True
                    _save_flat(best_path, params, best_score, generation, {
                        "args": vars(args),
                        "best_wr_2p": best_wr_2p,
                        "best_wr_4p": best_wr_4p,
                    })
                    _set_flat_weights(bot_v8_2, params)
                    bot_v8_2.save_checkpoint(args.export_bot_checkpoint, meta={
                        "generation": generation,
                        "score": best_score,
                        "wr_2p": best_wr_2p,
                        "wr_4p": best_wr_4p,
                    })
                elif gain > 0 and not non_regressing:
                    print(f"  [skip] eval gained {gain:+.3f} but mode-floor regressed "
                          f"(2p {eval_wr_2p:.3f} vs best {best_wr_2p:.3f}, "
                          f"4p {eval_wr_4p:.3f} vs best {best_wr_4p:.3f})",
                          flush=True)

            _save_flat(latest_path, params, best_score, generation, vars(args))

            elapsed_min = (time.time() - start) / 60.0
            line = (
                f"gen={generation:04d} "
                f"train={train_mean:.3f} (2p {train_wr_2p:.3f}/{n_train_2p} 4p {train_wr_4p:.3f}/{n_train_4p}) "
                f"eval={eval_mean:.3f} (2p {eval_wr_2p:.3f} 4p {eval_wr_4p:.3f}) "
                f"best={best_score:.3f} grad={grad_norm:.2f} "
                f"promo={int(promoted)} elapsed_min={elapsed_min:.1f}"
            )
            print(line, flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps({
                    "gen": generation,
                    "train_mean": train_mean,
                    "train_wr_2p": train_wr_2p,
                    "train_wr_4p": train_wr_4p,
                    "eval_mean": eval_mean,
                    "eval_wr_2p": eval_wr_2p,
                    "eval_wr_4p": eval_wr_4p,
                    "best": best_score,
                    "best_wr_2p": best_wr_2p,
                    "best_wr_4p": best_wr_4p,
                    "grad_norm": grad_norm,
                    "promoted": promoted,
                    "elapsed_min": elapsed_min,
                }) + "\n")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    _set_flat_weights(bot_v8_2, params)
    bot_v8_2.save_checkpoint(args.export_bot_checkpoint, meta={
        "generation": generation,
        "score": best_score,
        "wr_2p": best_wr_2p,
        "wr_4p": best_wr_4p,
    })
    print(f"Saved latest={latest_path} best={best_path} bot_checkpoint={args.export_bot_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
