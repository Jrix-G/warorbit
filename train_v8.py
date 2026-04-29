#!/usr/bin/env python3
"""Train the Orbit Wars V8 plan ranker.

The training loop is intentionally simple:
- roll out the current policy against notebook opponents,
- on sampled states, generate candidate plans,
- label them with a short-rollout oracle,
- update a linear plan ranker with DAgger-style aggregated data.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import pickle
import time
from collections import deque
from typing import Callable, Dict, List, Optional

import numpy as np

import bot_v7
import bot_v8
from SimGame import SimGame, run_match
from v8_core import (
    MODEL_PATH,
    LinearV8Model,
    TrainingExample,
    build_candidate_plans,
    build_state_features,
    candidate_scores,
    select_plan,
    step_policy_episode,
)


DEFAULT_OPPONENTS = [
    "notebook_physics_accurate",
    "notebook_orbitbotnext",
    "notebook_distance_prioritized",
]


def _call_obs_agent(agent: Callable, obs: dict):
    try:
        return agent(obs, None)
    except TypeError:
        return agent(obs)
    except Exception:
        return []


def _make_rollout_policy(train_player: int, opponent_policy: Callable) -> Callable:
    """Continuation policy used during oracle rollouts.

    Both sides play V7 inside the rollout. Tradeoff:

    - Cost: notebook opponents take ~50-150ms per call; V7 takes ~5ms. With
      9 plans x 15 rollout steps per labeled state, the difference is the
      gap between a tractable training loop and one that produces no
      episodes at all.
    - Bias: the oracle now estimates 'plan c -> V7 vs V7' instead of 'plan c
      vs the actual training opponent'. The label is biased toward plans
      that are good in the V7-vs-V7 distribution. We accept that bias to
      get any signal at all; the forward game still uses the real opponent,
      so the *training distribution* of states is correct.
    - Stationarity: V7 is a fixed reference. Oracle labels do not drift as
      the trained model changes, which is what an offline ranker needs.
    """
    del train_player, opponent_policy  # kept for API compatibility

    def policy(game, player: int):
        obs = game.observation(player)
        try:
            move = bot_v7.agent(obs, None)
        except Exception:
            move = []
        return move if isinstance(move, list) else []

    return policy


def _load_opponent(name: str) -> Callable:
    try:
        from opponents import ZOO

        return ZOO[name]
    except Exception:
        def passive(obs, config=None):
            return []

        return passive


def _make_opponent_policy(agent: Callable) -> Callable:
    def policy(game, player: int):
        obs = game.observation(player)
        return _call_obs_agent(agent, obs)

    return policy


def _final_value(scores: List[int], our_player: int, winner: int) -> float:
    if not scores:
        return 0.0
    best_other = max(float(s) for i, s in enumerate(scores) if i != our_player)
    our_score = float(scores[our_player])
    scale = max(1.0, sum(abs(float(s)) for s in scores))
    win_term = 1.0 if winner == our_player else (-1.0 if winner >= 0 else 0.0)
    margin_term = math.tanh(2.5 * (our_score - best_other) / scale)
    return float(0.7 * win_term + 0.3 * margin_term)


def _run_benchmark(model: LinearV8Model, games: int, opponents: List[str]) -> Dict[str, tuple[int, int]]:
    bot_v8.set_model(model)
    results: Dict[str, tuple[int, int]] = {}
    for opp_name in opponents:
        opp = _load_opponent(opp_name)
        wins = 0
        print(f"  Benchmarking vs {opp_name}...", flush=True)
        for i in range(games):
            if i % 2 == 0:
                result = run_match([bot_v8.agent, opp], seed=1000 + i)
                wins += 1 if result["winner"] == 0 else 0
            else:
                result = run_match([opp, bot_v8.agent], seed=1000 + i)
                wins += 1 if result["winner"] == 1 else 0
        results[opp_name] = (wins, games)
    return results


def _benchmark_score(results: Dict[str, tuple[int, int]]) -> float:
    per_opponent = [w / max(1, n) for w, n in results.values()]
    if not per_opponent:
        return 0.0
    global_rate = sum(w for w, _ in results.values()) / max(1, sum(n for _, n in results.values()))
    min_rate = min(per_opponent)
    mean_rate = sum(per_opponent) / len(per_opponent)
    # Stable selection criterion: reward broad competence, not a single matchup spike.
    return float(0.50 * global_rate + 0.35 * min_rate + 0.15 * mean_rate)


def _benchmark_stats(results: Dict[str, tuple[int, int]]) -> tuple[float, float, float]:
    if not results:
        return 0.0, 0.0, 0.0
    per_opponent = [w / max(1, n) for w, n in results.values()]
    global_rate = sum(w for w, _ in results.values()) / max(1, sum(n for _, n in results.values()))
    min_rate = min(per_opponent)
    mean_rate = sum(per_opponent) / len(per_opponent)
    return float(global_rate), float(min_rate), float(mean_rate)


def _sample_buffer(buffer: deque, n: int) -> List[TrainingExample]:
    if not buffer:
        return []
    n = min(n, len(buffer))
    if n <= 0:
        return []
    return random.sample(list(buffer), n)


def _default_state_path(out_path: str) -> str:
    root, _ = os.path.splitext(out_path)
    return f"{root}_state.pkl"


def _save_train_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_train_state(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--value-lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=2000)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--min-oracle-gap", type=float, default=0.01,
                        help="drop samples where the oracle cannot separate plans by at "
                             "least this margin; learning on flat states is pure noise. "
                             "Calibrate against the mean_span column of the oracle "
                             "diagnostic on YOUR plan set (seen ~0.05 with 9 plans @ H=15-20).")
    parser.add_argument("--benchmark-games", type=int, default=20)
    parser.add_argument("--holdout-opponent", type=str, default="orbit_stars",
                        help="opponent reserved for held-out validation; not used during training. "
                             "(notebook_tactical_heuristic is broken in the current zoo: "
                             "raises NameError 'Planet')")
    parser.add_argument("--benchmark-seconds", type=int, default=900)
    parser.add_argument("--save-seconds", type=int, default=900)
    # Keep the oracle longer: the policy is still brittle when beta gets too low.
    parser.add_argument("--beta-start", type=float, default=0.85)
    parser.add_argument("--beta-end", type=float, default=0.20)
    parser.add_argument("--beta-decay", type=float, default=60.0)
    parser.add_argument("--skip-initial-benchmark", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-state", type=str, default=None)
    parser.add_argument("--state-out", type=str, default=None)
    parser.add_argument("--out", type=str, default=MODEL_PATH)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    state_path = args.state_out or _default_state_path(args.out)
    deadline = time.time() + args.hours * 3600.0

    if args.resume and os.path.exists(args.resume):
        model = LinearV8Model.load(args.resume)
        print(f"Resumed from {args.resume}")
    else:
        model = LinearV8Model.zero()
        print("Warm-start: zero weights (pure V7 behavior)")

    bot_v7.set_scorer(None)
    bot_v8.set_model(model)

    print("\nMethod: DAgger + hard-negative plan ranking + value head")
    print(
        f"Config: lr={args.lr}  value_lr={args.value_lr}  l2={args.l2}  "
        f"batch={args.batch_size}  buffer={args.buffer_size}  sample_stride={args.sample_stride}"
    )
    print("Starting training...", flush=True)
    print()

    best_score = -1.0
    best_eval = (-1.0, -1.0, -1.0)
    best_model = model.copy()
    # Initialise these to "now" so --skip-initial-benchmark does not immediately
    # trigger a periodic benchmark at the first iteration (last_benchmark=0
    # makes the elapsed delta huge regardless of --benchmark-seconds).
    last_benchmark = time.time()
    last_save = time.time()
    episode = 0
    t_start = time.time()
    replay_buffer: deque[TrainingExample] = deque(maxlen=args.buffer_size)
    rng = random.Random(int(time.time() * 1000) % 100000)

    resume_state_path = args.resume_state or state_path
    resumed_state = _load_train_state(resume_state_path)
    if resumed_state:
        try:
            episode = int(resumed_state.get("episode", 0))
            best_score = float(resumed_state.get("best_score", best_score))
            best_eval = tuple(resumed_state.get("best_eval", best_eval))  # type: ignore[assignment]
            replay_items = resumed_state.get("replay_buffer", [])
            replay_buffer = deque(
                (item for item in replay_items if isinstance(item, TrainingExample)),
                maxlen=args.buffer_size,
            )
            rng_state = resumed_state.get("rng_state", None)
            if rng_state is not None:
                rng.setstate(rng_state)
            best_path = args.out.replace(".npz", "_best.npz")
            if os.path.exists(best_path):
                best_model = LinearV8Model.load(best_path)
            print(f"Resumed training state from {resume_state_path} (episode={episode}, buffer={len(replay_buffer)})")
        except Exception as exc:
            print(f"  (could not load training state from {resume_state_path}: {exc})")

    bench_opponents = list(DEFAULT_OPPONENTS)
    if args.holdout_opponent and args.holdout_opponent not in bench_opponents:
        try:
            _load_opponent(args.holdout_opponent)
            bench_opponents = bench_opponents + [args.holdout_opponent]
        except Exception:
            print(f"  (holdout '{args.holdout_opponent}' unavailable; benchmarking on training set only)")

    if not args.skip_initial_benchmark:
        print("[Initial benchmark (zero weights = pure V7)]", flush=True)
        bench = _run_benchmark(model, args.benchmark_games, bench_opponents)
        for k, (wins, total) in bench.items():
            tag = " (holdout)" if k == args.holdout_opponent else ""
            print(f"  {k}{tag}: {wins}/{total}")
        global_rate, min_rate, mean_rate = _benchmark_stats(bench)
        best_score = _benchmark_score(bench)
        best_eval = (best_score, global_rate, min_rate)
        print(f"  global: {global_rate:.3f}")
        print(f"  min   : {min_rate:.3f}")
        print(f"  mean  : {mean_rate:.3f}")
        print(f"  score : {best_score:.3f}")
        print()
        last_benchmark = time.time()

    while time.time() < deadline:
        episode += 1
        seed = rng.randrange(1, 2**31 - 1)
        opponent_name = DEFAULT_OPPONENTS[(episode - 1) % len(DEFAULT_OPPONENTS)]
        if rng.random() < 0.35:
            opponent_name = rng.choice(DEFAULT_OPPONENTS)
        opponent_agent = _load_opponent(opponent_name)
        opponent_policy = _make_opponent_policy(opponent_agent)
        game = SimGame.random_game(seed=seed, n_players=2, neutral_pairs=8)
        train_player = episode % 2
        beta = max(args.beta_end, args.beta_start * math.exp(-(episode - 1) / max(args.beta_decay, 1e-6)))

        episode_samples: List[dict] = []

        while not game.is_terminal():
            actions = {}
            for player in range(game.n_players):
                if player == train_player:
                    obs = game.observation(player)
                    world = bot_v7._build_world(obs)
                    state_feat = build_state_features(world)
                    plans = build_candidate_plans(world)
                    if not plans:
                        actions[player] = []
                        continue

                    feats = np.stack([p.features for p in plans], axis=0)
                    scores = model.score_candidates(feats)
                    model_idx = int(np.argmax(scores)) if len(scores) else 0
                    if game.state.step % max(1, args.sample_stride) == 0:
                        rollout_policy = _make_rollout_policy(train_player, opponent_policy)
                        steps = max(1, int(args.rollout_steps))
                        # Single-horizon rollout. The multi-horizon blend that
                        # was here added cost without measurable signal gain
                        # (see diagnostics/oracle_horizon.py).
                        rollout_values = [
                            float(step_policy_episode(
                                game,
                                train_player,
                                plan.actions,
                                rollout_policy,
                                steps=steps,
                            ))
                            for plan in plans
                        ]
                        # Margin oracle: the V7 baseline (plans[0]) is the control.
                        # Subtracting it cancels production / orbit drift that is
                        # common to all candidates within this state, so the
                        # remaining signal is the differential effect of the plan.
                        baseline = float(rollout_values[0]) if rollout_values else 0.0
                        margin_values = [float(v) - baseline for v in rollout_values]
                        oracle_idx = int(np.argmax(margin_values)) if margin_values else 0
                        oracle_value = float(margin_values[oracle_idx]) if margin_values else 0.0
                        # Oracle gap: only mine hard negatives where the oracle is
                        # actually confident the chosen plan beats the runner-up.
                        oracle_gap = 0.0
                        if len(margin_values) > 1:
                            sorted_margin = sorted(margin_values, reverse=True)
                            oracle_gap = float(sorted_margin[0] - sorted_margin[1])
                        confident = oracle_gap >= args.min_oracle_gap
                        disagree = (model_idx != oracle_idx)
                        repeats = 2 if (disagree and confident) else 1
                        # Diagnostic showed that ~70% of states have all
                        # candidates within rollout noise. Training on those
                        # states injects label noise without signal, so we
                        # drop them.
                        if confident:
                            episode_samples.append(
                                {
                                    "state": state_feat,
                                    "candidate_features": feats,
                                    "label": oracle_idx,
                                    "value_target": oracle_value,
                                    "repeats": repeats,
                                }
                            )
                        chosen_idx = oracle_idx if (confident and rng.random() < beta) else model_idx
                    else:
                        chosen_idx = model_idx
                    actions[player] = plans[chosen_idx].actions
                else:
                    actions[player] = opponent_policy(game, player)

            game.step(actions)

        scores = game.scores()
        winner = game.winner()
        episode_value = _final_value(scores, train_player, winner)

        for sample in episode_samples:
            merged_value = 0.5 * float(sample["value_target"]) + 0.5 * episode_value
            repeats = int(sample.get("repeats", 1))
            for _ in range(max(1, repeats)):
                replay_buffer.append(
                    TrainingExample(
                        state_features=np.asarray(sample["state"], dtype=np.float32),
                        candidate_features=np.asarray(sample["candidate_features"], dtype=np.float32),
                        label=int(sample["label"]),
                        value_target=float(merged_value),
                    )
                )

        update_count = max(1, min(len(replay_buffer), max(1, args.batch_size // 2)))
        batch = _sample_buffer(replay_buffer, update_count)
        stats = model.train_batch(batch, lr=args.lr, value_lr=args.value_lr, l2=args.l2)

        elapsed = time.time() - t_start
        print(
            f"[ep {episode:5d} | {elapsed/60:5.1f}min]  "
            f"value={episode_value:+.3f}  beta={beta:.2f}  "
            f"loss={stats['loss']:.3f}  rank={stats.get('rank_loss', 0.0):.3f}  "
            f"acc={stats['acc']:.2f}  "
            f"|w|={np.linalg.norm(model.score_w):.4f}"
        )

        now = time.time()
        if now - last_save >= args.save_seconds:
            model.save(args.out)
            print(f"  Saved: {args.out}")
            last_save = now

        if now - last_benchmark >= args.benchmark_seconds:
            print(f"  [Benchmark @ ep {episode}]")
            bench = _run_benchmark(model, args.benchmark_games, bench_opponents)
            for k, (wins, total) in bench.items():
                tag = " (holdout)" if k == args.holdout_opponent else ""
                print(f"    {k}{tag}: {wins}/{total}")
            global_rate, min_rate, mean_rate = _benchmark_stats(bench)
            score = _benchmark_score(bench)
            print(f"    global          : {global_rate:.3f}")
            print(f"    min             : {min_rate:.3f}")
            print(f"    mean            : {mean_rate:.3f}")
            print(f"    score           : {score:.3f}")
            current_eval = (score, global_rate, min_rate)
            if current_eval > best_eval:
                best_score = score
                best_eval = current_eval
                best_model = model.copy()
                best_path = args.out.replace(".npz", "_best.npz")
                best_model.save(best_path)
                print(f"    New best saved  : {best_path}")
            last_benchmark = now

        if now - last_save >= args.save_seconds or episode == 1:
            _save_train_state(
                state_path,
                {
                    "version": 2,
                    "episode": episode,
                    "best_score": best_score,
                    "best_eval": best_eval,
                    "rng_state": rng.getstate(),
                    "replay_buffer": list(replay_buffer),
                },
            )

    model.save(args.out)
    best_path = args.out.replace(".npz", "_best.npz")
    if not os.path.exists(best_path):
        best_model.save(best_path)
    _save_train_state(
        state_path,
        {
            "version": 2,
            "episode": episode,
            "best_score": best_score,
            "best_eval": best_eval,
            "rng_state": rng.getstate(),
            "replay_buffer": list(replay_buffer),
        },
    )

    elapsed_total = time.time() - t_start
    print(f"\nDone. {episode} episodes in {elapsed_total/60:.1f}min")
    print(f"Saved final: {args.out}")
    print(f"Saved state: {state_path}")
    final_bench = _run_benchmark(model, args.benchmark_games, bench_opponents)
    final_global, final_min, final_mean = _benchmark_stats(final_bench)
    final_score = _benchmark_score(final_bench)

    best_bench = _run_benchmark(best_model, args.benchmark_games, bench_opponents)
    best_global, best_min, best_mean = _benchmark_stats(best_bench)
    best_score_eval = _benchmark_score(best_bench)

    if (final_score, final_global, final_min) > (best_score_eval, best_global, best_min):
        best_model = model.copy()
        best_score = final_score
        best_eval = (final_score, final_global, final_min)
        best_model.save(best_path)
    else:
        best_score = best_score_eval
        best_eval = (best_score_eval, best_global, best_min)
        best_model.save(best_path)

    print(f"Saved best : {best_path}  score={best_score:.3f}")

    print("\n[Final benchmark]")
    for k, (wins, total) in best_bench.items():
        tag = " (holdout)" if k == args.holdout_opponent else ""
        print(f"  {k}{tag}: {wins}/{total}")
    print(f"  global: {best_global:.3f}")
    print(f"  min   : {best_min:.3f}")
    print(f"  mean  : {best_mean:.3f}")
    print(f"  score : {best_score_eval:.3f}")


if __name__ == "__main__":
    main()
