#!/usr/bin/env python3
"""Conservative V14 policy fine-tuning.

This intentionally keeps an imitation anchor. It is not meant to replace BC;
it only nudges the ranker from the BC checkpoint using terminal outcomes.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

import v14_core
from c_engine import CGame
from opponents import ZOO
from train_v14_bc import Adam, _backward


def _call_agent(fn: Callable, obs, config) -> list:
    obs = v14_core.obs_as_dict(obs)
    try:
        move = fn(obs, config)
    except TypeError:
        move = fn(obs)
    return move if isinstance(move, list) else []


def _candidate_actions(candidate: dict) -> list[list]:
    if candidate.get("type") == "noop":
        return []
    return [[int(sid), float(angle), int(ships)] for sid, angle, ships in candidate.get("moves", []) or []]


def _episode_reward(scores: list[float], my_slot: int) -> float:
    ours = float(scores[my_slot])
    best_other = max(float(s) for i, s in enumerate(scores) if i != my_slot)
    ordered = sorted(((float(s), i) for i, s in enumerate(scores)), reverse=True)
    rank = next(i for i, (_, player) in enumerate(ordered) if player == my_slot)
    scale = max(1.0, sum(abs(float(s)) for s in scores))
    margin = np.clip((ours - best_other) / scale, -1.0, 1.0)
    if ours > best_other and ours > 0:
        return float(1.0 + 0.5 * margin)
    rank_bonus = (len(scores) - 1 - rank) / max(1, len(scores) - 1)
    return float(-0.6 + 0.25 * margin + 0.2 * rank_bonus)


def _play_episode_task(task: tuple[dict, int, int, int, tuple[str, ...], float]):
    weights, seed, max_steps, n_players, opponent_names, temperature = task
    model = v14_core.V14Scorer(weights=weights)
    rng = np.random.default_rng(seed)
    game = CGame(n_players=n_players, seed=seed, episode_steps=max_steps)
    my_slot = int(rng.integers(0, n_players))
    opponents = [ZOO[name] for name in opponent_names]
    lineup: list[Callable | None] = []
    for player in range(n_players):
        if player == my_slot:
            lineup.append(None)
        else:
            lineup.append(opponents[int(rng.integers(0, len(opponents)))])
    records = []
    while not game.done:
        actions = []
        for player in range(n_players):
            obs = v14_core.obs_as_dict(game.observation(player))
            if player == my_slot:
                candidates = v14_core.get_candidates(obs)
                if not candidates:
                    actions.append([])
                    continue
                feats = v14_core.candidate_matrix(obs, candidates)
                scores = model.forward(feats) / max(0.05, temperature)
                probs = v14_core.softmax(scores)
                idx = int(rng.choice(len(candidates), p=probs))
                records.append((feats, idx))
                actions.append(_candidate_actions(candidates[idx]))
            else:
                actions.append(_call_agent(lineup[player], obs, game.configuration))
        game.step(actions)
    scores = game.scores()
    win = float(scores[my_slot]) > max(float(s) for i, s in enumerate(scores) if i != my_slot) and scores[my_slot] > 0
    return records, _episode_reward(scores, my_slot), bool(win), n_players


def _bc_anchor_grad(model: v14_core.V14Scorer, X, mask, y, weight: float, rng: np.random.Generator, n: int):
    if X is None or len(y) == 0 or weight <= 0.0:
        return None
    idxs = rng.choice(len(y), size=min(n, len(y)), replace=False)
    grads = {k: np.zeros_like(v) for k, v in model.to_dict().items()}
    for idx in idxs:
        k = int(mask[idx].sum())
        label = int(y[idx])
        if k <= 0 or label >= k:
            continue
        scores, cache = model.forward_with_cache(X[idx, :k])
        probs = v14_core.softmax(scores)
        probs[label] -= 1.0
        sample = _backward(model, cache, probs * (weight / max(1, len(idxs))))
        for key in grads:
            grads[key] += sample[key]
    return grads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--load", type=Path, default=Path("evaluations/scorer_v14.npz"))
    parser.add_argument("--out", type=Path, default=Path("evaluations/scorer_v14_ft.npz"))
    parser.add_argument("--bc-data", type=Path, default=Path("replay_dataset/v14_bc_top1.npz"))
    parser.add_argument("--bc-weight", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=32, help="Games per optimizer batch.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--modes", nargs="*", default=["4p", "2p"])
    parser.add_argument("--opponents", nargs="*", default=[
        "greedy",
        "starter",
        "notebook_distance_prioritized",
        "notebook_physics_accurate",
        "notebook_orbitbotnext",
        "notebook_pascalledesma_orbitwork_v14",
    ])
    parser.add_argument("--seed", type=int, default=1414)
    args = parser.parse_args()

    model = v14_core.V14Scorer.load(args.load) if args.load.exists() else v14_core.V14Scorer()
    opt = Adam(model.to_dict(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    data = np.load(args.bc_data) if args.bc_data.exists() else None
    X = data["X"].astype(np.float32) if data is not None else None
    mask = data["mask"].astype(np.float32) if data is not None else None
    y = data["y"].astype(np.int64) if data is not None else np.zeros(0, dtype=np.int64)
    opponent_names = tuple(name for name in args.opponents if name in ZOO)
    if not opponent_names:
        raise SystemExit("No valid opponents selected.")
    modes = tuple(4 if mode == "4p" else 2 for mode in args.modes)
    deadline = time.time() + args.minutes * 60.0
    batch = 0
    games_total = 0
    best_train_wr = -1.0
    started = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        while time.time() < deadline:
            weights = {k: v.copy() for k, v in model.to_dict().items()}
            tasks = []
            for game_i in range(args.batch_size):
                n_players = modes[(batch * args.batch_size + game_i) % len(modes)]
                tasks.append((
                    weights,
                    int(rng.integers(0, 1_000_000_000)),
                    args.max_steps,
                    n_players,
                    opponent_names,
                    args.temperature,
                ))
            results = list(pool.map(_play_episode_task, tasks))
            trajectories = [r[0] for r in results]
            rewards = [r[1] for r in results]
            wins = [r[2] for r in results]
            nps = [r[3] for r in results]

            adv = np.asarray(rewards, dtype=np.float32)
            adv = (adv - adv.mean()) / (adv.std() + 1e-6) if adv.std() > 1e-6 else adv - adv.mean()
            grads = {k: np.zeros_like(v) for k, v in model.to_dict().items()}
            steps = 0
            for records, a in zip(trajectories, adv):
                for feats, idx in records:
                    scores, cache = model.forward_with_cache(feats)
                    probs = v14_core.softmax(scores)
                    probs[int(idx)] -= 1.0
                    sample = _backward(model, cache, probs * float(a))
                    for key in grads:
                        grads[key] += sample[key]
                    steps += 1
            if steps:
                for key in grads:
                    grads[key] /= steps
            bc_grads = _bc_anchor_grad(model, X, mask, y, args.bc_weight, rng, args.batch_size)
            if bc_grads is not None:
                for key in grads:
                    grads[key] += bc_grads[key]
            params = model.to_dict()
            opt.step(params, grads)
            model.W1, model.b1 = params["W1"], params["b1"]
            model.W2, model.b2 = params["W2"], params["b2"]
            model.W3, model.b3 = params["W3"], params["b3"]
            np.savez(args.out, **model.to_dict())
            games_total += len(results)
            wr = sum(wins) / max(1, len(wins))
            wr2 = sum(w for w, np_ in zip(wins, nps) if np_ == 2) / max(1, sum(1 for np_ in nps if np_ == 2))
            wr4 = sum(w for w, np_ in zip(wins, nps) if np_ == 4) / max(1, sum(1 for np_ in nps if np_ == 4))
            best_path = args.out.with_suffix(".best.npz")
            if wr > best_train_wr:
                best_train_wr = wr
                np.savez(best_path, **model.to_dict())
            elapsed = time.time() - started
            print(
                f"[{elapsed:6.0f}s b{batch:04d}] games={games_total} "
                f"wr={sum(wins)}/{len(wins)} ({wr:.3f}) "
                f"wr2={wr2:.3f} wr4={wr4:.3f} "
                f"reward={float(np.mean(rewards)):+.3f} steps={steps} "
                f"saved={args.out}",
                flush=True,
            )
            batch += 1


if __name__ == "__main__":
    main()
