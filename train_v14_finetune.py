#!/usr/bin/env python3
"""V14 policy fine-tuning — 4p-first, dense reward, self-play.

Key improvements over previous version:
- Dense per-step rewards with discount instead of terminal-only signal
- Self-play: opponents sampled from checkpoint pool of past versions
- Separate best checkpoints for 4p and 2p
- BC anchor only for 2p episodes
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

import v14_core
from opponents import ZOO
from local_simulator.official_fast import OfficialFastGame
from train_v14_bc import Adam, _backward


_GAMMA = 0.99          # discount factor for dense returns
_DENSE_WEIGHT = 0.6    # weight of dense reward vs terminal reward
_TERMINAL_WEIGHT = 0.4


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


def _terminal_reward(scores: list[float], my_slot: int) -> float:
    n = len(scores)
    if n == 4:
        ordered = sorted(range(n), key=lambda i: -float(scores[i]))
        rank = ordered.index(my_slot)
        return (1.0, 0.3, -0.3, -0.8)[rank]
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


def _dense_step_reward(obs: dict, me: int, n_players: int) -> float:
    """Per-step reward: economic dominance relative to fair share."""
    planets = list(obs.get("planets", []) or [])
    fleets = list(obs.get("fleets", []) or [])
    if not planets:
        return 0.0
    my_planets = [p for p in planets if int(p[1]) == me]
    my_prod = sum(float(p[6]) for p in my_planets)
    my_ships = sum(float(p[5]) for p in my_planets) + sum(float(f[6]) for f in fleets if int(f[1]) == me)
    total_prod = max(1.0, sum(float(p[6]) for p in planets if int(p[1]) >= 0))
    total_ships = max(1.0, sum(float(p[5]) for p in planets if int(p[1]) >= 0)
                      + sum(float(f[6]) for f in fleets if int(f[1]) >= 0))
    fair_share = 1.0 / n_players
    prod_edge = (my_prod / total_prod) - fair_share
    ship_edge = (my_ships / total_ships) - fair_share
    # Bonus at conversion threshold (≥13 planets around t=60-80)
    step = int(obs.get("step", 0) or 0)
    threshold_bonus = 0.0
    if 60 <= step <= 80 and len(my_planets) >= 13:
        threshold_bonus = 0.3
    return float(0.5 * prod_edge + 0.5 * ship_edge + threshold_bonus)


def _discounted_returns(step_rewards: list[float], terminal: float, gamma: float) -> list[float]:
    """Compute discounted returns blending dense per-step rewards with terminal."""
    n = len(step_rewards)
    if n == 0:
        return []
    returns = [0.0] * n
    # Terminal reward added at last step
    running = _TERMINAL_WEIGHT * terminal
    for t in range(n - 1, -1, -1):
        running = _DENSE_WEIGHT * step_rewards[t] + gamma * running
        returns[t] = running
    return returns


def _play_episode_task(task: tuple):
    weights, seed, max_steps, n_players, opponent_names, temperature, selfplay_weights_list = task
    model = v14_core.V14Scorer(weights=weights)
    rng = np.random.default_rng(seed)
    game = OfficialFastGame(
        n_players=n_players,
        seed=seed,
        episode_steps=max_steps,
        use_c_accel=True,
    )
    my_slot = int(rng.integers(0, n_players))

    # Build lineup: mix ZOO opponents and self-play opponents
    lineup: list[Callable | None] = []
    for player in range(n_players):
        if player == my_slot:
            lineup.append(None)
            continue
        # With probability 0.4 use a self-play opponent if pool available
        if selfplay_weights_list and rng.random() < 0.4:
            sp_w = selfplay_weights_list[int(rng.integers(0, len(selfplay_weights_list)))]
            sp_model = v14_core.V14Scorer(weights=sp_w)
            def _make_sp_agent(m):
                def _sp(obs, config=None):
                    obs = v14_core.obs_as_dict(obs)
                    cands = v14_core.get_candidates(obs)
                    if not cands:
                        return []
                    feats = v14_core.candidate_matrix(obs, cands)
                    scores = m.forward(feats)
                    best = int(np.argmax(scores))
                    return _candidate_actions(cands[best])
                return _sp
            lineup.append(_make_sp_agent(sp_model))
        else:
            zoo_agents = [ZOO[name] for name in opponent_names]
            lineup.append(zoo_agents[int(rng.integers(0, len(zoo_agents)))])

    records: list[tuple[np.ndarray, int]] = []  # (feats, chosen_idx)
    step_rewards: list[float] = []

    while not game.done:
        actions = []
        my_obs_this_step = None
        for player in range(n_players):
            obs = v14_core.obs_as_dict(game.observation(player))
            if player == my_slot:
                my_obs_this_step = obs
                candidates = v14_core.get_candidates(obs)
                if not candidates:
                    actions.append([])
                    continue
                feats = v14_core.candidate_matrix(obs, candidates)
                scores = model.forward(feats) / max(0.05, temperature)
                probs = v14_core.softmax(scores)
                idx = int(rng.choice(len(candidates), p=probs))
                records.append((feats, idx))
                chosen = candidates[idx]
                # Anti-noop bias during training: if noop chosen but real candidate exists, 25% chance to override
                if chosen.get("type") == "noop" and len(candidates) > 1 and rng.random() < 0.25:
                    non_noop = [(i, c) for i, c in enumerate(candidates) if c.get("type") != "noop"]
                    if non_noop:
                        nn_scores = np.array([scores[i] for i, _ in non_noop])
                        nn_probs = v14_core.softmax(nn_scores)
                        pick = int(rng.choice(len(non_noop), p=nn_probs))
                        idx = non_noop[pick][0]
                        records[-1] = (feats, idx)
                        chosen = candidates[idx]
                actions.append(_candidate_actions(chosen))
            else:
                actions.append(_call_agent(lineup[player], obs, game.configuration))
        game.step(actions)
        if my_obs_this_step is not None:
            step_rewards.append(_dense_step_reward(my_obs_this_step, my_slot, n_players))

    final_scores = game.scores()
    terminal = _terminal_reward(final_scores, my_slot)
    win = float(final_scores[my_slot]) > max(
        float(s) for i, s in enumerate(final_scores) if i != my_slot
    ) and final_scores[my_slot] > 0

    returns = _discounted_returns(step_rewards, terminal, _GAMMA)
    # Align returns with records (some steps may have no decision if no candidates)
    # records and step_rewards may differ in length; use min
    aligned = list(zip(records, returns[:len(records)]))

    return aligned, terminal, bool(win), n_players


def _bc_anchor_grad(model, X, mask, y, weight, rng, n):
    if X is None or len(y) == 0 or weight <= 0.0:
        return None
    idxs = rng.choice(len(y), size=min(n, len(y)), replace=False)
    grads = {k: np.zeros_like(v) for k, v in model.to_dict().items()}
    used = 0
    for idx in idxs:
        k = int(mask[idx].sum())
        label = int(y[idx])
        if k <= 0 or label >= k:
            continue
        feat_slice = X[idx, :k, : v14_core.FEATURE_DIM]
        if feat_slice.shape[1] < v14_core.FEATURE_DIM:
            pad = np.zeros((k, v14_core.FEATURE_DIM - feat_slice.shape[1]), dtype=np.float32)
            feat_slice = np.concatenate([feat_slice, pad], axis=1)
        scores, cache = model.forward_with_cache(feat_slice)
        probs = v14_core.softmax(scores)
        probs[label] -= 1.0
        sample = _backward(model, cache, probs * (weight / max(1, len(idxs))))
        for key in grads:
            grads[key] += sample[key]
        used += 1
    return grads if used > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--load", type=Path, default=Path("evaluations/scorer_v14.npz"))
    parser.add_argument("--out", type=Path, default=Path("evaluations/scorer_v14_ft.npz"))
    parser.add_argument("--bc-data", type=Path, default=Path("replay_dataset/v14_bc_top1.npz"))
    parser.add_argument("--bc-weight-4p", type=float, default=0.0)
    parser.add_argument("--bc-weight-2p", type=float, default=0.40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Lower default LR for stable dense-reward training.")
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--modes", nargs="*", default=["4p", "4p", "4p", "2p"])
    parser.add_argument("--selfplay-pool", type=int, default=8,
                        help="Max number of past checkpoints kept for self-play.")
    parser.add_argument("--selfplay-every", type=int, default=3,
                        help="Save checkpoint to self-play pool every N batches.")
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
    modes = tuple(4 if m == "4p" else 2 for m in args.modes)

    # Self-play checkpoint pool: seed with BC checkpoint to avoid empty pool early
    selfplay_pool: list[dict] = [{k: v.copy() for k, v in model.to_dict().items()}]

    deadline = time.time() + args.minutes * 60.0
    batch = 0
    games_total = 0
    best_train_wr = -1.0
    best_train_wr4 = -1.0
    best_train_wr2 = -1.0
    started = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    path_best_global = args.out.with_suffix(".best.npz")
    path_best_4p = args.out.with_name(args.out.stem + ".best4p.npz")
    path_best_2p = args.out.with_name(args.out.stem + ".best2p.npz")

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        while time.time() < deadline:
            weights = {k: v.copy() for k, v in model.to_dict().items()}

            # Snapshot pool for workers (serializable copies, limited size)
            sp_snapshot = [
                {k: v.copy() for k, v in w.items()}
                for w in selfplay_pool[-args.selfplay_pool:]
            ]

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
                    sp_snapshot,
                ))
            results = list(pool.map(_play_episode_task, tasks))

            aligned_list = [r[0] for r in results]   # list of [(feats, idx, ret)]
            rewards = [r[1] for r in results]         # terminal rewards for logging
            wins = [r[2] for r in results]
            nps = [r[3] for r in results]

            # Collect all (feats, idx, return) triples across all episodes
            all_triples: list[tuple[np.ndarray, int, float]] = []
            for aligned in aligned_list:
                for (feats, idx), ret in aligned:
                    all_triples.append((feats, idx, ret))

            # Normalize returns across batch
            if all_triples:
                rets = np.array([t[2] for t in all_triples], dtype=np.float32)
                rets = (rets - rets.mean()) / (rets.std() + 1e-6) if rets.std() > 1e-6 else rets - rets.mean()

                grads = {k: np.zeros_like(v) for k, v in model.to_dict().items()}
                for (feats, idx, _), a in zip(all_triples, rets):
                    scores_out, cache = model.forward_with_cache(feats)
                    probs = v14_core.softmax(scores_out)
                    probs[int(idx)] -= 1.0
                    sample = _backward(model, cache, probs * float(a) / max(1, len(all_triples)))
                    for key in grads:
                        grads[key] += sample[key]
            else:
                grads = {k: np.zeros_like(v) for k, v in model.to_dict().items()}

            # BC anchor (2p only)
            n4p = sum(1 for np_ in nps if np_ == 4)
            n2p = sum(1 for np_ in nps if np_ == 2)
            bc_weight = (n4p * args.bc_weight_4p + n2p * args.bc_weight_2p) / max(1, len(nps))
            bc_grads = _bc_anchor_grad(model, X, mask, y, bc_weight, rng, args.batch_size)
            if bc_grads is not None:
                for key in grads:
                    grads[key] += bc_grads[key]

            params = model.to_dict()
            opt.step(params, grads)
            model.W1, model.b1 = params["W1"], params["b1"]
            model.W2, model.b2 = params["W2"], params["b2"]
            model.W3, model.b3 = params["W3"], params["b3"]
            np.savez(args.out, **model.to_dict())

            # Add to self-play pool periodically
            if batch % args.selfplay_every == 0:
                selfplay_pool.append({k: v.copy() for k, v in model.to_dict().items()})
                if len(selfplay_pool) > args.selfplay_pool * 2:
                    # Keep a diverse spread, not just the latest
                    step = max(1, len(selfplay_pool) // args.selfplay_pool)
                    selfplay_pool = selfplay_pool[::step][-args.selfplay_pool:]

            games_total += len(results)
            wr = sum(wins) / max(1, len(wins))
            wr4_wins = [w for w, np_ in zip(wins, nps) if np_ == 4]
            wr2_wins = [w for w, np_ in zip(wins, nps) if np_ == 2]
            wr4 = sum(wr4_wins) / max(1, len(wr4_wins))
            wr2 = sum(wr2_wins) / max(1, len(wr2_wins))

            if wr > best_train_wr:
                best_train_wr = wr
                np.savez(path_best_global, **model.to_dict())
            if wr4_wins and wr4 > best_train_wr4:
                best_train_wr4 = wr4
                np.savez(path_best_4p, **model.to_dict())
            if wr2_wins and wr2 > best_train_wr2:
                best_train_wr2 = wr2
                np.savez(path_best_2p, **model.to_dict())

            elapsed = time.time() - started
            sp_size = len(selfplay_pool)
            print(
                f"[{elapsed:6.0f}s b{batch:04d}] games={games_total} "
                f"wr={sum(wins)}/{len(wins)} ({wr:.3f}) "
                f"wr2={wr2:.3f}(best={best_train_wr2:.3f}) "
                f"wr4={wr4:.3f}(best={best_train_wr4:.3f}) "
                f"reward={float(np.mean(rewards)):+.3f} "
                f"decisions={len(all_triples)} sp_pool={sp_size}",
                flush=True,
            )
            batch += 1


if __name__ == "__main__":
    main()
