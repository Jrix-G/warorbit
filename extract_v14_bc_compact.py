#!/usr/bin/env python3
"""Extract V14 BC samples from compact JSONL.gz replay dataset using CGame replay."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

import v14_core
from c_engine.cgame import CGame


def _angle_dist(a: float, b: float) -> float:
    return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)


def _candidate_teacher_distance(candidate: dict, teacher_moves: list) -> float:
    if candidate.get("type") == "noop":
        return 999.0
    best = 999.0
    for cmove in candidate.get("moves", []) or []:
        if not isinstance(cmove, (list, tuple)) or len(cmove) < 3:
            continue
        csrc, cangle, cships = int(cmove[0]), float(cmove[1]), max(1.0, float(cmove[2]))
        for tmove in teacher_moves:
            if not isinstance(tmove, (list, tuple)) or len(tmove) < 3:
                continue
            if int(tmove[0]) != csrc:
                continue
            tships = max(1.0, float(tmove[2]))
            angle_cost = _angle_dist(cangle, float(tmove[1]))
            ship_cost = abs(math.log(cships / tships))
            best = min(best, angle_cost + 0.35 * ship_cost)
    return best


def _label_candidate(candidates: list[dict], teacher_moves: list) -> int | None:
    if not teacher_moves:
        return None
    scored = [(_candidate_teacher_distance(c, teacher_moves), i) for i, c in enumerate(candidates)]
    scored.sort()
    if not scored or scored[0][0] > 0.75:
        return None
    return int(scored[0][1])


def _obs_to_dict(ns) -> dict:
    """Convert SimpleNamespace obs → plain dict for v14_core."""
    return {
        "planets": ns.planets,
        "fleets": ns.fleets,
        "step": ns.step,
        "player": ns.player,
        "initial_planets": ns.initial_planets,
        "next_fleet_id": ns.next_fleet_id,
        "remainingOverageTime": ns.remainingOverageTime,
    }


def _extract_episode(ep: dict, winner_only: bool) -> tuple[list, list, list, list]:
    init = ep["initial"]
    n_players = ep["n_players"]
    actions = ep["actions"]
    episode_id = ep.get("episode_id", 0)

    rewards = ep.get("rewards", [0] * n_players)
    best_reward = max(rewards) if rewards else 0
    if winner_only:
        wanted = {i for i, r in enumerate(rewards) if r == best_reward and r > 0}
    else:
        wanted = set(range(n_players))

    if not wanted:
        return [], [], [], []

    game = CGame.from_planets(
        n_players,
        init["planets"],
        init["angular_velocity"],
        initial_planets=init.get("initial_planets"),
    )

    rows_x: list[np.ndarray] = []
    rows_y: list[int] = []
    rows_mask: list[np.ndarray] = []
    meta: list[tuple[int, int, int]] = []

    for step_idx, step_actions in enumerate(actions):
        for p in sorted(wanted):
            teacher_moves = step_actions[p] if p < len(step_actions) else []
            if not teacher_moves:
                continue
            ns = game.observation(p)
            obs = _obs_to_dict(ns)
            candidates = v14_core.get_candidates(obs)
            if not candidates:
                continue
            label = _label_candidate(candidates, teacher_moves)
            if label is None:
                continue
            feats = v14_core.candidate_matrix(obs, candidates)
            rows_x.append(feats)
            rows_y.append(label)
            rows_mask.append(np.ones(len(candidates), dtype=np.float32))
            meta.append((episode_id, step_idx, p))

        game.step(step_actions)
        if game._done:
            break

    return rows_x, rows_y, rows_mask, meta


def _pad_samples(samples: list[np.ndarray], masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    max_k = max((x.shape[0] for x in samples), default=1)
    xout = np.zeros((len(samples), max_k, v14_core.FEATURE_DIM), dtype=np.float32)
    mout = np.zeros((len(samples), max_k), dtype=np.float32)
    for i, (x, m) in enumerate(zip(samples, masks)):
        k = x.shape[0]
        xout[i, :k] = x
        mout[i, :k] = m
    return xout, mout


def _worker(args_tuple):
    ep, winner_only = args_tuple
    return _extract_episode(ep, winner_only=winner_only)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("replay_dataset/compact/episodes_2026-05-03_200.jsonl.gz"))
    parser.add_argument("--output", type=Path, default=Path("replay_dataset/v14_bc_compact.npz"))
    parser.add_argument("--all-players", action="store_true", help="Extract all players, not just winners.")
    parser.add_argument("--mode", choices=["2p", "4p", "all"], default="all")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    winner_only = not args.all_players

    episodes = []
    with gzip.open(args.input, "rt") as f:
        for line in f:
            ep = json.loads(line)
            n = ep.get("n_players", 2)
            if args.mode == "2p" and n != 2:
                continue
            if args.mode == "4p" and n != 4:
                continue
            episodes.append(ep)

    tasks = [(ep, winner_only) for ep in episodes]
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        results = pool.map(_worker, tasks)

    xs: list[np.ndarray] = []
    ys: list[int] = []
    masks_list: list[np.ndarray] = []
    metas: list[tuple[int, int, int]] = []
    n2p = n4p = 0

    for ep, (rx, ry, rm, rmeta) in zip(episodes, results):
        xs.extend(rx)
        ys.extend(ry)
        masks_list.extend(rm)
        metas.extend(rmeta)
        if ep.get("n_players", 2) == 2:
            n2p += 1
        else:
            n4p += 1

    print(f"BC samples: {len(ys)} from {n2p} 2p replays, {n4p} 4p replays")
    if n2p + n4p > 0:
        print(f"4p ratio: {n4p / (n2p + n4p) * 100:.1f}%")

    if not xs:
        raise SystemExit("No samples extracted.")

    xpad, mpad = _pad_samples(xs, masks_list)
    y = np.asarray(ys, dtype=np.int64)
    meta = np.asarray(metas, dtype=np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, X=xpad, y=y, mask=mpad, meta=meta)
    print(f"Saved → {args.output}  shape={xpad.shape}")


if __name__ == "__main__":
    main()
