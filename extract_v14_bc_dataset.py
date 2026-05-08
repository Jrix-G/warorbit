#!/usr/bin/env python3
"""Extract V14 behavioral-cloning samples from raw Kaggle replay JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import v14_core


def _angle_dist(a: float, b: float) -> float:
    return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)


def _iter_replay_paths(inputs: list[Path]) -> Iterable[Path]:
    for item in inputs:
        if item.is_dir():
            yield from sorted(item.glob("*.json"))
        elif item.is_file():
            yield item


def _winner_indices(replay: dict[str, Any]) -> set[int]:
    rewards = replay.get("rewards", [])
    if not rewards:
        return set()
    best = max(rewards)
    return {i for i, r in enumerate(rewards) if r == best and r > 0}


def _wanted_indices(replay: dict[str, Any], agent_name: str | None, winner_only: bool) -> set[int]:
    if agent_name:
        agents = replay.get("info", {}).get("Agents", [])
        names = [str(a.get("Name", "")) for a in agents]
        return {i for i, name in enumerate(names) if name == agent_name}
    if winner_only:
        return _winner_indices(replay)
    return set(range(len(replay.get("steps", [[]])[0])))


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


def _extract_from_replay(path: Path, agent_name: str | None, winner_only: bool, max_samples: int | None):
    replay = json.load(path.open("r", encoding="utf-8"))
    wanted = _wanted_indices(replay, agent_name=agent_name, winner_only=winner_only)
    rows_x: list[np.ndarray] = []
    rows_y: list[int] = []
    rows_mask: list[np.ndarray] = []
    meta: list[tuple[int, int, int]] = []
    episode_id = int(replay.get("info", {}).get("EpisodeId", 0) or 0)

    for turn, step in enumerate(replay.get("steps", [])):
        for player_idx in sorted(wanted):
            if player_idx >= len(step):
                continue
            entry = step[player_idx]
            teacher_moves = entry.get("action") or []
            if not teacher_moves:
                continue
            obs = entry.get("observation") or {}
            if "step" not in obs:
                obs = dict(obs)
                obs["step"] = turn
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
            meta.append((episode_id, turn, player_idx))
            if max_samples is not None and len(rows_y) >= max_samples:
                return rows_x, rows_y, rows_mask, meta
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, default=[Path("replays/top1-05-05")])
    parser.add_argument("--output", type=Path, default=Path("replay_dataset/v14_bc_top1.npz"))
    parser.add_argument("--agent-name", default=None, help="Only extract a specific replay agent name.")
    parser.add_argument("--all-players", action="store_true", help="Extract all players instead of winners only.")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    xs: list[np.ndarray] = []
    ys: list[int] = []
    masks: list[np.ndarray] = []
    metas: list[tuple[int, int, int]] = []
    limit = args.max_samples if args.max_samples > 0 else None
    for path in _iter_replay_paths(args.inputs):
        remaining = None if limit is None else max(0, limit - len(ys))
        if remaining == 0:
            break
        rx, ry, rm, rmeta = _extract_from_replay(
            path,
            agent_name=args.agent_name,
            winner_only=not args.all_players,
            max_samples=remaining,
        )
        xs.extend(rx)
        ys.extend(ry)
        masks.extend(rm)
        metas.extend(rmeta)

    if not xs:
        raise SystemExit("No V14 BC samples extracted.")

    xpad, mpad = _pad_samples(xs, masks)
    y = np.asarray(ys, dtype=np.int64)
    meta = np.asarray(metas, dtype=np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, X=xpad, mask=mpad, y=y, meta=meta)
    print(f"samples={len(y)} max_candidates={xpad.shape[1]} features={xpad.shape[2]} saved={args.output}")


if __name__ == "__main__":
    main()
