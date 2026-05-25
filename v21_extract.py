"""V21 lightweight extraction helpers.

The heavy replay parser is intentionally not here.  This module provides the
safe, testable conversion boundary from already-materialized game rows/states
into canonical V21 samples.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import v15_eval
import v15_fast_sim as fsim
import v15_search
import v21_dataset
import v21_policy_ranker


def sample_from_state(
    fs: fsim.FastState,
    player: int,
    action: list,
    *,
    episode_id: str,
    source: str,
    outcome: float,
    esc: float | None = None,
    v7_move: list | None = None,
) -> dict[str, Any]:
    """Build one canonical V21 sample from a FastState and a played action."""
    player_i = int(player)
    candidates = candidates_from_state(fs, player_i, v7_move=v7_move)
    if not candidates:
        candidates = [{"shot": [], "source_idx": -1, "target_idx": -1, "features": []}]
    chosen = closest_candidate(fs, player_i, action, candidates)
    state = state_payload(fs, player_i)
    sample = {
        "state": state,
        "candidates": candidates,
        "chosen": chosen,
        "outcome": float(outcome),
        "esc": float(v15_eval.evaluate(fs, player_i, v15_eval.ESC) if esc is None else esc),
        "episode_id": str(episode_id),
        "player": player_i,
        "n_players": int(getattr(fs, "n_players", 2) or 2),
        "source": str(source),
    }
    return v21_dataset.normalize_sample(sample)


def candidates_from_state(fs: fsim.FastState, player: int, v7_move: list | None = None) -> list[dict[str, Any]]:
    """Enumerate V15 legal-ish candidates and attach V21 feature vectors."""
    shots = v15_search._enumerate_shots(fs, int(player), v7_move or [])
    ranked = v21_policy_ranker.rank_candidates(fs, int(player), shots)
    return [
        {
            "shot": row.shot,
            "source_idx": row.source_idx,
            "target_idx": row.target_idx,
            "features": [float(x) for x in row.features.tolist()],
        }
        for row in ranked
    ]


def closest_candidate(
    fs: fsim.FastState,
    player: int,
    action: list,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Choose the candidate closest to a played launch list."""
    if not candidates:
        raise ValueError("candidates must be non-empty")
    played = [shot for shot in action or [] if isinstance(shot, list) and len(shot) == 3]
    if not played:
        return candidates[0]
    best = candidates[0]
    best_dist = float("inf")
    for candidate in candidates:
        shot = candidate.get("shot", [])
        dist = _action_distance(played, shot)
        if dist < best_dist:
            best = candidate
            best_dist = dist
    return best


def samples_from_rows(rows: Iterable[dict[str, Any]], *, source: str = "rows") -> list[dict[str, Any]]:
    """Convert dict rows containing `fs`, `player`, `action`, and outcome data."""
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if "fs" not in row:
            raise ValueError(f"row {idx} missing fs")
        out.append(
            sample_from_state(
                row["fs"],
                int(row.get("player", 0)),
                row.get("action", []),
                episode_id=str(row.get("episode_id", f"row-{idx}")),
                source=str(row.get("source", source)),
                outcome=float(row.get("outcome", row.get("reward", 0.0))),
                esc=float(row["esc"]) if "esc" in row and row["esc"] is not None else None,
                v7_move=row.get("v7_move"),
            )
        )
    return out


def state_payload(fs: fsim.FastState, player: int) -> dict[str, Any]:
    """Compact JSON-safe state payload sufficient for ranker training."""
    return {
        "player": int(player),
        "n_players": int(getattr(fs, "n_players", 2) or 2),
        "step": int(getattr(fs, "step", 0)),
        "episode_steps": int(getattr(fs, "episode_steps", 500) or 500),
        "planets": _round_rows(fs.planets),
        "fleets": _round_rows(fs.fleets),
    }


def write_samples(path: str | Path, samples: Iterable[dict[str, Any]]) -> int:
    return v21_dataset.write_jsonl(path, samples)


def load_rows_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load already-extracted sample JSONL, normalizing through v21_dataset."""
    return v21_dataset.load_jsonl(path)


def _action_distance(played: list[list], candidate_shot: Any) -> float:
    if not isinstance(candidate_shot, list) or len(candidate_shot) != 3:
        return 1.0e9
    c_src = int(candidate_shot[0])
    c_ang = float(candidate_shot[1])
    c_ships = max(1.0, float(candidate_shot[2]))
    best = 1.0e9
    for shot in played:
        src_penalty = 0.0 if int(shot[0]) == c_src else 1000.0
        angle_diff = abs((float(shot[1]) - c_ang + math.pi) % (2.0 * math.pi) - math.pi)
        ships_diff = abs(float(shot[2]) - c_ships) / c_ships
        best = min(best, src_penalty + angle_diff + 0.25 * ships_diff)
    return best


def _round_rows(rows) -> list[list[float]]:
    return [[round(float(x), 6) for x in row] for row in rows.tolist()]


def _cmd_smoke(args: argparse.Namespace) -> dict[str, Any]:
    planets = [
        [0, 0, 10.0, 10.0, 3.0, 50.0, 2.0],
        [1, -1, 25.0, 10.0, 3.0, 10.0, 3.0],
        [2, 1, 40.0, 10.0, 3.0, 20.0, 2.0],
    ]
    fs = fsim.FastState(
        planets=__import__("numpy").array(planets, dtype=float),
        p_init=__import__("numpy").array([[p[2], p[3]] for p in planets], dtype=float),
        p_comet=__import__("numpy").zeros(3, dtype=bool),
        fleets=__import__("numpy").zeros((0, 7), dtype=float),
        comets=[],
        step=3,
        angular_velocity=0.0,
        next_fleet_id=0,
        episode_steps=120,
        ship_speed=6.0,
        n_players=2,
    )
    sample = sample_from_state(fs, 0, [[0, 0.0, 20]], episode_id="smoke", source="smoke", outcome=1.0)
    if args.out:
        write_samples(args.out, [sample])
    return {"samples": 1, "candidates": len(sample["candidates"]), "episode_id": sample["episode_id"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V21 lightweight extractor")
    sub = parser.add_subparsers(dest="cmd", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "smoke":
        print(json.dumps(_cmd_smoke(args), sort_keys=True))


if __name__ == "__main__":
    main()
