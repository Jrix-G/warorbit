"""Counterfactual oracle for V21 candidate-ranker training.

The breakthrough hypothesis is simple and testable: instead of imitating one
observed action, train the ranker on same-state counterfactual comparisons.
For each state, enumerate legal candidates, simulate each candidate briefly,
and label the candidate with the best leaf value.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Iterable

import numpy as np

import v15_eval
import v15_fast_sim as fsim
import v15_search
import v21_dataset
import v21_extract
import v21_search


def oracle_sample_from_state(
    fs: fsim.FastState,
    player: int,
    *,
    episode_id: str,
    source: str = "oracle",
    horizon: int = 10,
    top_k: int = 12,
    v7_move: list | None = None,
) -> dict[str, Any]:
    """Return a V21 sample whose `chosen` candidate is the oracle best shot."""
    player_i = int(player)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    candidates = v21_extract.candidates_from_state(fs, player_i, v7_move=v7_move)[: int(top_k)]
    if not candidates:
        raise ValueError("no candidates available")

    baseline = float(v15_search._eval_combo(fs, player_i, [], int(horizon), False, v15_eval.ESC))
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        shot = candidate.get("shot", [])
        if not isinstance(shot, list) or len(shot) != 3:
            continue
        score = float(v15_search._eval_combo(fs, player_i, [shot], int(horizon), False, v15_eval.ESC))
        enriched = dict(candidate)
        enriched["oracle_score"] = score
        enriched["oracle_advantage"] = score - baseline
        scored.append(enriched)
    if not scored:
        raise ValueError("no scorable candidates available")

    chosen = max(scored, key=lambda row: (float(row["oracle_score"]), float(row["oracle_advantage"])))
    sample = {
        "state": v21_extract.state_payload(fs, player_i),
        "candidates": scored,
        "chosen": chosen,
        "outcome": float(chosen["oracle_score"]),
        "esc": baseline,
        "episode_id": str(episode_id),
        "player": player_i,
        "n_players": int(getattr(fs, "n_players", 2) or 2),
        "source": str(source),
    }
    return v21_dataset.normalize_sample(sample)


def combo_oracle_sample_from_state(
    fs: fsim.FastState,
    player: int,
    *,
    episode_id: str,
    source: str = "combo_oracle",
    horizon: int = 10,
    top_k: int = 12,
    max_combo: int = 4,
    beam_width: int = 32,
    min_advantage: float = 0.0,
    v7_move: list | None = None,
) -> dict[str, Any]:
    """Return one sample with soft targets for the best evaluated combo.

    V21's runtime search evaluates combinations.  This oracle therefore labels
    the atomic shots that make the best combo work, instead of forcing the
    ranker to imitate only the best isolated shot.
    """
    player_i = int(player)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if max_combo <= 0:
        raise ValueError("max_combo must be positive")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")

    candidates = v21_extract.candidates_from_state(fs, player_i, v7_move=v7_move)[: int(top_k)]
    if not candidates:
        raise ValueError("no candidates available")

    baseline = float(v15_search._eval_combo(fs, player_i, [], int(horizon), False, v15_eval.ESC))
    scored = _score_atomic_candidates(fs, player_i, candidates, int(horizon), baseline)
    if not scored:
        raise ValueError("no scorable candidates available")

    shots = [candidate["shot"] for candidate in scored]
    combos = v21_search._beam_combos(shots, max_combo=int(max_combo), beam_width=int(beam_width))
    if not combos:
        combos = [[shot] for shot in shots if v15_search._valid_combo([shot])]

    best_combo: list[list] = []
    best_score = baseline
    for combo in combos:
        if not v15_search._valid_combo(combo):
            continue
        score = float(v15_search._eval_combo(fs, player_i, combo, int(horizon), False, v15_eval.ESC))
        if score > best_score:
            best_combo = combo
            best_score = score
    if not best_combo or best_score <= baseline + float(min_advantage):
        raise ValueError("no profitable combo available")

    weights = _combo_component_weights(fs, player_i, best_combo, int(horizon), baseline, best_score)
    best_keys = {_shot_key(shot) for shot in best_combo}
    enriched: list[dict[str, Any]] = []
    for candidate in scored:
        key = _shot_key(candidate["shot"])
        row = dict(candidate)
        row["in_best_combo"] = key in best_keys
        row["target_weight"] = float(weights.get(key, 0.0))
        row["combo_oracle_score"] = best_score if key in best_keys else baseline
        row["combo_oracle_advantage"] = best_score - baseline if key in best_keys else 0.0
        enriched.append(row)

    chosen = max(
        enriched,
        key=lambda row: (
            float(row.get("target_weight", 0.0)),
            float(row.get("oracle_advantage", 0.0)),
            float(row.get("oracle_score", 0.0)),
        ),
    )
    sample = {
        "state": v21_extract.state_payload(fs, player_i),
        "candidates": enriched,
        "chosen": chosen,
        "outcome": float(best_score),
        "esc": baseline,
        "episode_id": str(episode_id),
        "player": player_i,
        "n_players": int(getattr(fs, "n_players", 2) or 2),
        "source": str(source),
    }
    return v21_dataset.normalize_sample(sample)


def oracle_samples_from_rows(
    rows: Iterable[dict[str, Any]],
    *,
    source: str = "oracle",
    horizon: int = 10,
    top_k: int = 12,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if "fs" not in row:
            raise ValueError(f"row {idx} missing fs")
        out.append(
            oracle_sample_from_state(
                row["fs"],
                int(row.get("player", 0)),
                episode_id=str(row.get("episode_id", f"oracle-{idx}")),
                source=str(row.get("source", source)),
                horizon=horizon,
                top_k=top_k,
                v7_move=row.get("v7_move"),
            )
        )
    return out


def write_oracle_samples(path: str, samples: Iterable[dict[str, Any]]) -> int:
    return v21_dataset.write_jsonl(path, samples)


def _score_atomic_candidates(
    fs: fsim.FastState,
    player: int,
    candidates: list[dict[str, Any]],
    horizon: int,
    baseline: float,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        shot = candidate.get("shot", [])
        if not isinstance(shot, list) or len(shot) != 3:
            continue
        score = float(v15_search._eval_combo(fs, int(player), [shot], int(horizon), False, v15_eval.ESC))
        enriched = dict(candidate)
        enriched["oracle_score"] = score
        enriched["oracle_advantage"] = score - float(baseline)
        scored.append(enriched)
    return scored


def _combo_component_weights(
    fs: fsim.FastState,
    player: int,
    combo: list[list],
    horizon: int,
    baseline: float,
    combo_score: float,
) -> dict[tuple[int, int, int], float]:
    weights: dict[tuple[int, int, int], float] = {}
    for idx, shot in enumerate(combo):
        key = _shot_key(shot)
        without = [other for j, other in enumerate(combo) if j != idx]
        if without:
            without_score = float(v15_search._eval_combo(fs, int(player), without, int(horizon), False, v15_eval.ESC))
        else:
            without_score = float(baseline)
        marginal = max(0.0, float(combo_score) - without_score)
        weights[key] = max(1.0e-6, marginal)
    return weights


def _shot_key(shot: list) -> tuple[int, int, int]:
    return (int(shot[0]), int(round(float(shot[1]) * 1_000_000)), int(shot[2]))


def _smoke_state() -> fsim.FastState:
    planets = np.array(
        [
            [0, 0, 10.0, 10.0, 3.0, 50.0, 2.0],
            [1, -1, 25.0, 10.0, 3.0, 10.0, 3.0],
            [2, 1, 40.0, 10.0, 3.0, 20.0, 2.0],
        ],
        dtype=np.float64,
    )
    return fsim.FastState(
        planets=planets,
        p_init=planets[:, [2, 3]].copy(),
        p_comet=np.zeros(3, dtype=bool),
        fleets=np.zeros((0, 7), dtype=np.float64),
        comets=[],
        step=3,
        angular_velocity=0.0,
        next_fleet_id=0,
        episode_steps=120,
        ship_speed=6.0,
        n_players=2,
    )


def _cmd_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "combo":
        sample = combo_oracle_sample_from_state(
            _smoke_state(),
            0,
            episode_id="oracle-smoke",
            horizon=args.horizon,
            top_k=args.top_k,
            max_combo=args.max_combo,
            beam_width=args.beam_width,
            min_advantage=args.min_advantage,
        )
    else:
        sample = oracle_sample_from_state(_smoke_state(), 0, episode_id="oracle-smoke", horizon=args.horizon, top_k=args.top_k)
    if args.out:
        write_oracle_samples(args.out, [sample])
    advantages = [float(c.get("oracle_advantage", 0.0)) for c in sample["candidates"]]
    target_mass = sum(float(c.get("target_weight", 0.0)) for c in sample["candidates"])
    return {
        "samples": 1,
        "candidates": len(sample["candidates"]),
        "best_advantage": max(advantages),
        "target_mass": target_mass,
        "chosen_score": float(sample["chosen"]["oracle_score"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V21 counterfactual oracle")
    sub = parser.add_subparsers(dest="cmd", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--out", default="")
    smoke.add_argument("--horizon", type=int, default=6)
    smoke.add_argument("--top-k", type=int, default=8)
    smoke.add_argument("--mode", choices=["atomic", "combo"], default="atomic")
    smoke.add_argument("--max-combo", type=int, default=4)
    smoke.add_argument("--beam-width", type=int, default=32)
    smoke.add_argument("--min-advantage", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "smoke":
        print(json.dumps(_cmd_smoke(args), sort_keys=True))


if __name__ == "__main__":
    main()
