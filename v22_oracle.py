"""V22 combo oracle.

This module labels complete combos with passive and active deterministic
continuation scores.  It is the data-generation target for V22/V23 training.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

import bot_v7
import v15_eval
import v15_fast_sim as fsim
import v15_search
import v21_extract
import v21_policy_ranker
import v21_search
import v22_dataset
import v22_features


def sample_from_state(
    fs: fsim.FastState,
    player: int,
    *,
    episode_id: str,
    source: str = "v22_oracle",
    horizon: int = 10,
    det_horizon: int = 8,
    top_k: int = 10,
    beam_width: int = 32,
    max_combo: int = 4,
    min_advantage: float = 0.0,
    v7_move: list | None = None,
) -> dict[str, Any]:
    player_i = int(player)
    if top_k <= 0 or beam_width <= 0 or max_combo <= 0:
        raise ValueError("top_k, beam_width and max_combo must be positive")
    if horizon <= 0 or det_horizon <= 0:
        raise ValueError("horizons must be positive")

    shots = _ranked_shots(fs, player_i, top_k=top_k, v7_move=v7_move)
    combos = v21_search._beam_combos(shots, max_combo=max_combo, beam_width=beam_width)
    if not combos:
        raise ValueError("no combos available")

    passive_baseline = float(v15_search._eval_combo(fs, player_i, [], horizon, False, v15_eval.ESC))
    det_baseline = _eval_combo_det(fs, player_i, [], det_horizon)
    baseline = max(passive_baseline, det_baseline)
    rows: list[dict[str, Any]] = []
    for combo in combos:
        if not v15_search._valid_combo(combo):
            continue
        passive = float(v15_search._eval_combo(fs, player_i, combo, horizon, False, v15_eval.ESC))
        det = _eval_combo_det(fs, player_i, combo, det_horizon)
        score = max(passive, det)
        feat = v22_features.combo_features(
            fs,
            player_i,
            combo,
            passive_score=passive,
            passive_baseline=passive_baseline,
            det_score=det,
            det_baseline=det_baseline,
            max_combo=max_combo,
        )
        rows.append(
            {
                "shots": _clean_combo(combo),
                "features": [float(x) for x in feat.tolist()],
                "score": score,
                "passive_score": passive,
                "det_score": det,
                "target_weight": max(0.0, score - baseline),
            }
        )
    if not rows:
        raise ValueError("no scorable combos")
    best_idx = max(range(len(rows)), key=lambda i: float(rows[i]["score"]))
    if float(rows[best_idx]["score"]) <= baseline + float(min_advantage):
        raise ValueError("no profitable combo")
    if sum(float(row["target_weight"]) for row in rows) <= 0.0:
        rows[best_idx]["target_weight"] = 1.0

    return v22_dataset.normalize_sample(
        {
            "state": v21_extract.state_payload(fs, player_i),
            "combos": rows,
            "chosen": best_idx,
            "baseline": baseline,
            "episode_id": str(episode_id),
            "player": player_i,
            "n_players": int(getattr(fs, "n_players", 2) or 2),
            "source": str(source),
        }
    )


def _ranked_shots(fs: fsim.FastState, player: int, *, top_k: int, v7_move: list | None = None) -> list[list]:
    if v7_move is None:
        try:
            obs = v15_search.state_to_obs(fs, int(player))
            v7_move = bot_v7.agent(obs, None)
        except Exception:
            v7_move = []
    atomic = v15_search._enumerate_shots(fs, int(player), v7_move or [])
    ranked = v21_policy_ranker.rank_candidates(fs, int(player), atomic)
    return [row.shot for row in ranked[: int(top_k)]]


def _eval_combo_det(fs: fsim.FastState, player: int, combo: list, horizon: int) -> float:
    actions = v15_search._det_policy(fs)
    actions[int(player)] = list(combo)
    st = fsim.step(fs, actions)
    for _ in range(max(1, int(horizon)) - 1):
        if st.done:
            break
        st = fsim.step(st, v15_search._det_policy(st))
    return float(v15_eval.evaluate(st, int(player), v15_eval.ESC))


def _clean_combo(combo: list) -> list[list]:
    out: list[list] = []
    for shot in combo:
        if isinstance(shot, list) and len(shot) == 3:
            out.append([int(shot[0]), float(shot[1]), int(shot[2])])
    return out


def _smoke_state() -> fsim.FastState:
    planets = np.array(
        [
            [0, 0, 10.0, 10.0, 3.0, 50.0, 2.0],
            [1, -1, 25.0, 10.0, 3.0, 10.0, 3.0],
            [2, 1, 40.0, 10.0, 3.0, 20.0, 2.0],
            [3, 0, 10.0, 25.0, 3.0, 45.0, 2.0],
        ],
        dtype=np.float64,
    )
    return fsim.FastState(
        planets=planets,
        p_init=planets[:, [2, 3]].copy(),
        p_comet=np.zeros(len(planets), dtype=bool),
        fleets=np.zeros((0, 7), dtype=np.float64),
        comets=[],
        step=3,
        angular_velocity=0.0,
        next_fleet_id=0,
        episode_steps=120,
        ship_speed=6.0,
        n_players=2,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V22 combo oracle")
    parser.add_argument("--out", default="")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--det-horizon", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = sample_from_state(
        _smoke_state(),
        0,
        episode_id="v22-smoke",
        horizon=args.horizon,
        det_horizon=args.det_horizon,
        top_k=args.top_k,
        beam_width=args.beam_width,
    )
    if args.out:
        v22_dataset.write_jsonl(args.out, [sample])
    print(json.dumps({"samples": 1, "combos": len(sample["combos"]), "chosen": sample["chosen"]}, sort_keys=True))


if __name__ == "__main__":
    main()
