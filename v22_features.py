"""V22 combo features.

The feature boundary is intentionally explicit: V22 learns over complete
combos, while reusing V21's stable per-shot feature extractor.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

import v15_fast_sim as fsim
import v21_policy_ranker

ID, OWNER, SHIPS = fsim.ID, fsim.OWNER, fsim.SHIPS

SHOT_FEATURE_NAMES = tuple(f"shot_mean_{name}" for name in v21_policy_ranker.FEATURE_NAMES)
SHOT_SUM_FEATURE_NAMES = tuple(f"shot_sum_{name}" for name in v21_policy_ranker.FEATURE_NAMES)
SHOT_MAX_FEATURE_NAMES = tuple(f"shot_max_{name}" for name in v21_policy_ranker.FEATURE_NAMES)

FEATURE_NAMES = (
    *SHOT_FEATURE_NAMES,
    *SHOT_SUM_FEATURE_NAMES,
    *SHOT_MAX_FEATURE_NAMES,
    "combo_size_norm",
    "unique_targets_norm",
    "enemy_target_share",
    "neutral_target_share",
    "own_target_share",
    "total_send_ship_share",
    "passive_score",
    "passive_delta",
    "det_score",
    "det_delta",
    "is_4p",
    "step_frac",
)


def combo_features(
    fs: fsim.FastState,
    player: int,
    combo: Iterable[list],
    *,
    passive_score: float = 0.0,
    passive_baseline: float = 0.0,
    det_score: float = 0.0,
    det_baseline: float = 0.0,
    max_combo: int = 4,
) -> np.ndarray:
    shots = [shot for shot in combo if isinstance(shot, list) and len(shot) == 3]
    shot_features: list[np.ndarray] = []
    target_owners: list[int] = []
    target_indices: list[int] = []
    total_send = 0.0
    total_my_ships = _my_ships(fs, player)
    for shot in shots:
        resolved = v21_policy_ranker.resolve_shot(fs, int(player), shot)
        if resolved is None:
            continue
        src_idx, tgt_idx = resolved
        feat = v21_policy_ranker.candidate_features(fs, int(player), shot, src_idx, tgt_idx).astype(np.float64)
        shot_features.append(feat)
        target_indices.append(int(tgt_idx))
        target_owners.append(int(fs.planets[tgt_idx, OWNER]))
        total_send += max(0.0, float(shot[2]))

    dim = len(v21_policy_ranker.FEATURE_NAMES)
    if shot_features:
        mat = np.vstack(shot_features)
        mean = mat.mean(axis=0)
        summ = mat.sum(axis=0)
        maxv = mat.max(axis=0)
    else:
        mean = np.zeros(dim, dtype=np.float64)
        summ = np.zeros(dim, dtype=np.float64)
        maxv = np.zeros(dim, dtype=np.float64)

    n = max(1, len(shots))
    enemy = sum(1 for owner in target_owners if owner >= 0 and owner != int(player))
    neutral = sum(1 for owner in target_owners if owner < 0)
    own = sum(1 for owner in target_owners if owner == int(player))
    step_frac = min(1.0, float(getattr(fs, "step", 0)) / max(1.0, float(getattr(fs, "episode_steps", 500) or 500)))
    tail = np.asarray(
        [
            min(1.5, len(shots) / max(1.0, float(max_combo))),
            len(set(target_indices)) / max(1.0, float(n)),
            enemy / float(n),
            neutral / float(n),
            own / float(n),
            min(2.0, total_send / max(1.0, total_my_ships)),
            float(passive_score),
            float(passive_score) - float(passive_baseline),
            float(det_score),
            float(det_score) - float(det_baseline),
            1.0 if int(getattr(fs, "n_players", 2) or 2) >= 4 else 0.0,
            step_frac,
        ],
        dtype=np.float64,
    )
    out = np.concatenate([mean, summ, maxv, tail]).astype(np.float32)
    if out.shape[0] != len(FEATURE_NAMES):
        raise ValueError(f"feature size {out.shape[0]} != {len(FEATURE_NAMES)}")
    return out


def _my_ships(fs: fsim.FastState, player: int) -> float:
    total = 0.0
    for row in fs.planets:
        if int(row[OWNER]) == int(player):
            total += float(row[SHIPS])
    for fleet in fs.fleets:
        if int(fleet[fsim.F_OWNER]) == int(player):
            total += float(fleet[fsim.F_SHIPS])
    return max(1.0, total)
