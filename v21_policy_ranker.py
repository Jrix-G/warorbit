"""V21 candidate ranker primitives.

This module is intentionally small and dependency-light.  V21's first learned
surface is a ranking model over legal candidate launches, not a raw action
decoder.  The ranker can run as a deterministic heuristic when no checkpoint is
provided, and as a linear NPZ model when weights are available.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np

import v15_fast_sim as fsim

ID, OWNER, X, Y, R, SHIPS, PROD = fsim.ID, fsim.OWNER, fsim.X, fsim.Y, fsim.R, fsim.SHIPS, fsim.PROD

FEATURE_NAMES = (
    "src_ship_share",
    "src_prod_norm",
    "target_ship_share",
    "target_prod_norm",
    "target_is_enemy",
    "target_is_neutral",
    "distance_norm",
    "send_fraction",
    "need_ratio",
    "step_frac",
    "is_4p",
    "my_ship_share",
    "my_prod_share",
    "my_planet_share",
    "target_owner_ship_share",
    "target_owner_prod_share",
)


@dataclass(frozen=True)
class RankedCandidate:
    shot: list
    score: float
    features: np.ndarray
    source_idx: int
    target_idx: int


class LinearCandidateRanker:
    """Small linear scorer over V21 candidate features.

    NPZ format:
      - `w`: float vector with len(FEATURE_NAMES)
      - optional `b`: scalar bias
    """

    def __init__(self, w: np.ndarray | None = None, b: float = 0.0) -> None:
        if w is None:
            self.w = default_weights()
        else:
            arr = np.asarray(w, dtype=np.float64).reshape(-1)
            if arr.shape[0] != len(FEATURE_NAMES):
                raise ValueError(f"ranker weight size {arr.shape[0]} != {len(FEATURE_NAMES)}")
            self.w = arr
        self.b = float(b)

    @classmethod
    def load(cls, path: str | Path | None) -> "LinearCandidateRanker":
        if path is None or str(path).strip() == "":
            return cls()
        p = Path(path)
        if not p.exists():
            return cls()
        data = np.load(p, allow_pickle=False)
        w = data["w"] if "w" in data.files else None
        b = float(data["b"]) if "b" in data.files else 0.0
        return cls(w=w, b=b)

    def score_features(self, features: np.ndarray) -> float:
        feat = np.asarray(features, dtype=np.float64).reshape(-1)
        if feat.shape[0] != len(FEATURE_NAMES):
            raise ValueError(f"feature size {feat.shape[0]} != {len(FEATURE_NAMES)}")
        return float(feat @ self.w + self.b)


def default_weights() -> np.ndarray:
    """Conservative prior: prefer useful attacks/support without all-in drift."""
    weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    weights[idx["src_ship_share"]] = 0.30
    weights[idx["src_prod_norm"]] = 0.08
    weights[idx["target_prod_norm"]] = 0.22
    weights[idx["target_is_enemy"]] = 0.35
    weights[idx["target_is_neutral"]] = 0.10
    weights[idx["distance_norm"]] = -0.28
    weights[idx["send_fraction"]] = -0.10
    weights[idx["need_ratio"]] = -0.22
    weights[idx["step_frac"]] = 0.04
    weights[idx["is_4p"]] = 0.04
    weights[idx["my_prod_share"]] = 0.15
    weights[idx["my_planet_share"]] = 0.08
    weights[idx["target_owner_ship_share"]] = -0.10
    weights[idx["target_owner_prod_share"]] = 0.12
    return weights


def rank_candidates(
    fs: fsim.FastState,
    player: int,
    shots: Iterable[list],
    ranker: LinearCandidateRanker | None = None,
) -> list[RankedCandidate]:
    scorer = ranker or LinearCandidateRanker()
    out: list[RankedCandidate] = []
    for shot in shots:
        resolved = resolve_shot(fs, player, shot)
        if resolved is None:
            continue
        source_idx, target_idx = resolved
        features = candidate_features(fs, player, shot, source_idx, target_idx)
        out.append(
            RankedCandidate(
                shot=[int(shot[0]), float(shot[1]), int(shot[2])],
                score=scorer.score_features(features),
                features=features,
                source_idx=source_idx,
                target_idx=target_idx,
            )
        )
    out.sort(key=lambda row: row.score, reverse=True)
    return out


def resolve_shot(fs: fsim.FastState, player: int, shot: list) -> tuple[int, int] | None:
    """Map `[src_id, angle, ships]` to source/target planet indices."""
    if not isinstance(shot, list) or len(shot) != 3:
        return None
    planets = fs.planets
    if len(planets) == 0:
        return None
    src_id = int(shot[0])
    ships = int(shot[2])
    if ships <= 0:
        return None
    source_idx = -1
    for i, row in enumerate(planets):
        if int(row[ID]) == src_id:
            source_idx = int(i)
            break
    if source_idx < 0 or int(planets[source_idx, OWNER]) != int(player):
        return None
    angle = float(shot[1])
    sx, sy = float(planets[source_idx, X]), float(planets[source_idx, Y])
    best_idx = -1
    best_diff = float("inf")
    for j, row in enumerate(planets):
        if j == source_idx:
            continue
        bearing = math.atan2(float(row[Y]) - sy, float(row[X]) - sx)
        diff = abs((bearing - angle + math.pi) % (2.0 * math.pi) - math.pi)
        if diff < best_diff:
            best_diff = diff
            best_idx = int(j)
    return (source_idx, best_idx) if best_idx >= 0 else None


def candidate_features(
    fs: fsim.FastState,
    player: int,
    shot: list,
    source_idx: int,
    target_idx: int,
) -> np.ndarray:
    planets = fs.planets
    owners = planets[:, OWNER].astype(np.int64) if len(planets) else np.zeros(0, dtype=np.int64)
    ships_by_owner, prod_by_owner, planet_by_owner = _player_totals(fs)
    n_players = max(2, int(getattr(fs, "n_players", 2) or 2))
    total_ships = max(1.0, float(sum(ships_by_owner)))
    total_prod = max(1.0, float(sum(prod_by_owner)))
    total_planets = max(1.0, float(sum(planet_by_owner)))

    src = planets[source_idx]
    tgt = planets[target_idx]
    tgt_owner = int(tgt[OWNER])
    src_ships = max(1.0, float(src[SHIPS]))
    send = max(0.0, float(shot[2]))
    dist = math.hypot(float(tgt[X]) - float(src[X]), float(tgt[Y]) - float(src[Y]))
    eta = max(1.0, dist / max(1.0, float(getattr(fs, "ship_speed", 6.0) or 6.0)))
    needed = max(1.0, float(tgt[SHIPS]) + max(0.0, float(tgt[PROD])) * eta + (5.0 if tgt_owner >= 0 else 3.0))
    target_owner_idx = tgt_owner if 0 <= tgt_owner < n_players else -1
    target_owner_ships = ships_by_owner[target_owner_idx] if target_owner_idx >= 0 else 0.0
    target_owner_prod = prod_by_owner[target_owner_idx] if target_owner_idx >= 0 else 0.0

    return np.asarray(
        [
            float(src[SHIPS]) / total_ships,
            min(3.0, max(0.0, float(src[PROD]) / 5.0)),
            float(tgt[SHIPS]) / total_ships,
            min(3.0, max(0.0, float(tgt[PROD]) / 5.0)),
            1.0 if tgt_owner >= 0 and tgt_owner != int(player) else 0.0,
            1.0 if tgt_owner < 0 else 0.0,
            min(2.0, dist / 100.0),
            min(1.5, send / src_ships),
            min(4.0, send / needed),
            min(1.0, float(getattr(fs, "step", 0)) / max(1.0, float(getattr(fs, "episode_steps", 500) or 500))),
            1.0 if n_players >= 4 else 0.0,
            ships_by_owner[int(player)] / total_ships if 0 <= int(player) < len(ships_by_owner) else 0.0,
            prod_by_owner[int(player)] / total_prod if 0 <= int(player) < len(prod_by_owner) else 0.0,
            planet_by_owner[int(player)] / total_planets if 0 <= int(player) < len(planet_by_owner) else 0.0,
            target_owner_ships / total_ships,
            target_owner_prod / total_prod,
        ],
        dtype=np.float32,
    )


def _player_totals(fs: fsim.FastState) -> tuple[list[float], list[float], list[float]]:
    n_players = max(2, int(getattr(fs, "n_players", 2) or 2))
    ships = [0.0 for _ in range(n_players)]
    prod = [0.0 for _ in range(n_players)]
    planets = [0.0 for _ in range(n_players)]
    for row in fs.planets:
        owner = int(row[OWNER])
        if 0 <= owner < n_players:
            ships[owner] += float(row[SHIPS])
            prod[owner] += float(row[PROD])
            planets[owner] += 1.0
    for fleet in fs.fleets:
        owner = int(fleet[fsim.F_OWNER])
        if 0 <= owner < n_players:
            ships[owner] += float(fleet[fsim.F_SHIPS])
    return ships, prod, planets
