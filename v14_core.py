"""Shared V14 candidate ranking utilities.

V14 keeps the V13/V12-style tactical candidate generator, but replaces the
12-feature scorer with a wider candidate ranker suitable for behavioral
cloning and later policy fine-tuning.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

import bot_v13


FEATURE_DIM = 64
HIDDEN1 = 128
HIDDEN2 = 64
MAX_ACTIONS = 8


def _get(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def obs_as_dict(obs: Any) -> dict[str, Any]:
    if isinstance(obs, dict):
        return obs
    data = vars(obs).copy()
    data.setdefault("remainingOverageTime", 60.0)
    return data


def _angle_dist(a: float, b: float) -> float:
    return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)


def _dist(a: list, b: list) -> float:
    return math.hypot(float(a[2]) - float(b[2]), float(a[3]) - float(b[3]))


def _infer_target(planets: list, move: tuple[int, float, int] | list) -> list | None:
    if not isinstance(move, (list, tuple)) or len(move) < 3:
        return None
    src_id = int(move[0])
    angle = float(move[1])
    src = next((p for p in planets if int(p[0]) == src_id), None)
    if src is None:
        return None
    sx, sy = float(src[2]), float(src[3])
    dx, dy = math.cos(angle), math.sin(angle)
    best = None
    for p in planets:
        if int(p[0]) == src_id:
            continue
        vx = float(p[2]) - sx
        vy = float(p[3]) - sy
        proj = vx * dx + vy * dy
        if proj <= 0:
            continue
        perp = abs(vx * dy - vy * dx)
        threshold = float(p[4]) + 5.5
        if perp > threshold:
            continue
        score = (perp, proj)
        if best is None or score < best[0]:
            best = (score, p)
    return None if best is None else best[1]


def _world_stats(obs: Any) -> dict[str, float]:
    planets = list(_get(obs, "planets", []) or [])
    fleets = list(_get(obs, "fleets", []) or [])
    me = int(_get(obs, "player", 0) or 0)
    my_planets = [p for p in planets if int(p[1]) == me]
    enemy_planets = [p for p in planets if int(p[1]) not in (-1, me)]
    neutral_planets = [p for p in planets if int(p[1]) == -1]
    my_planet_ships = sum(float(p[5]) for p in my_planets)
    enemy_planet_ships = sum(float(p[5]) for p in enemy_planets)
    my_fleet_ships = sum(float(f[6]) for f in fleets if int(f[1]) == me)
    enemy_fleet_ships = sum(float(f[6]) for f in fleets if int(f[1]) != me)
    my_total = my_planet_ships + my_fleet_ships
    enemy_total = enemy_planet_ships + enemy_fleet_ships
    my_prod = sum(float(p[6]) for p in my_planets)
    enemy_prod = sum(float(p[6]) for p in enemy_planets)
    incoming_threat = 0.0
    my_ids = {int(p[0]) for p in my_planets}
    for f in fleets:
        if int(f[1]) != me and len(f) > 5 and int(f[5]) in my_ids:
            incoming_threat += float(f[6])
    owners = {int(p[1]) for p in planets if int(p[1]) >= 0}
    return {
        "me": float(me),
        "n_players": float(max(2, len(owners))),
        "my_planets": float(len(my_planets)),
        "enemy_planets": float(len(enemy_planets)),
        "neutral_planets": float(len(neutral_planets)),
        "my_total": my_total,
        "enemy_total": enemy_total,
        "total_ships": my_total + enemy_total,
        "my_prod": my_prod,
        "enemy_prod": enemy_prod,
        "total_prod": my_prod + enemy_prod,
        "my_planet_ships": my_planet_ships,
        "my_fleet_ships": my_fleet_ships,
        "enemy_fleet_ships": enemy_fleet_ships,
        "incoming_threat": incoming_threat,
        "my_fleets": float(sum(1 for f in fleets if int(f[1]) == me)),
        "enemy_fleets": float(sum(1 for f in fleets if int(f[1]) != me)),
    }


def build_v14_features(obs: Any, candidate: dict) -> np.ndarray:
    planets = list(_get(obs, "planets", []) or [])
    me = int(_get(obs, "player", 0) or 0)
    step = int(_get(obs, "step", 0) or 0)
    stats = _world_stats(obs)
    moves = list(candidate.get("moves", []) or [])
    base = np.asarray(candidate.get("features", np.zeros(bot_v13.FEATURE_DIM)), dtype=np.float32)

    out = np.zeros(FEATURE_DIM, dtype=np.float32)
    out[: min(len(base), bot_v13.FEATURE_DIM)] = base[:bot_v13.FEATURE_DIM]

    ctype = str(candidate.get("type", "noop"))
    type_names = ("attack", "expand", "defense", "staging", "noop")
    for i, name in enumerate(type_names):
        out[12 + i] = 1.0 if ctype == name else 0.0

    sent = float(sum(max(0, int(m[2])) for m in moves if len(m) >= 3))
    sources = {int(m[0]) for m in moves if len(m) >= 3}
    source_planets = [p for p in planets if int(p[0]) in sources]
    src_ships = sum(float(p[5]) for p in source_planets)
    src_prod = sum(float(p[6]) for p in source_planets)
    target = _infer_target(planets, moves[0]) if moves else None
    first_src = source_planets[0] if source_planets else None

    my_total = max(1.0, stats["my_total"])
    total_ships = max(1.0, stats["total_ships"])
    total_prod = max(1.0, stats["total_prod"])
    out[17] = min(1.0, len(moves) / 8.0)
    out[18] = min(2.0, sent / my_total)
    out[19] = sent / max(1.0, src_ships)
    out[20] = max([int(m[2]) / max(1.0, src_ships) for m in moves], default=0.0)
    out[21] = len(sources) / max(1.0, stats["my_planets"])
    out[22] = stats["my_planets"] / 40.0
    out[23] = stats["enemy_planets"] / 40.0
    out[24] = stats["neutral_planets"] / 40.0
    out[25] = stats["my_prod"] / 80.0
    out[26] = stats["enemy_prod"] / 120.0
    out[27] = stats["my_prod"] / total_prod
    out[28] = stats["my_total"] / total_ships
    out[29] = step / 500.0
    out[30] = 1.0 if stats["n_players"] >= 4 else 0.0
    out[31] = stats["my_planet_ships"] / my_total
    out[32] = stats["my_fleet_ships"] / my_total
    out[33] = stats["incoming_threat"] / my_total
    out[34] = stats["my_fleets"] / 120.0
    out[35] = stats["enemy_fleets"] / 240.0

    if target is not None:
        owner = int(target[1])
        out[36] = 1.0 if owner == -1 else 0.0
        out[37] = 1.0 if owner not in (-1, me) else 0.0
        out[38] = 1.0 if owner == me else 0.0
        out[39] = float(target[6]) / 5.0
        out[40] = float(target[5]) / my_total
        out[41] = sent / max(1.0, float(target[5]))
        out[42] = (sent - float(target[5])) / my_total
        out[43] = float(target[6]) / max(1.0, float(target[5]))
        if first_src is not None:
            d = _dist(first_src, target)
            out[44] = d / 100.0
            out[45] = 1.0 if d <= 42.0 else 0.0
            out[46] = 1.0 if d > 55.0 and step < 60 else 0.0
    if first_src is not None:
        out[47] = float(first_src[5]) / my_total
        out[48] = float(first_src[6]) / 5.0
        enemy_planets = [p for p in planets if int(p[1]) not in (-1, me)]
        if enemy_planets:
            nd = min(_dist(first_src, ep) for ep in enemy_planets)
            out[49] = nd / 100.0
            out[50] = 1.0 if nd < 35.0 else 0.0

    out[51] = 1.0 if step < 60 else 0.0
    out[52] = 1.0 if 60 <= step < 160 else 0.0
    out[53] = 1.0 if step >= 160 else 0.0
    out[54] = math.tanh(float(candidate.get("score_hint", 0.0)))
    out[55] = 1.0 if ctype in ("expand", "attack") and out[42] > 0.0 else 0.0
    out[56] = 1.0 if ctype == "staging" and out[31] > 0.35 else 0.0
    out[57] = 1.0 if ctype == "defense" and out[33] > 0.05 else 0.0
    out[58] = min(1.0, src_prod / 20.0)
    out[59] = min(1.0, sent / 250.0)
    out[60] = 1.0 if sent >= 15.0 and step < 60 else 0.0
    out[61] = 1.0 if sent >= 60.0 and step >= 60 else 0.0
    out[62] = stats["enemy_total"] / total_ships
    out[63] = 1.0
    return out


def get_candidates(obs: Any) -> list[dict]:
    obs = obs_as_dict(obs)
    my_id = int(_get(obs, "player", 0) or 0)
    current_step = int(_get(obs, "step", 0) or 0)
    av = float(_get(obs, "angular_velocity", 0.03) or 0.03)
    planets = list(_get(obs, "planets", []) or [])
    if not planets:
        return []
    ip = bot_v13._build_initial_map(obs)
    arrivals = bot_v13._build_arrival_table(obs, ip, av, current_step)
    return bot_v13.generate_all_candidates(obs, my_id, ip, av, current_step, arrivals)


def candidate_matrix(obs: Any, candidates: list[dict] | None = None) -> np.ndarray:
    candidates = get_candidates(obs) if candidates is None else candidates
    if not candidates:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.stack([build_v14_features(obs, c) for c in candidates]).astype(np.float32)


class V14Scorer:
    def __init__(self, weights: dict[str, np.ndarray] | None = None, seed: int = 14):
        if weights is not None:
            self.W1 = weights["W1"].astype(np.float32)
            self.b1 = weights["b1"].astype(np.float32)
            self.W2 = weights["W2"].astype(np.float32)
            self.b2 = weights["b2"].astype(np.float32)
            self.W3 = weights["W3"].astype(np.float32)
            self.b3 = weights["b3"].astype(np.float32)
        else:
            rng = np.random.default_rng(seed)
            self.W1 = (rng.standard_normal((FEATURE_DIM, HIDDEN1)) * math.sqrt(2.0 / FEATURE_DIM)).astype(np.float32)
            self.b1 = np.zeros(HIDDEN1, dtype=np.float32)
            self.W2 = (rng.standard_normal((HIDDEN1, HIDDEN2)) * math.sqrt(2.0 / HIDDEN1)).astype(np.float32)
            self.b2 = np.zeros(HIDDEN2, dtype=np.float32)
            self.W3 = (rng.standard_normal((HIDDEN2, 1)) * math.sqrt(2.0 / HIDDEN2)).astype(np.float32)
            self.b3 = np.zeros(1, dtype=np.float32)

    @classmethod
    def load(cls, path: str | Path) -> "V14Scorer":
        return cls(weights=dict(np.load(path)))

    def to_dict(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2, "W3": self.W3, "b3": self.b3}

    def forward(self, x: np.ndarray) -> np.ndarray:
        h1 = np.maximum(0.0, x @ self.W1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.W2 + self.b2)
        return (h2 @ self.W3 + self.b3).reshape(-1)

    def forward_with_cache(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(0.0, z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(0.0, z2)
        out = (h2 @ self.W3 + self.b3).reshape(-1)
        return out, {"x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2}


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    exp = np.exp(shifted)
    return exp / max(1e-12, float(exp.sum()))


def select_actions(candidates: list[dict], scores: np.ndarray, max_actions: int = MAX_ACTIONS) -> list[list]:
    if not candidates or scores.size == 0:
        return []
    order = np.argsort(-scores)
    used_sources: set[int] = set()
    actions: list[list] = []
    for idx in order:
        cand = candidates[int(idx)]
        if cand.get("type") == "noop":
            if not actions:
                return []
            continue
        if cand.get("sources", set()) & used_sources:
            continue
        for sid, angle, ships in cand.get("moves", []) or []:
            sid = int(sid)
            if sid in used_sources:
                continue
            actions.append([sid, float(angle), int(ships)])
            used_sources.add(sid)
            if len(actions) >= max_actions:
                return actions
    return actions
