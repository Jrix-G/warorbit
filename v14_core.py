"""Shared V14 candidate ranking utilities.

V14 keeps the V13/V12-style tactical candidate generator, but replaces the
12-feature scorer with a wider candidate ranker suitable for behavioral
cloning and later policy fine-tuning.

Feature layout (FEATURE_DIM=82):
  [0:12]   bot_v13 base features (preserved for BC label alignment)
  [12:17]  candidate type one-hot (attack/expand/defense/staging/noop)
  [17:30]  candidate-level stats (ships sent, source ratio, game state)
  [30:36]  global game state (players, garrison ratios, threat)
  [36:51]  target features (owner, ships, prod, distance)
  [51:64]  source + tactical flags
  [64]     my rank among all players (0=dominant, 1=weakest)
  [65]     inter-enemy fight flag (1=two enemies attacking each other)
  [66:71]  enemy[0] strongest: planets/40, ships/ts, prod/tp, fleets/ts, is_almost_dead
  [71:76]  enemy[1] middle
  [76:81]  enemy[2] weakest
  [81]     target enemy slot index / 2.0 (0=attacking strongest, 1=attacking weakest)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

import bot_v13


FEATURE_DIM = 82
HIDDEN1 = 128
HIDDEN2 = 64
MAX_ACTIONS = 8

# Minimum ships to bother sending in 4p political candidates
_4P_MIN_SEND = 4
# Fraction of source ships sent in focus_finish
_FOCUS_SEND_RATIO = 0.70
# Fraction of source ships sent in opportunistic_expand
_OPP_SEND_RATIO = 0.45
_OPENING_4P_TURNS = 50


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


def _per_enemy_stats(
    planets: list,
    fleets: list,
    me: int,
    total_ships: float,
    total_prod: float,
) -> list[dict]:
    """Per-enemy stats sorted by total strength descending. Max 3 entries."""
    enemy_ids = {int(p[1]) for p in planets if int(p[1]) not in (-1, me)}
    ts = max(1.0, total_ships)
    tp = max(1.0, total_prod)
    result = []
    for eid in enemy_ids:
        ep = [p for p in planets if int(p[1]) == eid]
        ef = [f for f in fleets if int(f[1]) == eid]
        planet_ships = sum(float(p[5]) for p in ep)
        fleet_ships = sum(float(f[6]) for f in ef)
        prod = sum(float(p[6]) for p in ep)
        result.append({
            "id": eid,
            "planets": len(ep),
            "ships": (planet_ships + fleet_ships) / ts,
            "prod": prod / tp,
            "fleets": fleet_ships / ts,
            "is_almost_dead": 1.0 if len(ep) <= 2 else 0.0,
        })
    result.sort(key=lambda x: -x["ships"])
    return result[:3]


def _inter_enemy_fight(planets: list, fleets: list, me: int) -> float:
    """1.0 if any enemy fleet targets another enemy's planet."""
    enemy_ids = {int(p[1]) for p in planets if int(p[1]) not in (-1, me)}
    if len(enemy_ids) < 2:
        return 0.0
    planet_owner: dict[int, int] = {int(p[0]): int(p[1]) for p in planets}
    for f in fleets:
        fowner = int(f[1])
        if fowner not in enemy_ids or len(f) <= 5:
            continue
        dst_owner = planet_owner.get(int(f[5]), -1)
        if dst_owner in enemy_ids and dst_owner != fowner:
            return 1.0
    return 0.0


# Candidate type → one-hot slot (focus_finish→attack, opportunistic_expand→expand)
_TYPE_SLOT: dict[str, int] = {
    "attack": 0,
    "focus_finish": 0,
    "pressure": 0,
    "expand": 1,
    "opportunistic_expand": 1,
    "frontier_expand": 1,
    "defense": 2,
    "staging": 3,
    "noop": 4,
}


def build_v14_features(obs: Any, candidate: dict) -> np.ndarray:
    planets = list(_get(obs, "planets", []) or [])
    fleets = list(_get(obs, "fleets", []) or [])
    me = int(_get(obs, "player", 0) or 0)
    step = int(_get(obs, "step", 0) or 0)
    stats = _world_stats(obs)
    moves = list(candidate.get("moves", []) or [])
    base = np.asarray(candidate.get("features", np.zeros(bot_v13.FEATURE_DIM)), dtype=np.float32)

    out = np.zeros(FEATURE_DIM, dtype=np.float32)
    out[: min(len(base), bot_v13.FEATURE_DIM)] = base[: bot_v13.FEATURE_DIM]

    # [12:17] type one-hot
    ctype = str(candidate.get("type", "noop"))
    slot = _TYPE_SLOT.get(ctype, -1)
    if slot >= 0:
        out[12 + slot] = 1.0

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

    # [17:30] candidate-level stats
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

    # [30:36] global game state
    out[30] = stats["n_players"] / 4.0
    out[31] = stats["my_planet_ships"] / my_total
    out[32] = stats["my_fleet_ships"] / my_total
    out[33] = stats["incoming_threat"] / my_total
    out[34] = stats["my_fleets"] / 120.0
    out[35] = stats["enemy_fleets"] / 240.0

    # [36:51] target features
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

    # [47:51] source features
    if first_src is not None:
        out[47] = float(first_src[5]) / my_total
        out[48] = float(first_src[6]) / 5.0
        enemy_planets_list = [p for p in planets if int(p[1]) not in (-1, me)]
        if enemy_planets_list:
            nd = min(_dist(first_src, ep) for ep in enemy_planets_list)
            out[49] = nd / 100.0
            out[50] = 1.0 if nd < 35.0 else 0.0

    # [51:64] tactical flags
    out[51] = 1.0 if step < 60 else 0.0
    out[52] = 1.0 if 60 <= step < 160 else 0.0
    out[53] = 1.0 if step >= 160 else 0.0
    out[54] = math.tanh(float(candidate.get("score_hint", 0.0)))
    out[55] = 1.0 if ctype in ("expand", "attack", "focus_finish", "opportunistic_expand") and out[42] > 0.0 else 0.0
    out[56] = 1.0 if ctype == "staging" and out[31] > 0.35 else 0.0
    out[57] = 1.0 if ctype == "defense" and out[33] > 0.05 else 0.0
    out[58] = min(1.0, src_prod / 20.0)
    out[59] = min(1.0, sent / 250.0)
    out[60] = 1.0 if sent >= 15.0 and step < 60 else 0.0
    out[61] = 1.0 if sent >= 60.0 and step >= 60 else 0.0
    out[62] = stats["enemy_total"] / total_ships
    out[63] = 1.0  # bias constant

    # [64:82] 4p political context
    enemies = _per_enemy_stats(planets, fleets, me, total_ships, total_prod)

    # [64] my rank among all players (0=dominant, 1=weakest)
    if enemies:
        n_stronger = sum(1 for e in enemies if e["ships"] > stats["my_total"] / total_ships)
        out[64] = float(n_stronger) / max(1, len(enemies))

    # [65] inter-enemy fight flag
    out[65] = _inter_enemy_fight(planets, fleets, me)

    # [66:81] per-enemy slots (sorted strongest→weakest)
    for i, e in enumerate(enemies):
        b = 66 + i * 5
        out[b + 0] = float(e["planets"]) / 40.0
        out[b + 1] = e["ships"]          # already normalized by total_ships
        out[b + 2] = e["prod"]           # already normalized by total_prod
        out[b + 3] = e["fleets"]         # already normalized by total_ships
        out[b + 4] = e["is_almost_dead"]

    # [81] which ranked enemy slot is being attacked (0=strongest/2, 1=weakest/2)
    if target is not None:
        target_owner = int(target[1])
        for i, e in enumerate(enemies):
            if e["id"] == target_owner:
                out[81] = float(i) / max(1, len(enemies) - 1) if len(enemies) > 1 else 0.0
                break

    return out


def _gen_4p_candidates(
    obs: Any,
    me: int,
    planets: list,
    fleets: list,
    step: int,
) -> list[dict]:
    """Political candidates for 4p: pressure weakest enemies and take nearby neutrals early."""
    enemy_ids = {int(p[1]) for p in planets if int(p[1]) not in (-1, me)}
    if len(enemy_ids) < 2:
        return []

    my_planets = [p for p in planets if int(p[1]) == me]
    if not my_planets:
        return []

    my_total = sum(float(p[5]) for p in my_planets)
    total_enemy_strength = 0.0
    enemy_data: list[tuple[float, int, list, list]] = []
    for eid in enemy_ids:
        ep = [p for p in planets if int(p[1]) == eid]
        ef = [f for f in fleets if int(f[1]) == eid]
        strength = sum(float(p[5]) for p in ep) + sum(float(f[6]) for f in ef)
        total_enemy_strength += strength
        enemy_data.append((strength, eid, ep, ef))
    enemy_data.sort()
    weakest_strength, weakest_id, weakest_planets, _ = enemy_data[0]
    strongest_strength, strongest_id, strongest_planets, _ = enemy_data[-1]
    candidates: list[dict] = []

    # opening_expand: V13/V12 often refuse close neutrals when the home planet
    # has only 10 ships. In 4p that creates fatal inactivity or early enemy hits.
    if step < _OPENING_4P_TURNS:
        neutrals = [p for p in planets if int(p[1]) == -1]
        opening_options: list[tuple[float, list, list, int, float]] = []
        for src in my_planets:
            src_ships = int(float(src[5]))
            if src_ships < 7:
                continue
            for tgt in neutrals:
                d = _dist(src, tgt)
                if d > 58.0 and float(tgt[6]) < 3.0:
                    continue
                defenders = float(tgt[5])
                prod = float(tgt[6])
                needed = int(math.ceil(defenders + max(1.0, prod)))
                if needed > src_ships:
                    if defenders <= src_ships:
                        needed = src_ships
                    else:
                        continue
                angle = math.atan2(float(tgt[3]) - float(src[3]), float(tgt[2]) - float(src[2]))
                roi = (3.0 * prod + max(0.0, 18.0 - defenders)) / max(8.0, d)
                opening_options.append((roi, src, tgt, needed, d))
        opening_options.sort(key=lambda item: item[0], reverse=True)
        for roi, src, tgt, needed, d in opening_options[:6]:
            angle = math.atan2(float(tgt[3]) - float(src[3]), float(tgt[2]) - float(src[2]))
            candidates.append({
                "type": "frontier_expand",
                "moves": [[int(src[0]), float(angle), int(needed)]],
                "score_hint": float(roi),
                "sources": {int(src[0])},
                "features": np.zeros(bot_v13.FEATURE_DIM, dtype=np.float32),
            })

    # focus_finish: coordinate multi-source strike on the weakest enemy once the
    # board is no longer opening-only, or when we are already clearly stronger.
    allow_finish = (
        step >= max(40, _OPENING_4P_TURNS - 10)
        or weakest_strength < my_total * 0.80
        or my_total > total_enemy_strength * 0.60
    )
    if allow_finish and weakest_planets and my_total > weakest_strength * 0.75:
        # Sort sources by available ships descending
        srcs_sorted = sorted(my_planets, key=lambda p: -float(p[5]))
        for tgt in weakest_planets[:4]:
            tx, ty = float(tgt[2]), float(tgt[3])
            moves: list[list] = []
            for src in srcs_sorted[:8]:
                ships_avail = int(float(src[5]) * _FOCUS_SEND_RATIO)
                if ships_avail < _4P_MIN_SEND:
                    continue
                angle = math.atan2(ty - float(src[3]), tx - float(src[2]))
                moves.append([int(src[0]), float(angle), ships_avail])
            total_sent = sum(m[2] for m in moves)
            if len(moves) >= 1 and total_sent >= float(tgt[5]) + max(4.0, float(tgt[6]) * 1.1):
                total_sent = sum(m[2] for m in moves)
                score_hint = float(tgt[6]) / max(1.0, float(tgt[5]) + 1.0) * min(2.0, total_sent / max(1.0, float(tgt[5])))
                candidates.append({
                    "type": "focus_finish",
                    "moves": moves,
                    "score_hint": score_hint,
                    "sources": {int(m[0]) for m in moves},
                    "features": np.zeros(bot_v13.FEATURE_DIM, dtype=np.float32),
                })

    # pressure: single-source strikes on weak enemy targets that can actually land.
    if step < 120 or my_total > total_enemy_strength * 0.55:
        for strength, eid, ep, _ in enemy_data:
            if not ep:
                continue
            # Pressure the weakest and strongest enemy, but only if the target is viable.
            for tgt in sorted(ep, key=lambda p: (float(p[5]), -float(p[6])))[:2]:
                best_src = max(
                    my_planets,
                    key=lambda src: float(src[5]) - 0.75 * _dist(src, tgt),
                )
                src_ships = int(float(best_src[5]))
                if src_ships < 8:
                    continue
                d = _dist(best_src, tgt)
                defenders = float(tgt[5])
                prod = float(tgt[6])
                needed = int(math.ceil(defenders + max(2.0, 1.0 + prod)))
                if needed > src_ships:
                    continue
                send = max(_4P_MIN_SEND, min(src_ships, int(math.ceil(needed * 1.05))))
                if send > src_ships or send < _4P_MIN_SEND:
                    continue
                angle = math.atan2(float(tgt[3]) - float(best_src[3]), float(tgt[2]) - float(best_src[2]))
                pressure_hint = (3.5 * prod + max(0.0, 20.0 - defenders)) / max(7.0, d)
                if eid == weakest_id:
                    pressure_hint += 1.5
                if eid == strongest_id and my_total > total_enemy_strength * 0.50:
                    pressure_hint += 0.8
                if step < 55:
                    pressure_hint += 0.7
                candidates.append({
                    "type": "pressure",
                    "moves": [[int(best_src[0]), float(angle), int(send)]],
                    "score_hint": float(pressure_hint),
                    "sources": {int(best_src[0])},
                    "features": np.zeros(bot_v13.FEATURE_DIM, dtype=np.float32),
                })

    # opportunistic_expand: grab neutral planets while enemies are fighting each other
    planet_owner: dict[int, int] = {int(p[0]): int(p[1]) for p in planets}
    inter_fight = False
    for f in fleets:
        fowner = int(f[1])
        if fowner not in enemy_ids or len(f) <= 5:
            continue
        dst_owner = planet_owner.get(int(f[5]), -1)
        if dst_owner in enemy_ids and dst_owner != fowner:
            inter_fight = True
            break

    if inter_fight or step < 90 or len(my_planets) <= 5:
        neutrals = [p for p in planets if int(p[1]) == -1]
        if neutrals and my_planets:
            # Precompute centroid of my empire for proximity sort
            cx = sum(float(p[2]) for p in my_planets) / len(my_planets)
            cy = sum(float(p[3]) for p in my_planets) / len(my_planets)
            neutrals_sorted = sorted(
                neutrals,
                key=lambda p: math.hypot(float(p[2]) - cx, float(p[3]) - cy),
            )
            for tgt in neutrals_sorted[:3]:
                tx, ty = float(tgt[2]), float(tgt[3])
                src = min(my_planets, key=lambda p: math.hypot(float(p[2]) - tx, float(p[3]) - ty))
                ships_avail = int(float(src[5]) * (_OPP_SEND_RATIO if inter_fight else 0.42))
                if ships_avail < _4P_MIN_SEND:
                    continue
                needed = int(float(tgt[5])) + _4P_MIN_SEND
                ships_avail = max(ships_avail, needed)
                if ships_avail > float(src[5]):
                    continue
                angle = math.atan2(ty - float(src[3]), tx - float(src[2]))
                candidates.append({
                    "type": "opportunistic_expand",
                    "moves": [[int(src[0]), float(angle), ships_avail]],
                    "score_hint": float(tgt[6]) / max(1.0, float(tgt[5]) + 1.0),
                    "sources": {int(src[0])},
                    "features": np.zeros(bot_v13.FEATURE_DIM, dtype=np.float32),
                })

    return candidates


def get_candidates(obs: Any) -> list[dict]:
    obs = obs_as_dict(obs)
    my_id = int(_get(obs, "player", 0) or 0)
    current_step = int(_get(obs, "step", 0) or 0)
    av = float(_get(obs, "angular_velocity", 0.03) or 0.03)
    planets = list(_get(obs, "planets", []) or [])
    fleets = list(_get(obs, "fleets", []) or [])
    if not planets:
        return []
    ip = bot_v13._build_initial_map(obs)
    arrivals = bot_v13._build_arrival_table(obs, ip, av, current_step)
    base = bot_v13.generate_all_candidates(obs, my_id, ip, av, current_step, arrivals)
    extra = _gen_4p_candidates(obs, my_id, planets, fleets, current_step)
    return base + extra


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
    top_is_noop = candidates[int(order[0])].get("type") == "noop"
    noop_margin = float(scores[int(order[0])] - scores[int(order[1])]) if top_is_noop and len(order) > 1 else 0.0
    for idx in order:
        cand = candidates[int(idx)]
        if cand.get("type") == "noop":
            if not actions and noop_margin > 0.5:
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
