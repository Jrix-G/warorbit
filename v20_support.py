"""V20 4p support/supply-chain overlay.

The support layer is intentionally defensive: it only proposes own-planet
reinforcement pairs.  V20 search still evaluates the resulting shots before
executing them, so bad support reads fall back to the existing tactical layer.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

import v15_fast_sim as fsim
import v20_macro

ID, OWNER, X, Y, R, SHIPS, PROD = fsim.ID, fsim.OWNER, fsim.X, fsim.Y, fsim.R, fsim.SHIPS, fsim.PROD
F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = (
    fsim.F_OWNER,
    fsim.F_X,
    fsim.F_Y,
    fsim.F_ANGLE,
    fsim.F_FROM,
    fsim.F_SHIPS,
)


@dataclass(frozen=True)
class SupportScore:
    idx: int
    threat: float
    incoming: float
    front: float
    low: float
    score: float


def choose_intent(
    fs: fsim.FastState,
    player: int,
    base_intent: v20_macro.MacroIntent | None = None,
) -> v20_macro.MacroIntent | None:
    """Return a support intent when 4p supply-chain play should pre-empt macro."""
    if _flag("V20_DISABLE_SUPPORT") or _flag("V20_DISABLE_SUPPORT_INTENT"):
        return None
    n_players = max(2, int(getattr(fs, "n_players", 2) or 2))
    if n_players < 4 and not _flag("V20_ENABLE_SUPPORT_2P"):
        return None
    planets = fs.planets
    if len(planets) == 0:
        return None
    owners = planets[:, OWNER].astype(np.int64)
    mine = [int(i) for i in np.where(owners == int(player))[0]]
    if len(mine) < 2:
        return None

    targets = _support_targets(fs, player, limit=8)
    sources = _support_sources(fs, player, limit=8)
    if not targets or not sources:
        return None

    best = targets[0]
    step = int(fs.step)
    remaining = max(0, int(fs.episode_steps) - step)
    leader_feed_risk = _leader_feed_risk(fs, player, base_intent)
    late = remaining <= _env_int("V20_SUPPORT_LATE_REMAINING", 50) or step >= _env_int("V20_SUPPORT_LATE_STEP", 110)

    if (
        best.low < _env_float("V20_SUPPORT_MIN_TARGET_LOW", 0.20)
        and best.incoming < _env_float("V20_SUPPORT_MIN_TARGET_INCOMING", 6.0)
        and best.front < _env_float("V20_SUPPORT_MIN_TARGET_FRONT", 0.25)
    ):
        return None

    emergency_threat = _env_float("V20_SUPPORT_EMERGENCY_THREAT", 0.86)
    if best.threat >= emergency_threat:
        return _intent(fs, "support_emergency", best, targets, sources, 0.72, "incoming_or_front_emergency")

    if base_intent is not None and base_intent.name == "pressure_leader":
        return None

    if (
        late
        and not _flag("V20_DISABLE_LATE_MASS_SUPPORT")
        and len(sources) >= 2
        and best.score >= _env_float("V20_SUPPORT_LATE_SCORE", 0.55)
        and (best.low >= 0.24 or best.front >= 0.34 or best.incoming >= 8.0)
    ):
        return _intent(fs, "support_late_mass", best, targets, sources, 0.66, "late_mass_rear_to_front_support")

    if (
        leader_feed_risk
        and not _flag("V20_DISABLE_AVOID_FEED_LEADER")
        and best.score >= _env_float("V20_SUPPORT_FEED_GUARD_SCORE", 0.50)
    ):
        return _intent(fs, "support_supply_chain", best, targets, sources, 0.62, "avoid_feeding_current_leader")

    if (
        not _flag("V20_DISABLE_REAR_TO_FRONT")
        and best.score >= _env_float("V20_SUPPORT_SCORE", 0.58)
        and _has_rear_to_front_route(fs, player, sources, targets)
    ):
        return _intent(fs, "support_supply_chain", best, targets, sources, 0.56, "rear_to_front_supply_chain")

    return None


def candidate_pairs(
    fs: fsim.FastState,
    player: int,
    intent: v20_macro.MacroIntent,
    *,
    max_pairs: int = 24,
) -> list[v20_macro.MacroPair]:
    """Return rear-to-front own-planet transfer pairs for support intents."""
    if _flag("V20_DISABLE_SUPPORT") or _flag("V20_DISABLE_SUPPORT_CANDIDATES") or max_pairs <= 0:
        return []
    sources = _rank_sources(fs, player, intent)
    targets = _rank_targets(fs, player, intent)
    if not sources or not targets:
        return []
    pairs: list[v20_macro.MacroPair] = []
    seen: set[tuple[int, int]] = set()
    for tgt_idx, tgt_score in targets:
        for src_idx, src_score in sources:
            if src_idx == tgt_idx:
                continue
            key = (src_idx, tgt_idx)
            if key in seen:
                continue
            seen.add(key)
            dist = _dist(fs.planets[src_idx], fs.planets[tgt_idx])
            long_supply = min(0.18, dist / 500.0)
            bias = tgt_score + src_score + long_supply - 0.006 * dist
            pairs.append(v20_macro.MacroPair(src_idx, tgt_idx, float(bias), intent.name))
            if len(pairs) >= max_pairs:
                return _normalise(pairs)
    return _normalise(pairs)


def is_support_intent(intent: v20_macro.MacroIntent | None) -> bool:
    return intent is not None and intent.name.startswith("support_")


def _intent(
    fs: fsim.FastState,
    name: str,
    best: SupportScore,
    targets: list[SupportScore],
    sources: list[tuple[int, float]],
    base_conf: float,
    reason: str,
) -> v20_macro.MacroIntent:
    conf = _clip01(base_conf + 0.22 * best.threat + 0.10 * min(1.0, best.incoming / 40.0))
    return v20_macro.MacroIntent(
        name,
        conf,
        target_owner=None,
        target_ids=tuple(int(fs.planets[s.idx, ID]) for s in targets[:8]),
        source_ids=tuple(int(fs.planets[idx, ID]) for idx, _ in sources[:8]),
        pressure=best.score,
        reason=reason,
    )


def _support_targets(fs: fsim.FastState, player: int, *, limit: int) -> list[SupportScore]:
    rows = _rank_targets(fs, player, v20_macro.MacroIntent("support_probe", 1.0))
    out: list[SupportScore] = []
    for idx, score in rows[:limit]:
        threat, incoming, front, low = _target_components(fs, player, idx)
        out.append(SupportScore(idx, threat, incoming, front, low, score))
    return out


def _support_sources(fs: fsim.FastState, player: int, *, limit: int) -> list[tuple[int, float]]:
    return _rank_sources(fs, player, v20_macro.MacroIntent("support_probe", 1.0))[:limit]


def _rank_targets(
    fs: fsim.FastState,
    player: int,
    intent: v20_macro.MacroIntent,
) -> list[tuple[int, float]]:
    planets = fs.planets
    owners = planets[:, OWNER].astype(np.int64) if len(planets) else np.zeros(0, dtype=np.int64)
    preferred = set(int(x) for x in intent.target_ids)
    rows: list[tuple[int, float]] = []
    for idx in np.where(owners == int(player))[0]:
        idx = int(idx)
        threat, incoming, front, low = _target_components(fs, player, idx)
        pref = 0.18 if idx in preferred or int(planets[idx, ID]) in preferred else 0.0
        score = 0.32 * threat + 0.34 * front + 0.30 * low + 0.012 * float(planets[idx, PROD]) + pref
        if intent.name == "support_late_mass":
            score += 0.12 * front + 0.08 * min(1.0, incoming / 35.0)
        rows.append((idx, float(score)))
    rows.sort(key=lambda kv: kv[1], reverse=True)
    return rows


def _rank_sources(
    fs: fsim.FastState,
    player: int,
    intent: v20_macro.MacroIntent,
) -> list[tuple[int, float]]:
    planets = fs.planets
    owners = planets[:, OWNER].astype(np.int64) if len(planets) else np.zeros(0, dtype=np.int64)
    enemy_idx = np.where((owners >= 0) & (owners != int(player)))[0]
    target_pref = set(int(x) for x in intent.source_ids)
    rows: list[tuple[int, float]] = []
    for idx in np.where(owners == int(player))[0]:
        idx = int(idx)
        p = planets[idx]
        reserve = _source_reserve(fs, player, idx)
        surplus = max(0.0, float(p[SHIPS]) - reserve)
        if surplus < _env_float("V20_SUPPORT_MIN_SURPLUS", 8.0):
            continue
        near_enemy = _nearest_dist(planets, idx, enemy_idx)
        rear = min(1.0, near_enemy / 58.0) if len(enemy_idx) else 0.55
        incoming = _incoming_enemy_ships(fs, player, idx, max_eta=18)
        source_safety = 1.0 / (1.0 + max(0.0, incoming - reserve) / 20.0)
        pref = 0.16 if idx in target_pref or int(p[ID]) in target_pref else 0.0
        score = 0.36 * math.log1p(surplus) + 0.34 * rear + 0.18 * source_safety + 0.018 * float(p[PROD]) + pref
        rows.append((idx, float(score)))
    rows.sort(key=lambda kv: kv[1], reverse=True)
    return rows


def _target_components(fs: fsim.FastState, player: int, idx: int) -> tuple[float, float, float, float]:
    planets = fs.planets
    owners = planets[:, OWNER].astype(np.int64)
    enemy_idx = np.where((owners >= 0) & (owners != int(player)))[0]
    p = planets[idx]
    incoming = 0.0 if _flag("V20_DISABLE_INCOMING_THREAT") else _incoming_enemy_ships(fs, player, idx, max_eta=24)
    near_enemy = _nearest_dist(planets, idx, enemy_idx)
    front = max(0.0, 52.0 - near_enemy) / 52.0 if len(enemy_idx) else 0.0
    desired = 15.0 + float(p[PROD]) * 3.8 + front * 14.0
    low = _clip01((desired + incoming * 0.65 - float(p[SHIPS])) / 44.0)
    threat = _clip01(0.52 * low + 0.32 * front + 0.16 * min(1.0, incoming / 32.0))
    return threat, incoming, front, low


def _incoming_enemy_ships(fs: fsim.FastState, player: int, idx: int, *, max_eta: int) -> float:
    if len(fs.fleets) == 0:
        return 0.0
    p = fs.planets[idx]
    total = 0.0
    for f in fs.fleets:
        owner = int(f[F_OWNER])
        if owner < 0 or owner == int(player):
            continue
        dx = float(p[X] - f[F_X])
        dy = float(p[Y] - f[F_Y])
        along = dx * math.cos(float(f[F_ANGLE])) + dy * math.sin(float(f[F_ANGLE]))
        if along <= 0.0:
            continue
        eta = along / max(0.1, float(fs.ship_speed))
        if eta > max_eta:
            continue
        perp = abs(dx * math.sin(float(f[F_ANGLE])) - dy * math.cos(float(f[F_ANGLE])))
        if perp <= float(p[R]) + 1.2:
            total += float(f[F_SHIPS]) * (1.0 - 0.45 * eta / max(1.0, float(max_eta)))
    return total


def _leader_feed_risk(
    fs: fsim.FastState,
    player: int,
    base_intent: v20_macro.MacroIntent | None,
) -> bool:
    if base_intent is None or base_intent.target_owner is None:
        return False
    if base_intent.name not in {"focus_weak_enemy", "finisher", "expansion"}:
        return False
    n_players = max(2, int(getattr(fs, "n_players", 2) or 2))
    stats = _player_stats(fs, n_players)
    enemies = [p for p in range(n_players) if p != int(player) and stats[p][2] > 0.0]
    if not enemies:
        return False
    leader = max(enemies, key=lambda p: (stats[p][0] + stats[p][1] * 18.0, stats[p][1]))
    if int(base_intent.target_owner) == int(leader):
        return False
    my_empire = stats[player][0] + stats[player][1] * 18.0
    leader_empire = stats[leader][0] + stats[leader][1] * 18.0
    return leader_empire >= my_empire * _env_float("V20_SUPPORT_LEADER_AHEAD", 1.10)


def _player_stats(fs: fsim.FastState, n_players: int) -> list[tuple[float, float, float]]:
    ships = [0.0] * n_players
    prod = [0.0] * n_players
    planets_n = [0.0] * n_players
    for p in fs.planets:
        owner = int(p[OWNER])
        if 0 <= owner < n_players:
            ships[owner] += float(p[SHIPS])
            prod[owner] += float(p[PROD])
            planets_n[owner] += 1.0
    for f in fs.fleets:
        owner = int(f[F_OWNER])
        if 0 <= owner < n_players:
            ships[owner] += float(f[F_SHIPS])
    return list(zip(ships, prod, planets_n))


def _has_rear_to_front_route(
    fs: fsim.FastState,
    player: int,
    sources: list[tuple[int, float]],
    targets: list[SupportScore],
) -> bool:
    if not sources or not targets:
        return False
    planets = fs.planets
    owners = planets[:, OWNER].astype(np.int64)
    enemy_idx = np.where((owners >= 0) & (owners != int(player)))[0]
    for src_idx, _ in sources[:4]:
        src_front = _nearest_dist(planets, src_idx, enemy_idx)
        for target in targets[:4]:
            tgt_front = _nearest_dist(planets, target.idx, enemy_idx)
            if src_front >= tgt_front + 12.0:
                return True
    return False


def _source_reserve(fs: fsim.FastState, player: int, idx: int) -> float:
    planets = fs.planets
    owners = planets[:, OWNER].astype(np.int64)
    enemy_idx = np.where((owners >= 0) & (owners != int(player)))[0]
    near_enemy = _nearest_dist(planets, idx, enemy_idx)
    front_pad = max(0.0, 35.0 - near_enemy) * 0.22 if len(enemy_idx) else 0.0
    incoming = _incoming_enemy_ships(fs, player, idx, max_eta=16)
    return max(3.0, float(planets[idx, PROD]) * 2.8 + front_pad + incoming * 0.55)


def _nearest_dist(planets: np.ndarray, idx: int, candidates: Iterable[int]) -> float:
    cand = list(candidates)
    if not cand:
        return 120.0
    p = planets[int(idx)]
    pts = planets[np.array(cand, dtype=np.int64)]
    d = np.hypot(pts[:, X] - p[X], pts[:, Y] - p[Y])
    return float(d.min()) if len(d) else 120.0


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.hypot(float(a[X] - b[X]), float(a[Y] - b[Y])))


def _normalise(pairs: list[v20_macro.MacroPair]) -> list[v20_macro.MacroPair]:
    if not pairs:
        return []
    vals = np.array([p.bias for p in pairs], dtype=np.float64)
    lo = float(vals.min())
    hi = float(vals.max())
    if hi - lo < 1e-9:
        return [v20_macro.MacroPair(p.src_idx, p.tgt_idx, 0.58, p.role) for p in pairs]
    return [v20_macro.MacroPair(p.src_idx, p.tgt_idx, 0.25 + 0.75 * (p.bias - lo) / (hi - lo), p.role) for p in pairs]


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)
