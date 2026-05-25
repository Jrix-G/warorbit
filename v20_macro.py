"""V20 macro layer: conservative hierarchical intent selection.

This module intentionally stays small and deterministic.  It looks at the
current FastState, picks one macro intent, then exposes source/target pairs
that a tactical layer can convert into V15-compatible launch candidates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

import v15_fast_sim as fsim

ID, OWNER, X, Y, R, SHIPS, PROD = fsim.ID, fsim.OWNER, fsim.X, fsim.Y, fsim.R, fsim.SHIPS, fsim.PROD
F_OWNER, F_SHIPS = fsim.F_OWNER, fsim.F_SHIPS


@dataclass(frozen=True)
class MacroIntent:
    name: str
    confidence: float
    target_owner: int | None = None
    target_ids: tuple[int, ...] = ()
    source_ids: tuple[int, ...] = ()
    pressure: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class MacroPair:
    src_idx: int
    tgt_idx: int
    bias: float
    role: str


def choose_intent(
    fs: fsim.FastState,
    player: int,
    *,
    enable_expansion: bool = True,
    enable_focus_weak_enemy: bool = True,
    enable_consolidation: bool = True,
    enable_front_reinforcement: bool = True,
    enable_finisher: bool = True,
) -> MacroIntent:
    """Pick a single high-level intent for the current turn.

    The ordering is deliberately conservative: immediate finishing windows and
    endangered fronts can pre-empt expansion, but otherwise neutrals remain the
    default growth plan until the board has mostly converted.
    """
    planets = fs.planets
    if len(planets) == 0:
        return MacroIntent("consolidation", 0.0, reason="empty_board")

    n_players = max(int(getattr(fs, "n_players", 2) or 2), _infer_n_players(fs))
    player = int(player)
    stats = _player_stats(fs, n_players)
    my = stats[player]
    remaining = max(0, int(fs.episode_steps) - int(fs.step))

    owners = planets[:, OWNER].astype(np.int64)
    my_idx = np.where(owners == player)[0]
    neutral_idx = np.where(owners == -1)[0]
    enemy_idx = np.where((owners >= 0) & (owners != player))[0]
    enemies = [p for p in range(n_players) if p != player and stats[p]["planets"] > 0]

    if len(my_idx) == 0:
        return MacroIntent("consolidation", 0.0, reason="no_owned_planets")

    weak_enemy = _weakest_enemy(stats, player, enemies)
    strong_enemy = _strongest_enemy(stats, player, enemies)
    weak_score = stats[weak_enemy]["empire"] if weak_enemy is not None else 0.0
    max_enemy = max((stats[p]["empire"] for p in enemies), default=0.0)
    my_score = my["empire"]
    my_planets = int(my["planets"])
    best_enemy_planets = max((int(stats[p]["planets"]) for p in enemies), default=0)
    my_prod = float(my["prod"])
    max_enemy_prod = max((float(stats[p]["prod"]) for p in enemies), default=0.0)

    front_pressure, front_ids = _front_pressure(fs, player)

    if enable_finisher and weak_enemy is not None:
        step = int(fs.step)
        weak_planets = int(stats[weak_enemy]["planets"])
        weak_ships = float(stats[weak_enemy]["ships"])
        dominant = my_score >= weak_score * 1.35 and float(my["ships"]) >= weak_ships + 16.0
        late_window = remaining <= 45 and weak_score <= my_score * 0.75
        crushed = weak_score <= my_score * 0.42 and (step >= 35 or my_planets >= 3)
        small_enemy = weak_planets <= 2 and (step >= 70 or dominant or late_window)
        if n_players >= 4:
            finish_window = late_window or (weak_planets <= 1 and (dominant or int(fs.step) >= 90))
        else:
            finish_window = crushed or small_enemy or late_window
        if finish_window and len(enemy_idx) > 0 and my_score > 0:
            target_ids = _enemy_target_ids(fs, weak_enemy, limit=6)
            return MacroIntent(
                "finisher",
                _clip01(0.58 + (my_score - weak_score) / max(my_score + weak_score, 1.0) * 0.35),
                target_owner=weak_enemy,
                target_ids=target_ids,
                source_ids=_source_ids(fs, player, limit=8),
                pressure=weak_score,
                reason="weak_enemy_finish_window",
            )

    if enable_front_reinforcement and front_pressure > 0.58 and my_planets >= 2:
        return MacroIntent(
            "front_reinforcement",
            _clip01(front_pressure),
            target_owner=player,
            target_ids=front_ids,
            source_ids=_source_ids(fs, player, limit=8),
            pressure=front_pressure,
            reason="front_low_garrison_or_enemy_nearby",
        )

    if (
        enable_focus_weak_enemy
        and n_players >= 4
        and strong_enemy is not None
        and len(enemy_idx) > 0
        and not _env_flag(_GLOBAL_ENV_GET("V20_DISABLE_PRESSURE_LEADER", ""))
        and (n_players < 4 or _env_flag(_GLOBAL_ENV_GET("V20_ENABLE_4P_PRESSURE_LEADER", "")))
    ):
        step = int(fs.step)
        leader_score = float(stats[strong_enemy]["empire"])
        leader_prod = float(stats[strong_enemy]["prod"])
        total_prod = sum(float(row["prod"]) for row in stats) or 1.0
        my_prod_share = float(my_prod) / total_prod
        reach = _owner_reachability(fs, player, strong_enemy)
        leader_ahead = leader_score >= my_score * 1.05 or leader_prod >= my_prod * 1.08
        board_is_open = len(neutral_idx) > 0 and my_planets < 4 and step < 38
        min_step = int(_env_float_value("V20_PRESSURE_MIN_STEP", 45.0))
        max_reach = _env_float_value("V20_PRESSURE_MAX_REACH", 55.0)
        min_prod_share = _env_float_value("V20_PRESSURE_MIN_MY_PROD_SHARE", 0.18)
        min_score_vs_weak = _env_float_value("V20_PRESSURE_MIN_SCORE_VS_WEAK", 0.82)
        not_last = my_score >= weak_score * min_score_vs_weak
        if (
            step >= min_step
            and leader_ahead
            and not board_is_open
            and reach <= max_reach
            and my_prod_share >= min_prod_share
            and not_last
        ):
            return MacroIntent(
                "pressure_leader",
                _clip01(0.48 + (leader_score - my_score) / max(leader_score + my_score, 1.0) * 0.42),
                target_owner=strong_enemy,
                target_ids=_enemy_target_ids(fs, strong_enemy, limit=8),
                source_ids=_source_ids(fs, player, limit=8),
                pressure=leader_score,
                reason="4p_pressure_current_leader",
            )

    if enable_consolidation and n_players >= 4 and my_planets >= 5 and int(fs.step) >= 55:
        # Top10 4p winners pivot toward support/consolidation after the
        # opening. Keep this before generic expansion so we do not chase
        # low-value neutrals while the board is already contested.
        return MacroIntent(
            "consolidation",
            _clip01(0.50 + 0.20 * min(1.0, my_planets / 10.0)),
            target_owner=player,
            target_ids=_frontier_own_ids(fs, player, limit=8),
            source_ids=_source_ids(fs, player, limit=8),
            pressure=front_pressure,
            reason="4p_midgame_support_pivot",
        )

    if enable_expansion and len(neutral_idx) > 0:
        early = int(fs.step) < (70 if n_players <= 2 else 42)
        behind_growth = my_prod <= max_enemy_prod * 1.08 or my_planets <= best_enemy_planets
        sparse_owned = my_planets < max(5, int(len(planets) * (0.25 if n_players <= 2 else 0.18)))
        my_frac = my_planets / max(1.0, float(len(planets)))
        four_p_growth_cap = n_players >= 4 and int(fs.step) >= 70 and (my_planets >= 8 or my_frac >= 0.25)
        if (early or behind_growth or sparse_owned) and not four_p_growth_cap:
            target_ids = _neutral_target_ids(fs, player, limit=8)
            conf = 0.46 + (0.20 if early else 0.0) + (0.16 if behind_growth else 0.0)
            return MacroIntent(
                "expansion",
                _clip01(conf),
                target_owner=-1,
                target_ids=target_ids,
                source_ids=_source_ids(fs, player, limit=8),
                pressure=max_enemy_prod - my_prod,
                reason="neutral_growth_available",
            )

    if enable_focus_weak_enemy and n_players < 4 and weak_enemy is not None and len(enemy_idx) > 0:
        weak_is_exposed = weak_score <= max_enemy * 0.82 if max_enemy > 0 else False
        two_player_pressure = n_players <= 2 and (len(neutral_idx) == 0 or int(fs.step) >= 80)
        if two_player_pressure:
            return MacroIntent(
                "focus_weak_enemy",
                _clip01(0.50 + (max_enemy - weak_score) / max(max_enemy, 1.0) * 0.35),
                target_owner=weak_enemy,
                target_ids=_enemy_target_ids(fs, weak_enemy, limit=8),
                source_ids=_source_ids(fs, player, limit=8),
                pressure=weak_score,
                reason="weak_enemy_focus",
            )

    if enable_consolidation:
        return MacroIntent(
            "consolidation",
            0.42,
            target_owner=player,
            target_ids=_frontier_own_ids(fs, player, limit=6),
            source_ids=_source_ids(fs, player, limit=8),
            pressure=front_pressure,
            reason="default_stage_and_hold",
        )

    return MacroIntent("expansion", 0.0, target_owner=-1, reason="all_macro_modules_disabled")


def candidate_pairs(
    fs: fsim.FastState,
    player: int,
    intent: MacroIntent,
    *,
    max_pairs: int = 24,
) -> list[MacroPair]:
    """Return source/target index pairs ordered by macro preference."""
    planets = fs.planets
    if len(planets) == 0 or max_pairs <= 0:
        return []
    owners = planets[:, OWNER].astype(np.int64)
    my_idx = [int(i) for i in np.where(owners == int(player))[0]]
    if not my_idx:
        return []

    source_rank = _rank_sources(fs, player, intent)
    target_rank = _rank_targets(fs, player, intent)
    if not source_rank or not target_rank:
        return []

    pairs: list[MacroPair] = []
    seen: set[tuple[int, int]] = set()
    for tgt_idx, tgt_score in target_rank:
        for src_idx, src_score in source_rank:
            if src_idx == tgt_idx:
                continue
            if intent.name in ("consolidation", "front_reinforcement") and owners[tgt_idx] != player:
                continue
            if intent.name not in ("consolidation", "front_reinforcement") and owners[tgt_idx] == player:
                continue
            key = (src_idx, tgt_idx)
            if key in seen:
                continue
            seen.add(key)
            dist = _dist_rows(planets[src_idx], planets[tgt_idx])
            bias = float(tgt_score + src_score - 0.012 * dist)
            pairs.append(MacroPair(src_idx=src_idx, tgt_idx=tgt_idx, bias=bias, role=intent.name))
            if len(pairs) >= max_pairs:
                return _normalise_pair_bias(pairs)
    return _normalise_pair_bias(pairs)


def intent_env_kwargs(env_getter) -> dict:
    """Map V20_DISABLE_* env vars to choose_intent keyword flags."""
    return {
        "enable_expansion": not _env_flag(env_getter("V20_DISABLE_EXPANSION", "")),
        "enable_focus_weak_enemy": not _env_flag(env_getter("V20_DISABLE_FOCUS_WEAK_ENEMY", "")),
        "enable_consolidation": not _env_flag(env_getter("V20_DISABLE_CONSOLIDATION", "")),
        "enable_front_reinforcement": not _env_flag(env_getter("V20_DISABLE_FRONT_REINFORCEMENT", "")),
        "enable_finisher": not _env_flag(env_getter("V20_DISABLE_FINISHER", "")),
    }


def _rank_sources(fs: fsim.FastState, player: int, intent: MacroIntent) -> list[tuple[int, float]]:
    planets = fs.planets
    owners = planets[:, OWNER].astype(np.int64)
    idxs = [int(i) for i in np.where(owners == int(player))[0]]
    front_targets = _foreign_indices(fs, player)
    ranked: list[tuple[int, float]] = []
    for i in idxs:
        p = planets[i]
        reserve = _source_reserve(fs, i, player)
        surplus = max(0.0, float(p[SHIPS]) - reserve)
        if surplus < 2.0:
            continue
        front_dist = _nearest_dist(planets, i, front_targets)
        rear_bonus = 0.0
        if intent.name in ("consolidation", "front_reinforcement"):
            rear_bonus = min(0.85, front_dist / 90.0)
        score = math.log1p(surplus) * 0.38 + float(p[PROD]) * 0.06 + rear_bonus
        ranked.append((i, score))
    ranked.sort(key=lambda kv: kv[1], reverse=True)
    return ranked


def _rank_targets(fs: fsim.FastState, player: int, intent: MacroIntent) -> list[tuple[int, float]]:
    planets = fs.planets
    owners = planets[:, OWNER].astype(np.int64)
    ids = planets[:, ID].astype(np.int64)
    preferred_ids = set(int(x) for x in intent.target_ids)
    rows: Iterable[int]

    if intent.name == "expansion":
        rows = np.where(owners == -1)[0]
    elif intent.name in ("focus_weak_enemy", "finisher", "pressure_leader"):
        if intent.target_owner is None:
            rows = np.where((owners >= 0) & (owners != int(player)))[0]
        else:
            rows = np.where(owners == int(intent.target_owner))[0]
    elif intent.name == "front_reinforcement":
        rows = np.where(owners == int(player))[0]
    else:
        rows = np.where(owners == int(player))[0]

    ranked: list[tuple[int, float]] = []
    my_idx = np.where(owners == int(player))[0]
    enemy_idx = np.where((owners >= 0) & (owners != int(player)))[0]
    for j in rows:
        j = int(j)
        p = planets[j]
        preferred = 0.25 if int(ids[j]) in preferred_ids else 0.0
        near_my = _nearest_dist(planets, j, my_idx)
        near_enemy = _nearest_dist(planets, j, enemy_idx)
        prod = float(p[PROD])
        ships = float(p[SHIPS])
        if intent.name == "expansion":
            score = 0.42 * prod - 0.045 * ships - 0.010 * near_my + preferred
            if _is_static(p):
                score += 0.12
        elif intent.name == "finisher":
            score = 0.22 * prod - 0.030 * ships - 0.009 * near_my + 0.28 + preferred
        elif intent.name in ("focus_weak_enemy", "pressure_leader"):
            score = 0.30 * prod - 0.026 * ships - 0.008 * near_my + preferred
            if intent.name == "pressure_leader":
                score += 0.08 * max(0.0, 55.0 - near_my) / 55.0
        elif intent.name == "front_reinforcement":
            low_garrison = max(0.0, 16.0 + prod * 4.0 - ships) / 35.0
            score = 0.40 * low_garrison - 0.011 * near_enemy + 0.005 * near_my + preferred
        else:
            frontier = 120.0 if len(enemy_idx) == 0 else near_enemy
            score = 0.25 * prod - 0.018 * frontier + 0.006 * near_my + preferred
        ranked.append((j, float(score)))
    ranked.sort(key=lambda kv: kv[1], reverse=True)
    return ranked


def _normalise_pair_bias(pairs: list[MacroPair]) -> list[MacroPair]:
    if not pairs:
        return []
    vals = np.array([p.bias for p in pairs], dtype=np.float64)
    lo = float(vals.min())
    hi = float(vals.max())
    if hi - lo < 1e-9:
        return [MacroPair(p.src_idx, p.tgt_idx, 0.5, p.role) for p in pairs]
    return [MacroPair(p.src_idx, p.tgt_idx, (p.bias - lo) / (hi - lo), p.role) for p in pairs]


def _player_stats(fs: fsim.FastState, n_players: int) -> list[dict[str, float]]:
    garrison = np.zeros(n_players, dtype=np.float64)
    fleets = np.zeros(n_players, dtype=np.float64)
    prod = np.zeros(n_players, dtype=np.float64)
    planets = np.zeros(n_players, dtype=np.float64)
    if len(fs.planets):
        owners = fs.planets[:, OWNER].astype(np.int64)
        for p in range(n_players):
            mask = owners == p
            if np.any(mask):
                garrison[p] = fs.planets[mask, SHIPS].sum()
                prod[p] = fs.planets[mask, PROD].sum()
                planets[p] = float(mask.sum())
    if len(fs.fleets):
        owners = fs.fleets[:, F_OWNER].astype(np.int64)
        for p in range(n_players):
            mask = owners == p
            if np.any(mask):
                fleets[p] = fs.fleets[mask, F_SHIPS].sum()
    rows = []
    for p in range(n_players):
        ships = float(garrison[p] + fleets[p])
        rows.append({
            "ships": ships,
            "prod": float(prod[p]),
            "planets": float(planets[p]),
            "empire": ships + float(prod[p]) * 18.0 + float(planets[p]) * 6.0,
        })
    return rows


def _front_pressure(fs: fsim.FastState, player: int) -> tuple[float, tuple[int, ...]]:
    planets = fs.planets
    owners = planets[:, OWNER].astype(np.int64) if len(planets) else np.zeros(0, dtype=np.int64)
    my_idx = np.where(owners == int(player))[0]
    enemy_idx = np.where((owners >= 0) & (owners != int(player)))[0]
    if len(my_idx) == 0 or len(enemy_idx) == 0:
        return 0.0, ()
    scored: list[tuple[float, int]] = []
    for i in my_idx:
        p = planets[int(i)]
        d = _nearest_dist(planets, int(i), enemy_idx)
        low = max(0.0, 14.0 + float(p[PROD]) * 3.0 - float(p[SHIPS])) / 35.0
        near = max(0.0, 38.0 - d) / 38.0
        pressure = _clip01(0.62 * low + 0.38 * near)
        if pressure > 0.20:
            scored.append((pressure, int(p[ID])))
    scored.sort(reverse=True)
    if not scored:
        return 0.0, ()
    return float(scored[0][0]), tuple(pid for _, pid in scored[:5])


def _source_reserve(fs: fsim.FastState, idx: int, player: int) -> float:
    p = fs.planets[idx]
    enemy_idx = _enemy_indices(fs, player)
    near_enemy = _nearest_dist(fs.planets, idx, enemy_idx)
    front_pad = max(0.0, 26.0 - near_enemy) * 0.16 if len(enemy_idx) else 0.0
    return max(2.0, float(p[PROD]) * 2.6 + front_pad)


def _neutral_target_ids(fs: fsim.FastState, player: int, *, limit: int) -> tuple[int, ...]:
    ranked = _rank_targets(fs, player, MacroIntent("expansion", 1.0, target_owner=-1))
    return tuple(int(fs.planets[i, ID]) for i, _ in ranked[:limit])


def _enemy_target_ids(fs: fsim.FastState, owner: int, *, limit: int) -> tuple[int, ...]:
    planets = fs.planets
    owners = planets[:, OWNER].astype(np.int64)
    rows = [int(i) for i in np.where(owners == int(owner))[0]]
    rows.sort(key=lambda i: (float(planets[i, SHIPS]) - float(planets[i, PROD]) * 2.0, -float(planets[i, PROD])))
    return tuple(int(planets[i, ID]) for i in rows[:limit])


def _frontier_own_ids(fs: fsim.FastState, player: int, *, limit: int) -> tuple[int, ...]:
    planets = fs.planets
    owners = planets[:, OWNER].astype(np.int64) if len(planets) else np.zeros(0, dtype=np.int64)
    my_idx = [int(i) for i in np.where(owners == int(player))[0]]
    foreign = _foreign_indices(fs, player)
    my_idx.sort(key=lambda i: _nearest_dist(planets, i, foreign))
    return tuple(int(planets[i, ID]) for i in my_idx[:limit])


def _source_ids(fs: fsim.FastState, player: int, *, limit: int) -> tuple[int, ...]:
    ranked = _rank_sources(fs, player, MacroIntent("source_rank", 1.0))
    return tuple(int(fs.planets[i, ID]) for i, _ in ranked[:limit])


def _weakest_enemy(stats: list[dict[str, float]], player: int, enemies: list[int]) -> int | None:
    if not enemies:
        return None
    return min(enemies, key=lambda p: (stats[p]["empire"], stats[p]["ships"], p))


def _strongest_enemy(stats: list[dict[str, float]], player: int, enemies: list[int]) -> int | None:
    if not enemies:
        return None
    return max(enemies, key=lambda p: (stats[p]["empire"], stats[p]["prod"], -p))


def _infer_n_players(fs: fsim.FastState) -> int:
    max_owner = 1
    if len(fs.planets):
        owners = fs.planets[:, OWNER]
        if len(owners):
            max_owner = max(max_owner, int(np.max(owners)))
    if len(fs.fleets):
        owners = fs.fleets[:, F_OWNER]
        if len(owners):
            max_owner = max(max_owner, int(np.max(owners)))
    return max(2, max_owner + 1)


def _foreign_indices(fs: fsim.FastState, player: int) -> np.ndarray:
    if len(fs.planets) == 0:
        return np.zeros(0, dtype=np.int64)
    owners = fs.planets[:, OWNER].astype(np.int64)
    return np.where(owners != int(player))[0]


def _enemy_indices(fs: fsim.FastState, player: int) -> np.ndarray:
    if len(fs.planets) == 0:
        return np.zeros(0, dtype=np.int64)
    owners = fs.planets[:, OWNER].astype(np.int64)
    return np.where((owners >= 0) & (owners != int(player)))[0]


def _nearest_dist(planets: np.ndarray, idx: int, candidates: Iterable[int]) -> float:
    cand = list(candidates)
    if not cand:
        return 120.0
    p = planets[int(idx)]
    pts = planets[np.array(cand, dtype=np.int64)]
    d = np.hypot(pts[:, X] - p[X], pts[:, Y] - p[Y])
    return float(d.min()) if len(d) else 120.0


def _dist_rows(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.hypot(float(a[X] - b[X]), float(a[Y] - b[Y])))


def _is_static(row: np.ndarray) -> bool:
    return math.hypot(float(row[X] - 50.0), float(row[Y] - 50.0)) + float(row[R]) >= 50.0


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _env_flag(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_float_value(name: str, default: float) -> float:
    try:
        return float(_GLOBAL_ENV_GET(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _GLOBAL_ENV_GET(name: str, default: str = "") -> str:
    try:
        import os

        return os.environ.get(name, default)
    except Exception:
        return default


def _owner_reachability(fs: fsim.FastState, player: int, owner: int) -> float:
    planets = fs.planets
    if len(planets) == 0:
        return 120.0
    owners = planets[:, OWNER].astype(np.int64)
    mine = np.where(owners == int(player))[0]
    theirs = np.where(owners == int(owner))[0]
    if len(mine) == 0 or len(theirs) == 0:
        return 120.0
    best = 120.0
    for i in mine:
        d = _nearest_dist(planets, int(i), theirs)
        if d < best:
            best = d
    return float(best)
