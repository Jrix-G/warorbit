"""V20 search: V15-compatible combo search with a macro prior.

The macro layer does not execute moves directly.  It chooses an intent, turns
that intent into V7-aimed atomic shots, and lets the V15 deterministic combo
evaluator decide whether any combination is actually worth playing.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import numpy as np

import bot_v7
import bot_v15
import v15_eval
import v15_fast_sim as fsim
import v15_search
import v20_macro
import v20_rank_value
import v20_support

ID, OWNER, X, Y, SHIPS, PROD = fsim.ID, fsim.OWNER, fsim.X, fsim.Y, fsim.SHIPS, fsim.PROD


@dataclass(frozen=True)
class V20SearchConfig:
    time_budget: float = 0.55
    horizon: int = 24
    top_k: int = 7
    max_combo: int = 4
    macro_pair_limit: int = 24
    macro_shot_limit: int = 14
    bias_weight: float = 0.045
    staging_bias_weight: float = 0.055
    max_esc_loss: float = 0.020
    min_objective_gain: float = 0.0
    rank_value_weight: float = 0.04
    bc_cont: bool = False
    use_v15_atomic: bool = True
    use_macro_candidates: bool = False
    use_top10_policy: bool = False
    use_macro_bias: bool = True


def from_env(*, time_budget: float | None = None, horizon: int | None = None) -> V20SearchConfig:
    cfg = V20SearchConfig()
    return V20SearchConfig(
        time_budget=float(time_budget if time_budget is not None else _env_float("V20_TIME_BUDGET", cfg.time_budget)),
        horizon=int(horizon if horizon is not None else _env_int("V20_HORIZON", cfg.horizon)),
        top_k=_env_int("V20_TOP_K", cfg.top_k),
        max_combo=_env_int("V20_MAX_COMBO", cfg.max_combo),
        macro_pair_limit=_env_int("V20_MACRO_PAIR_LIMIT", cfg.macro_pair_limit),
        macro_shot_limit=_env_int("V20_MACRO_SHOT_LIMIT", cfg.macro_shot_limit),
        bias_weight=_env_float("V20_BIAS_WEIGHT", cfg.bias_weight),
        staging_bias_weight=_env_float("V20_STAGING_BIAS_WEIGHT", cfg.staging_bias_weight),
        max_esc_loss=_env_float("V20_MAX_ESC_LOSS", cfg.max_esc_loss),
        min_objective_gain=_env_float("V20_MIN_OBJECTIVE_GAIN", cfg.min_objective_gain),
        rank_value_weight=_rank_value_weight_from_env(cfg.rank_value_weight),
        bc_cont=_env_flag("V20_BC_CONT"),
        use_v15_atomic=not _env_flag("V20_DISABLE_V15_ATOMIC"),
        use_macro_candidates=_env_flag("V20_ENABLE_MACRO_CANDIDATES") and not _env_flag("V20_DISABLE_MACRO_CANDIDATES"),
        use_top10_policy=_env_flag("V20_ENABLE_TOP10_POLICY"),
        use_macro_bias=not _env_flag("V20_DISABLE_BIAS"),
    )


def search(
    obs,
    config=None,
    *,
    time_budget: float | None = None,
    horizon: int | None = None,
    macro_intent: v20_macro.MacroIntent | None = None,
) -> list:
    """Return a V15-format action list: [[from_id, angle, ships], ...]."""
    if _env_flag("V20_DISABLE") or _env_flag("V20_DISABLE_SEARCH"):
        return fallback_v15(obs, config)

    cfg = from_env(time_budget=time_budget, horizon=horizon)
    if cfg.max_combo <= 0 or cfg.top_k <= 0:
        return fallback_v15(obs, config)

    deadline = time.monotonic() + max(0.03, cfg.time_budget)
    try:
        player = _player_from_obs(obs)
        fs = _state_from_obs(obs, config)
        eval_weights = _load_eval_weights()

        try:
            v7_move = bot_v7.agent(obs, config)
        except Exception:
            v7_move = []
        if not isinstance(v7_move, list):
            v7_move = []

        if macro_intent is None and not _env_flag("V20_DISABLE_MACRO"):
            macro_intent = v20_macro.choose_intent(
                fs, player, **v20_macro.intent_env_kwargs(os.environ.get)
            )

        shots, shot_bias = _candidate_shots(fs, obs, player, v7_move, macro_intent, cfg)
        if not shots:
            return v7_move if v7_move else fallback_v15(obs, config)

        baseline = _eval_combo_v20(fs, player, [], cfg.horizon, cfg.bc_cont, eval_weights, cfg)
        h1 = max(4, cfg.horizon // 2)

        scored: list[tuple[list, float, float]] = []
        for shot in shots:
            if _env_flag("V20_STRICT_TIME") and time.monotonic() > deadline:
                break
            esc = _eval_combo_v20(fs, player, [shot], h1, cfg.bc_cont, eval_weights, cfg)
            obj = esc + _shot_bonus(shot, shot_bias, macro_intent, cfg)
            scored.append((shot, esc, obj))
        if not scored:
            return v7_move if v7_move else fallback_v15(obs, config)
        scored.sort(key=lambda row: row[2], reverse=True)
        top = [shot for shot, _, _ in scored[:cfg.top_k]]

        best_combo: list = []
        best_esc = baseline
        best_obj = baseline
        max_combo = min(cfg.max_combo, len(top))
        for r in range(1, max_combo + 1):
            for combo_tuple in combinations(top, r):
                if _env_flag("V20_STRICT_TIME") and time.monotonic() > deadline:
                    break
                combo = list(combo_tuple)
                if not v15_search._valid_combo(combo):
                    continue
                esc = _eval_combo_v20(fs, player, combo, cfg.horizon, cfg.bc_cont, eval_weights, cfg)
                obj = esc + _combo_bonus(combo, shot_bias, macro_intent, cfg)
                allowed_loss = cfg.max_esc_loss + _macro_loss_allowance(combo, shot_bias, macro_intent, cfg)
                if esc + allowed_loss < baseline:
                    continue
                if obj > best_obj + cfg.min_objective_gain:
                    best_combo, best_esc, best_obj = combo, esc, obj

        if best_combo:
            return best_combo

        if _env_flag("V20_EMPTY_RETURNS_V15"):
            return fallback_v15(obs, config)
        return []
    except Exception:
        return fallback_v15(obs, config)


_EVAL_WEIGHTS_CACHE: dict[str, v15_eval.EvalWeights] = {}


def _load_eval_weights() -> v15_eval.EvalWeights:
    path = os.environ.get("V20_EVAL_WEIGHTS", "").strip()
    if not path:
        return v15_eval.ESC
    if path not in _EVAL_WEIGHTS_CACHE:
        try:
            _EVAL_WEIGHTS_CACHE[path] = v15_eval.EvalWeights.load(path)
        except Exception:
            _EVAL_WEIGHTS_CACHE[path] = v15_eval.ESC
    return _EVAL_WEIGHTS_CACHE[path]


def _eval_combo_v20(
    fs: fsim.FastState,
    player: int,
    combo: list,
    horizon: int,
    bc_cont: bool,
    weights: v15_eval.EvalWeights,
    cfg: V20SearchConfig,
) -> float:
    if cfg.rank_value_weight <= 0.0:
        return v15_search._eval_combo(fs, player, combo, horizon, bc_cont, weights)
    return v20_rank_value.eval_combo(
        fs,
        player,
        combo,
        horizon,
        bc_cont,
        weights,
        cfg.rank_value_weight,
    )


def policy_fn_from_intent(intent: v20_macro.MacroIntent | None) -> Callable | None:
    """Return a V15 `policy_fn(fs, player) -> [(src_idx, tgt_idx), ...]`.

    This makes the macro layer usable by vanilla v15_search.search, even when
    the V20 custom scorer is disabled by env vars.
    """
    if intent is None:
        return None

    def _policy(fs, player: int):
        if v20_support.is_support_intent(intent):
            pairs = v20_support.candidate_pairs(fs, player, intent, max_pairs=_env_int("V20_MACRO_PAIR_LIMIT", 24))
        else:
            pairs = v20_macro.candidate_pairs(fs, player, intent, max_pairs=_env_int("V20_MACRO_PAIR_LIMIT", 24))
        return [(p.src_idx, p.tgt_idx) for p in pairs]

    return _policy


def fallback_v15(obs, config=None) -> list:
    """Conservative fallback chain: bot_v15, then bot_v7, then pass."""
    try:
        if _env_flag("V20_FALLBACK_V15_SEARCH"):
            return v15_search.search(
                obs,
                config,
                time_budget=_env_float("V20_FALLBACK_TIME_BUDGET", 0.35),
                horizon=_env_int("V20_FALLBACK_HORIZON", 22),
            )
        move = bot_v15.agent(obs, config)
        return move if isinstance(move, list) else []
    except Exception:
        try:
            move = bot_v7.agent(obs, config)
            return move if isinstance(move, list) else []
        except Exception:
            return []


def _candidate_shots(
    fs: fsim.FastState,
    obs,
    player: int,
    v7_move: list,
    intent: v20_macro.MacroIntent | None,
    cfg: V20SearchConfig,
) -> tuple[list[list], dict[tuple[int, float], float]]:
    shots: list[list] = []
    shot_bias: dict[tuple[int, float], float] = {}
    seen: set[tuple[int, float]] = set()

    def add(shot, bias: float = 0.0) -> None:
        clean = _clean_shot(shot)
        if clean is None:
            return
        key = _shot_key(clean)
        if key not in seen:
            seen.add(key)
            shots.append(clean)
        if bias > 0.0:
            shot_bias[key] = max(float(bias), shot_bias.get(key, 0.0))

    if cfg.use_v15_atomic:
        try:
            for shot in v15_search._enumerate_shots(fs, player, v7_move):
                add(shot, 0.0)
        except Exception:
            for shot in v7_move or []:
                add(shot, 0.0)
    else:
        for shot in v7_move or []:
            add(shot, 0.0)

    support_overlay = None
    try:
        support_overlay = v20_support.choose_intent(fs, player, intent)
    except Exception:
        support_overlay = None
    support_intent = v20_support.is_support_intent(intent)
    has_support_overlay = v20_support.is_support_intent(support_overlay)

    if intent is None or (not cfg.use_macro_candidates and not support_intent and not has_support_overlay):
        if cfg.use_top10_policy:
            _add_top10_policy_shots(fs, world=None, obs=obs, player=player, add=add, limit=cfg.macro_shot_limit)
        return shots, shot_bias

    try:
        world = bot_v7._build_world(obs)
        if cfg.use_top10_policy:
            _add_top10_policy_shots(fs, world=world, obs=obs, player=player, add=add, limit=cfg.macro_shot_limit)
        if support_intent:
            pairs = v20_support.candidate_pairs(fs, player, intent, max_pairs=cfg.macro_pair_limit)
        elif has_support_overlay and support_overlay is not None:
            pairs = v20_support.candidate_pairs(fs, player, support_overlay, max_pairs=cfg.macro_pair_limit)
        else:
            pairs = v20_macro.candidate_pairs(fs, player, intent, max_pairs=cfg.macro_pair_limit)
        added = 0
        for pair in pairs:
            if added >= cfg.macro_shot_limit:
                break
            active_intent = support_overlay if has_support_overlay and support_overlay is not None and not support_intent else intent
            shot = _shot_from_pair(fs, world, player, active_intent, pair)
            if shot is None:
                continue
            add(shot, pair.bias)
            added += 1
    except Exception:
        pass
    return shots, shot_bias


_TOP10_POLICY_CACHE: dict | None = None


def _load_top10_policy() -> dict | None:
    global _TOP10_POLICY_CACHE
    if _TOP10_POLICY_CACHE is not None:
        return _TOP10_POLICY_CACHE
    try:
        path = os.environ.get("V20_TOP10_POLICY_PATH", "analysis/v20_top10_linear_policy.npz")
        data = np.load(path, allow_pickle=True)
        _TOP10_POLICY_CACHE = {
            "w": data["w"].astype(np.float64),
            "b": float(data["b"]),
            "mu": data["mu"].astype(np.float64),
            "sig": data["sig"].astype(np.float64),
        }
    except Exception:
        _TOP10_POLICY_CACHE = {}
    return _TOP10_POLICY_CACHE or None


def _add_top10_policy_shots(fs, world, obs, player: int, add: Callable, limit: int) -> None:
    model = _load_top10_policy()
    if not model or len(fs.planets) == 0:
        return
    if world is None:
        try:
            world = bot_v7._build_world(obs)
        except Exception:
            return
    planets = fs.planets
    owners = planets[:, OWNER].astype(np.int64)
    mine = [int(i) for i in np.where(owners == int(player))[0]]
    if not mine:
        return
    ships_by_owner, prod_by_owner = _totals_for_policy(fs)
    n_players = max(2, int(getattr(fs, "n_players", 2) or 2))
    rows: list[tuple[float, int, int]] = []
    for src_idx in mine:
        if float(planets[src_idx, SHIPS]) < 2.0:
            continue
        for tgt_idx in range(len(planets)):
            if tgt_idx == src_idx:
                continue
            feat = _policy_features(fs, src_idx, tgt_idx, player, n_players, ships_by_owner, prod_by_owner)
            x = (feat - model["mu"]) / model["sig"]
            score = float(x @ model["w"] + model["b"])
            rows.append((score, src_idx, tgt_idx))
    rows.sort(reverse=True)
    added = 0
    for _, src_idx, tgt_idx in rows[: max(1, int(limit))]:
        intent = v20_macro.MacroIntent("top10_policy", 0.65)
        pair = v20_macro.MacroPair(src_idx=src_idx, tgt_idx=tgt_idx, bias=0.65, role="top10_policy")
        shot = _shot_from_pair(fs, world, player, intent, pair)
        if shot is None:
            continue
        add(shot, 0.35)
        added += 1
        if added >= limit:
            break


def _totals_for_policy(fs) -> tuple[list[float], list[float]]:
    n_players = max(2, int(getattr(fs, "n_players", 2) or 2))
    ships = [0.0] * n_players
    prod = [0.0] * n_players
    for p in fs.planets:
        owner = int(p[OWNER])
        if 0 <= owner < n_players:
            ships[owner] += float(p[SHIPS])
            prod[owner] += float(p[PROD])
    for f in fs.fleets:
        owner = int(f[fsim.F_OWNER])
        if 0 <= owner < n_players:
            ships[owner] += float(f[fsim.F_SHIPS])
    return ships, prod


def _policy_features(fs, src_idx: int, tgt_idx: int, player: int, n_players: int, ships: list[float], prod: list[float]) -> np.ndarray:
    planets = fs.planets
    src = planets[src_idx]
    tgt = planets[tgt_idx]
    tot_s = sum(ships) or 1.0
    tot_p = sum(prod) or 1.0
    dist = math.hypot(float(src[X] - tgt[X]), float(src[Y] - tgt[Y]))
    s_ships = float(src[SHIPS])
    t_ships = float(tgt[SHIPS])
    t_owner = int(tgt[OWNER])
    return np.asarray([
        s_ships / tot_s,
        float(tgt[PROD]) / tot_p,
        dist / 141.42,
        1.0 if (t_owner >= 0 and t_owner != player) else 0.0,
        1.0 if t_owner == -1 else 0.0,
        (s_ships - t_ships) / (s_ships + t_ships + 1.0),
        min(float(fs.step) / 500.0, 1.0),
        ships[player] / tot_s,
        prod[player] / tot_p,
        t_ships / (s_ships + 1.0),
        min(float(src[PROD]) / 10.0, 1.0),
        1.0 if n_players >= 4 else 0.0,
    ], dtype=np.float64)


def _shot_from_pair(
    fs: fsim.FastState,
    world,
    player: int,
    intent: v20_macro.MacroIntent,
    pair: v20_macro.MacroPair,
) -> list | None:
    planets = fs.planets
    if pair.src_idx >= len(planets) or pair.tgt_idx >= len(planets):
        return None
    src = planets[pair.src_idx]
    tgt = planets[pair.tgt_idx]
    if int(src[OWNER]) != int(player):
        return None
    src_id = int(src[ID])
    tgt_id = int(tgt[ID])
    if src_id == tgt_id:
        return None

    budget = _source_budget(fs, world, pair.src_idx, src_id, own_target=int(tgt[OWNER]) == int(player))
    if budget <= 0:
        return None

    probe = max(1, min(budget, int(max(1.0, budget * 0.55))))
    aim = world.plan_shot(src_id, tgt_id, probe)
    if aim is None:
        return None
    angle, turns = aim

    if int(tgt[OWNER]) == int(player):
        need = int(world.reinforcement_needed_for(tgt_id, turns))
        if intent.name == "support_late_mass":
            send = max(need, int(budget * 0.78), 8)
        elif intent.name == "support_emergency":
            send = max(need, int(budget * 0.66), 6)
        elif intent.name == "support_supply_chain":
            send = max(need, int(budget * 0.58), 5)
        elif intent.name == "front_reinforcement":
            send = max(need, int(budget * 0.50), 4)
        else:
            send = max(need, int(budget * 0.62), 5)
        send = min(budget, send)
        if send <= 0:
            return None
    else:
        need = int(world.ships_needed_to_capture(tgt_id, turns))
        if need <= 0:
            need = max(1, int(float(tgt[SHIPS])) + 1)
        margin = 1
        min_frac = 0.34
        if intent.name == "focus_weak_enemy":
            margin, min_frac = 2, 0.42
        elif intent.name == "finisher":
            margin, min_frac = 4, 0.52
        send = max(need + margin, int(budget * min_frac))
        if send > budget:
            if intent.name == "finisher" and budget >= max(6, int(need * 0.70)):
                send = budget
            else:
                return None
        send = max(1, min(budget, send))

    final_aim = world.plan_shot(src_id, tgt_id, send)
    if final_aim is None:
        return None
    return [src_id, float(final_aim[0]), int(send)]


def _source_budget(fs: fsim.FastState, world, src_idx: int, src_id: int, *, own_target: bool) -> int:
    ships = int(max(0.0, float(fs.planets[src_idx, SHIPS])))
    if ships <= 1:
        return 0
    if own_target:
        reserve = max(2, int(float(fs.planets[src_idx, PROD]) * 1.8))
        return max(0, ships - reserve)
    try:
        available = int(world.available.get(src_id, ships))
    except Exception:
        available = ships
    reserve = max(2, int(float(fs.planets[src_idx, PROD]) * 2.4))
    return max(0, min(available, ships - reserve))


def _shot_bonus(
    shot: list,
    shot_bias: dict[tuple[int, float], float],
    intent: v20_macro.MacroIntent | None,
    cfg: V20SearchConfig,
) -> float:
    if not cfg.use_macro_bias or cfg.bias_weight <= 0.0:
        return 0.0
    b = shot_bias.get(_shot_key(shot), 0.0)
    if b <= 0.0:
        return 0.0
    conf = intent.confidence if intent is not None else 0.5
    weight = _macro_bias_weight(intent, cfg)
    return weight * float(b) * (0.55 + 0.45 * float(conf))


def _combo_bonus(
    combo: list[list],
    shot_bias: dict[tuple[int, float], float],
    intent: v20_macro.MacroIntent | None,
    cfg: V20SearchConfig,
) -> float:
    if not combo:
        return 0.0
    vals = [shot_bias.get(_shot_key(s), 0.0) for s in combo]
    if not vals or max(vals) <= 0.0:
        return 0.0
    mean_bias = float(sum(vals) / len(vals))
    size_factor = 1.0 + min(0.20, 0.05 * (len(combo) - 1))
    conf = intent.confidence if intent is not None else 0.5
    weight = _macro_bias_weight(intent, cfg)
    return weight * mean_bias * size_factor * (0.55 + 0.45 * float(conf))


def _macro_bias_weight(intent: v20_macro.MacroIntent | None, cfg: V20SearchConfig) -> float:
    if intent is not None and (intent.name in ("consolidation", "front_reinforcement") or v20_support.is_support_intent(intent)):
        return max(float(cfg.bias_weight), float(cfg.staging_bias_weight))
    return float(cfg.bias_weight)


def _macro_loss_allowance(
    combo: list[list],
    shot_bias: dict[tuple[int, float], float],
    intent: v20_macro.MacroIntent | None,
    cfg: V20SearchConfig,
) -> float:
    if intent is None or (intent.name not in ("consolidation", "front_reinforcement") and not v20_support.is_support_intent(intent)):
        return 0.0
    vals = [shot_bias.get(_shot_key(s), 0.0) for s in combo]
    if not vals:
        return 0.0
    mean_bias = float(sum(vals) / len(vals))
    conf = float(intent.confidence)
    return min(0.030, 0.010 + 0.022 * mean_bias * (0.50 + 0.50 * conf))


def _state_from_obs(obs, config=None) -> fsim.FastState:
    episode_steps = _episode_steps(obs, config)
    n_players = _configured_n_players(obs, config)
    fs = fsim.from_obs(obs, n_players=n_players, episode_steps=episode_steps)
    if n_players <= 0:
        n_players = _infer_n_players(fs)
    fs.n_players = n_players
    return fs


def _configured_n_players(obs, config=None) -> int:
    if _env_flag("V20_INFER_N_PLAYERS"):
        return 0
    env_override = os.environ.get("V20_N_PLAYERS")
    if env_override:
        try:
            return max(2, int(env_override))
        except ValueError:
            pass
    for obj in (config, obs):
        for key in ("nPlayers", "n_players", "numPlayers", "num_players"):
            try:
                value = obj.get(key, None) if isinstance(obj, dict) else getattr(obj, key, None)
            except Exception:
                value = None
            if value:
                try:
                    return max(2, int(value))
                except (TypeError, ValueError):
                    pass
    return 0


def _episode_steps(obs, config=None) -> int:
    env_override = os.environ.get("V20_EPISODE_STEPS")
    if env_override:
        try:
            return int(env_override)
        except ValueError:
            pass
    for obj in (config, obs):
        for key in ("episode_steps", "episodeSteps", "episodeStepsTotal"):
            try:
                value = obj.get(key, None) if isinstance(obj, dict) else getattr(obj, key, None)
            except Exception:
                value = None
            if value:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
    return 500


def _infer_n_players(fs: fsim.FastState) -> int:
    max_owner = 1
    if len(fs.planets):
        owners = fs.planets[:, OWNER]
        owners = owners[owners >= 0]
        if len(owners):
            max_owner = max(max_owner, int(np.max(owners)))
    if len(fs.fleets):
        owners = fs.fleets[:, fsim.F_OWNER]
        owners = owners[owners >= 0]
        if len(owners):
            max_owner = max(max_owner, int(np.max(owners)))
    return max(2, max_owner + 1)


def _player_from_obs(obs) -> int:
    try:
        return int(obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0))
    except Exception:
        return 0


def _clean_shot(shot) -> list | None:
    if not isinstance(shot, (list, tuple)) or len(shot) != 3:
        return None
    try:
        src = int(shot[0])
        angle = float(shot[1])
        ships = int(shot[2])
    except (TypeError, ValueError):
        return None
    if ships <= 0 or not math.isfinite(angle):
        return None
    return [src, angle, ships]


def _shot_key(shot: list) -> tuple[int, float]:
    return int(shot[0]), round(float(shot[1]), 2)


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


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


def _rank_value_weight_from_env(default: float) -> float:
    if _env_flag("V20_DISABLE_RANK_VALUE"):
        return 0.0
    if "V20_RANK_VALUE_WEIGHT" in os.environ:
        return _env_float("V20_RANK_VALUE_WEIGHT", default)
    if _env_flag("V20_ENABLE_RANK_VALUE"):
        return 0.020
    return float(default)
