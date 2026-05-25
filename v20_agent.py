"""V20 agent entry point.

Prototype V20-MACRO: choose one conservative macro intent, inject it into a
V15-style tactical combo search, and fall back to V15/V7 on any issue.
"""

from __future__ import annotations

import os

import v15_eval
import v15_search
import v20_macro
import v20_rank_value
import v20_search
import v20_support


LAST_INTENT: v20_macro.MacroIntent | None = None
PROFILE_BY_PLAYER: dict[int, str] = {}


def agent(
    obs, config=None, *, time_budget: float | None = None, horizon: int | None = None
):
    """Kaggle-compatible agent(obs, config) entry point."""
    global LAST_INTENT
    if _env_flag("V20_DISABLE") or _env_flag("V20_DISABLE_AGENT"):
        LAST_INTENT = None
        return v20_search.fallback_v15(obs, config)

    try:
        intent = None
        if not _env_flag("V20_DISABLE_MACRO"):
            fs = v20_search._state_from_obs(obs, config)
            player = v20_search._player_from_obs(obs)
            intent = v20_macro.choose_intent(
                fs, player, **v20_macro.intent_env_kwargs(os.environ.get)
            )
        LAST_INTENT = intent

        if _env_flag("V20_DISABLE_SEARCH"):
            return v20_search.fallback_v15(obs, config)

        if _env_flag("V20_ENABLE_OPENING_PROFILE"):
            return _profile_agent(obs, config)

        if _env_flag("V20_ENABLE_PORTFOLIO"):
            return _portfolio_agent(obs, config)

        if _env_flag("V20_V15_SEARCH_ONLY"):
            policy_fn = v20_search.policy_fn_from_intent(intent)
            return v15_search.search(
                obs,
                config,
                time_budget=float(os.environ.get("V20_TIME_BUDGET", "0.55")),
                horizon=int(os.environ.get("V20_HORIZON", "24")),
                policy_fn=policy_fn,
            )

        if time_budget is not None:
            time_budget = max(
                float(time_budget),
                float(os.environ.get("V20_MIN_TIME_BUDGET", "0.20")),
            )
        seat_overrides = _seat_profile_overrides(obs, config)
        if seat_overrides:
            return _run_with_env(
                seat_overrides,
                lambda: v20_search.search(
                    obs,
                    config,
                    time_budget=time_budget,
                    horizon=horizon,
                    macro_intent=intent,
                ),
            )
        return v20_search.search(
            obs,
            config,
            time_budget=time_budget,
            horizon=horizon,
            macro_intent=intent,
        )
    except Exception:
        LAST_INTENT = None
        return v20_search.fallback_v15(obs, config)


def last_intent() -> v20_macro.MacroIntent | None:
    return LAST_INTENT


def _seat_profile_overrides(obs, config=None) -> dict[str, str]:
    if _env_flag("V20_DISABLE_SEAT_PROFILE"):
        return {}
    try:
        fs = v20_search._state_from_obs(obs, config)
        if int(getattr(fs, "n_players", 2) or 2) < 4:
            return {}
        player = v20_search._player_from_obs(obs)
        if _env_flag("V20_DISABLE_4P_PRESSURE_PROFILE"):
            return {}
        if player == 1:
            return {
                "V20_PRESSURE_MIN_STEP": "60",
                "V20_PRESSURE_MAX_REACH": "48",
            }
        if _env_flag("V20_DISABLE_PRESSURE_PROFILE"):
            return {}
        return {
            "V20_PRESSURE_MIN_SCORE_VS_WEAK": "1.08",
        }
    except Exception:
        return {}
    return {}


def _portfolio_agent(obs, config=None) -> list:
    fs = v20_search._state_from_obs(obs, config)
    player = v20_search._player_from_obs(obs)
    per_budget = float(os.environ.get("V20_PORTFOLIO_BUDGET", "0.026"))
    horizon = int(os.environ.get("V20_PORTFOLIO_HORIZON", "34"))
    rank_weight = float(
        os.environ.get(
            "V20_PORTFOLIO_RANK_WEIGHT", os.environ.get("V20_RANK_VALUE_WEIGHT", "0.04")
        )
    )
    variants = [
        {
            "V20_RANK_VALUE_WEIGHT": "0.04",
            "V20_BIAS_WEIGHT": "0.045",
            "V20_DISABLE_SUPPORT": "1",
            "V20_INFER_N_PLAYERS": "1",
        },
        {
            "V20_RANK_VALUE_WEIGHT": "0.04",
            "V20_BIAS_WEIGHT": "0.045",
            "V20_DISABLE_SUPPORT": "1",
        },
        {
            "V20_RANK_VALUE_WEIGHT": "0.04",
            "V20_BIAS_WEIGHT": "0.045",
            "V20_DISABLE_SUPPORT": "1",
            "V20_DISABLE_PRESSURE_LEADER": "1",
        },
        {
            "V20_RANK_VALUE_WEIGHT": "0.04",
            "V20_BIAS_WEIGHT": "0.045",
        },
    ]
    candidates: list[list] = []
    for overrides in variants:
        move = _run_with_env(
            overrides, lambda: v20_search.search(obs, config, time_budget=per_budget)
        )
        if isinstance(move, list) and move:
            candidates.append(move)
    try:
        policy_fn = v20_search.policy_fn_from_intent(LAST_INTENT)
        move = v15_search.search(
            obs, config, time_budget=per_budget, horizon=22, policy_fn=policy_fn
        )
        if isinstance(move, list) and move:
            candidates.append(move)
    except Exception:
        pass
    if not candidates:
        return v20_search.search(obs, config)

    best_move: list | None = None
    best_score = v20_rank_value.eval_combo(
        fs, player, [], horizon, False, v15_eval.ESC, rank_weight
    )
    seen: set[tuple[tuple[int, int], ...]] = set()
    for move in candidates:
        key = tuple(
            sorted(
                (int(s[0]), int(s[2]))
                for s in move
                if isinstance(s, list) and len(s) == 3
            )
        )
        if key in seen:
            continue
        seen.add(key)
        try:
            score = v20_rank_value.eval_combo(
                fs, player, move, horizon, False, v15_eval.ESC, rank_weight
            )
        except Exception:
            continue
        if score > best_score + 0.0002:
            best_score = score
            best_move = move
    return best_move if best_move is not None else []


def _profile_agent(obs, config=None) -> list:
    player = v20_search._player_from_obs(obs)
    step = _obs_step(obs)
    if step <= 1 or player not in PROFILE_BY_PLAYER:
        PROFILE_BY_PLAYER[player] = _opening_profile(obs, player)
    profile = PROFILE_BY_PLAYER.get(player, "gate")
    overrides = _profile_overrides(profile)
    return _run_with_env(overrides, lambda: v20_search.search(obs, config))


def _opening_profile(obs, player: int) -> str:
    rows = _obs_rows(obs, "planets")
    if not rows:
        return "gate"
    mine = [p for p in rows if int(p[1]) == int(player)]
    neutrals = [p for p in rows if int(p[1]) == -1]
    enemies = [p for p in rows if int(p[1]) >= 0 and int(p[1]) != int(player)]
    if not mine or not neutrals or not enemies:
        return "gate"
    home = mine[0]
    hx, hy = float(home[2]), float(home[3])
    nearest_enemy = min(_dist(hx, hy, float(p[2]), float(p[3])) for p in enemies)
    ranked_neutrals = sorted(
        neutrals, key=lambda p: _dist(hx, hy, float(p[2]), float(p[3]))
    )
    near = ranked_neutrals[:4]
    nearest_neutral = _dist(hx, hy, float(near[0][2]), float(near[0][3]))
    near_prod = sum(float(p[6]) for p in near)
    static_near = sum(1 for p in near if _is_static_initial(p))
    seat = int(player)

    if nearest_enemy < 55.0:
        if nearest_neutral > 18.0 or seat == 3:
            return "strict"
        if near_prod >= 12.0:
            return "infer"
        return "nosupport"
    if seat == 3 and near_prod >= 14.0:
        return "strict"
    if seat == 1 and near_prod >= 10.0 and static_near == 2:
        return "support"
    if near_prod >= 14.0:
        return "nosupport"
    return "gate"


def _profile_overrides(profile: str) -> dict[str, str]:
    common = {
        "V20_BIAS_WEIGHT": "0.045",
    }
    broad_pressure = {
        "V20_PRESSURE_MIN_STEP": "35",
        "V20_PRESSURE_MAX_REACH": "120",
        "V20_PRESSURE_MIN_MY_PROD_SHARE": "0",
        "V20_PRESSURE_MIN_SCORE_VS_WEAK": "0",
    }
    if profile == "strict":
        return {
            **common,
            **broad_pressure,
            "V20_RANK_VALUE_WEIGHT": "0",
            "V20_DISABLE_SUPPORT": "1",
            "V20_INFER_N_PLAYERS": "1",
        }
    if profile == "support":
        return {
            **common,
            "V20_RANK_VALUE_WEIGHT": "0.04",
            "V20_INFER_N_PLAYERS": "1",
        }
    if profile == "infer":
        return {
            **common,
            "V20_RANK_VALUE_WEIGHT": "0.04",
            "V20_DISABLE_SUPPORT": "1",
            "V20_INFER_N_PLAYERS": "1",
        }
    if profile == "nosupport":
        return {
            **common,
            **broad_pressure,
            "V20_RANK_VALUE_WEIGHT": "0.04",
            "V20_DISABLE_SUPPORT": "1",
            "V20_INFER_N_PLAYERS": "1",
        }
    return {
        **common,
        "V20_RANK_VALUE_WEIGHT": "0.04",
        "V20_DISABLE_SUPPORT": "1",
        "V20_PRESSURE_MIN_SCORE_VS_WEAK": "1.08",
    }


def _obs_rows(obs, key: str) -> list:
    try:
        return list(
            obs.get(key, []) if isinstance(obs, dict) else getattr(obs, key, [])
        )
    except Exception:
        return []


def _obs_step(obs) -> int:
    try:
        return int(
            obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0)
        )
    except Exception:
        return 0


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _is_static_initial(row) -> bool:
    return _dist(float(row[2]), float(row[3]), 50.0, 50.0) + float(row[4]) >= 50.0


def _run_with_env(overrides: dict[str, str], fn):
    managed = set(overrides) | {
        "V20_DISABLE_SUPPORT",
        "V20_DISABLE_PRESSURE_LEADER",
        "V20_INFER_N_PLAYERS",
        "V20_PRESSURE_MIN_STEP",
        "V20_PRESSURE_MAX_REACH",
        "V20_PRESSURE_MIN_MY_PROD_SHARE",
        "V20_PRESSURE_MIN_SCORE_VS_WEAK",
        "V20_ENABLE_MACRO_CANDIDATES",
        "V20_ENABLE_TOP10_POLICY",
        "V20_EVAL_WEIGHTS",
        "V20_V15_SEARCH_ONLY",
    }
    old = {key: os.environ.get(key) for key in managed}
    try:
        for key in managed:
            os.environ.pop(key, None)
        for key, value in overrides.items():
            os.environ[key] = value
        return fn()
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


if __name__ == "__main__":
    import v14_core
    from local_simulator.official_fast import OfficialFastGame

    for n_players in (2, 4):
        game = OfficialFastGame(
            n_players=n_players, seed=20, episode_steps=160, use_c_accel=False
        )
        for _ in range(25):
            game.step([[] for _ in range(n_players)])
        move = agent(v14_core.obs_as_dict(game.observation(0)), game.configuration)
        intent = last_intent()
        name = intent.name if intent is not None else "none"
        print(
            f"{n_players}p intent={name} launches={len(move) if isinstance(move, list) else 0}"
        )
