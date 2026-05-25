"""V21 agent entry point."""

from __future__ import annotations

import os

import v15_eval
import v15_fast_sim as fsim
import v15_search
import v20_agent
import v20_search
import v21_search


def agent(obs, config=None, *, time_budget: float | None = None, horizon: int | None = None):
    if _env_flag("V21_DISABLE"):
        return v20_search.fallback_v15(obs, config)
    try:
        if _env_flag("V21_ENABLE_4P_PORTFOLIO") and _is_4p(obs, config):
            return _v21_4p_portfolio(obs, config, time_budget=time_budget, horizon=horizon)
        if _use_4p_guard(obs, config):
            return _v20_4p_guard(obs, config, time_budget=time_budget, horizon=horizon)
        return v21_search.search(obs, config, time_budget=time_budget, horizon=horizon)
    except Exception:
        return v20_search.fallback_v15(obs, config)


def _use_4p_guard(obs, config=None) -> bool:
    if _env_flag("V21_DISABLE_4P_GUARD"):
        return False
    if _env_flag("V21_FORCE_LEARNED"):
        return False
    try:
        return v21_search._n_players_from_obs(obs, config) >= 4
    except Exception:
        return False


def _is_4p(obs, config=None) -> bool:
    try:
        return v21_search._n_players_from_obs(obs, config) >= 4
    except Exception:
        return False


def _v20_4p_guard(obs, config=None, *, time_budget: float | None = None, horizon: int | None = None) -> list:
    overrides = {
        "V20_DISABLE_PRESSURE_LEADER": "1",
        "V20_DISABLE_SUPPORT": "1",
        "V20_RANK_VALUE_WEIGHT": os.environ.get("V21_4P_RANK_VALUE_WEIGHT", "0.04"),
        "V20_BIAS_WEIGHT": os.environ.get("V21_4P_BIAS_WEIGHT", "0.045"),
        "V20_INFER_N_PLAYERS": "1",
    }
    return v20_agent._run_with_env(
        overrides,
        lambda: v20_agent.agent(obs, config, time_budget=time_budget, horizon=horizon),
    )


def _v21_4p_portfolio(obs, config=None, *, time_budget: float | None = None, horizon: int | None = None) -> list:
    total_budget = float(time_budget if time_budget is not None else os.environ.get("V21_TIME_BUDGET", "0.45"))
    per_budget = float(os.environ.get("V21_PORTFOLIO_PER_BUDGET", max(0.018, total_budget * 0.45)))
    eval_horizon = int(os.environ.get("V21_PORTFOLIO_HORIZON", horizon if horizon is not None else 16))
    max_esc_loss = float(os.environ.get("V21_PORTFOLIO_MAX_ESC_LOSS", "0.0"))
    eval_mode = os.environ.get("V21_PORTFOLIO_EVAL", "passive").strip().lower()
    candidates: list[list] = []

    _append_move(
        candidates,
        v21_search.search(obs, config, time_budget=per_budget, horizon=horizon),
    )
    _append_move(
        candidates,
        _v20_4p_guard(obs, config, time_budget=per_budget, horizon=horizon),
    )
    _append_move(
        candidates,
        v20_agent._run_with_env(
            {},
            lambda: v20_agent.agent(obs, config, time_budget=per_budget, horizon=horizon),
        ),
    )
    if _env_flag("V21_PORTFOLIO_TOP10"):
        _append_move(
            candidates,
            v20_agent._run_with_env(
                {"V20_ENABLE_TOP10_POLICY": "1"},
                lambda: v20_agent.agent(obs, config, time_budget=per_budget, horizon=horizon),
            ),
        )
    if not candidates:
        return []

    try:
        fs = v21_search._state_from_obs(obs, config)
        player = v21_search._player_from_obs(obs)
        baseline = float(v15_search._eval_combo(fs, player, [], eval_horizon, False, v15_eval.ESC))
        if eval_mode == "det":
            baseline = _eval_combo_det(fs, player, [], eval_horizon)
        best_move: list = []
        best_score = baseline - max(0.0, max_esc_loss)
        seen: set[tuple[tuple[int, int], ...]] = set()
        for move in candidates:
            key = _move_key(move)
            if key in seen or not v15_search._valid_combo(move):
                continue
            seen.add(key)
            if eval_mode == "det":
                score = _eval_combo_det(fs, player, move, eval_horizon)
            else:
                score = float(v15_search._eval_combo(fs, player, move, eval_horizon, False, v15_eval.ESC))
            if score + max_esc_loss < baseline:
                continue
            if score > best_score:
                best_score = score
                best_move = move
        return best_move
    except Exception:
        return candidates[0] if candidates else []


def _append_move(candidates: list[list], move) -> None:
    clean = _normalise_move(move)
    if clean:
        candidates.append(clean)


def _normalise_move(move) -> list:
    out: list = []
    if not isinstance(move, list):
        return out
    for shot in move:
        if not isinstance(shot, list) or len(shot) != 3:
            continue
        try:
            ships = int(shot[2])
            if ships <= 0:
                continue
            out.append([int(shot[0]), float(shot[1]), ships])
        except Exception:
            continue
    return out


def _move_key(move: list) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(s[0]), int(round(float(s[1]) * 1000))) for s in move))


def _eval_combo_det(fs: fsim.FastState, player: int, combo: list, horizon: int) -> float:
    actions = v15_search._det_policy(fs)
    actions[int(player)] = list(combo)
    st = fsim.step(fs, actions)
    for _ in range(max(1, int(horizon)) - 1):
        if st.done:
            break
        st = fsim.step(st, v15_search._det_policy(st))
    return float(v15_eval.evaluate(st, int(player), v15_eval.ESC))


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}
