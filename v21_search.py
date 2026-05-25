"""V21 search scaffold: learned-candidate ranking plus deterministic guardrails."""

from __future__ import annotations

from itertools import combinations
import os
import time
from typing import Any

import bot_v7
import v15_eval
import v15_fast_sim as fsim
import v15_search
import v20_search
import v21_policy_ranker


def search(
    obs,
    config=None,
    *,
    time_budget: float | None = None,
    horizon: int | None = None,
    ranker_path: str | None = None,
) -> list:
    """Kaggle-compatible V21 search.

    The initial V21 runtime is deliberately conservative: it enumerates legal
    V15/V7 shots, ranks them with a V21 candidate-ranker surface, then evaluates
    a bounded beam of valid combos with V15 ESC.  A missing learned checkpoint
    falls back to deterministic default ranker weights.
    """
    budget = float(time_budget if time_budget is not None else os.environ.get("V21_TIME_BUDGET", "0.45"))
    deadline = time.monotonic() + max(0.03, budget)
    h = int(horizon if horizon is not None else os.environ.get("V21_HORIZON", "20"))
    top_k = max(1, int(os.environ.get("V21_TOP_K", "10")))
    beam_width = max(1, int(os.environ.get("V21_BEAM_WIDTH", "24")))
    max_combo = max(1, int(os.environ.get("V21_MAX_COMBO", "4")))
    min_gain = float(os.environ.get("V21_MIN_GAIN", "0.00005"))

    try:
        player = _player_from_obs(obs)
        fs = _state_from_obs(obs, config)
        try:
            v7_move = bot_v7.agent(obs, config)
        except Exception:
            v7_move = []
        if not isinstance(v7_move, list):
            v7_move = []

        atomic = v15_search._enumerate_shots(fs, player, v7_move)
        if not atomic:
            return v20_search.fallback_v15(obs, config)
        ranker = v21_policy_ranker.LinearCandidateRanker.load(
            ranker_path if ranker_path is not None else os.environ.get("V21_RANKER", "")
        )
        ranked = v21_policy_ranker.rank_candidates(fs, player, atomic, ranker=ranker)
        if not ranked:
            return v20_search.fallback_v15(obs, config)
        top = [row.shot for row in ranked[:top_k]]
        n_players = max(2, int(getattr(fs, "n_players", 2) or 2))
        max_esc_loss = _env_float("V21_MAX_ESC_LOSS", _env_float("V21_MAX_ESC_LOSS_4P", 0.0) if n_players >= 4 else 0.0)
        rank_bias = _env_float("V21_RANK_BIAS_WEIGHT", _env_float("V21_RANK_BIAS_WEIGHT_4P", 0.0) if n_players >= 4 else 0.0)
        rank_bonus = _rank_bonus_map(ranked[:top_k])

        baseline = v15_search._eval_combo(fs, player, [], h, False, v15_eval.ESC)
        best_combo: list = []
        best_score = baseline
        best_obj = baseline
        for combo in _beam_combos(top, max_combo=max_combo, beam_width=beam_width):
            if _strict_time() and time.monotonic() > deadline:
                break
            if not v15_search._valid_combo(combo):
                continue
            score = v15_search._eval_combo(fs, player, combo, h, False, v15_eval.ESC)
            obj = score + rank_bias * _combo_rank_bonus(combo, rank_bonus)
            if score + max_esc_loss < baseline:
                continue
            if obj > best_obj + min_gain:
                best_score = score
                best_obj = obj
                best_combo = combo
        if best_combo:
            return best_combo
        return v20_search.fallback_v15(obs, config) if _env_flag("V21_EMPTY_RETURNS_FALLBACK") else []
    except Exception:
        return v20_search.fallback_v15(obs, config)


def _beam_combos(shots: list[list], *, max_combo: int, beam_width: int) -> list[list[list]]:
    """Deterministically build source-compatible combos in ranked-shot order."""
    beam: list[list[list]] = [[]]
    emitted: list[list[list]] = []
    for shot in shots:
        next_beam = list(beam)
        for combo in beam:
            candidate = combo + [shot]
            if len(candidate) > max_combo or not v15_search._valid_combo(candidate):
                continue
            emitted.append(candidate)
            next_beam.append(candidate)
        next_beam.sort(key=lambda c: (len(c), tuple(int(s[0]) for s in c)), reverse=True)
        beam = next_beam[:beam_width]
    # Include simple combinations as a safety net when top_k is small.
    for r in range(1, min(max_combo, len(shots)) + 1):
        for combo_tuple in combinations(shots[: min(len(shots), 8)], r):
            combo = list(combo_tuple)
            if v15_search._valid_combo(combo):
                emitted.append(combo)
    unique: list[list[list]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for combo in emitted:
        key = tuple(sorted((int(s[0]), int(round(float(s[1]) * 1000))) for s in combo))
        if key in seen:
            continue
        seen.add(key)
        unique.append(combo)
        if len(unique) >= beam_width:
            break
    return unique


def _rank_bonus_map(ranked: list[v21_policy_ranker.RankedCandidate]) -> dict[tuple[int, int], float]:
    total = max(1, len(ranked))
    out: dict[tuple[int, int], float] = {}
    for idx, row in enumerate(ranked):
        out[_shot_bonus_key(row.shot)] = max(0.0, (total - idx) / total)
    return out


def _combo_rank_bonus(combo: list[list], rank_bonus: dict[tuple[int, int], float]) -> float:
    if not combo:
        return 0.0
    total = 0.0
    for shot in combo:
        total += rank_bonus.get(_shot_bonus_key(shot), 0.0)
    return total / max(1.0, len(combo) ** 0.5)


def _shot_bonus_key(shot: list) -> tuple[int, int]:
    return (int(shot[0]), int(round(float(shot[1]) * 1000)))


def _state_from_obs(obs, config=None) -> fsim.FastState:
    n_players = _n_players_from_obs(obs, config)
    episode_steps = _config_int(config, "episodeSteps", _obs_int(obs, "episode_steps", 500))
    ship_speed = float(_config_get(config, "shipSpeed", 6.0))
    return fsim.from_obs(obs, n_players=n_players, episode_steps=episode_steps, ship_speed=ship_speed)


def _player_from_obs(obs) -> int:
    if isinstance(obs, dict):
        return int(obs.get("player", 0) or 0)
    return int(getattr(obs, "player", 0) or 0)


def _n_players_from_obs(obs, config=None) -> int:
    for val in (
        _config_get(config, "nPlayers", None),
        _config_get(config, "n_players", None),
        _obs_get(obs, "n_players", None),
        _obs_get(obs, "num_players", None),
    ):
        if val is not None:
            try:
                n = int(val)
                if n in (2, 4):
                    return n
            except Exception:
                pass
    owners = []
    for row in _obs_get(obs, "planets", []) or []:
        try:
            owner = int(row[1])
            if owner >= 0:
                owners.append(owner)
        except Exception:
            pass
    return max(2, min(4, max(owners, default=1) + 1))


def _obs_get(obs, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _obs_int(obs, key: str, default: int) -> int:
    try:
        return int(_obs_get(obs, key, default))
    except Exception:
        return int(default)


def _config_get(config, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:
            pass
    return getattr(config, key, default)


def _config_int(config, key: str, default: int) -> int:
    try:
        return int(_config_get(config, key, default))
    except Exception:
        return int(default)


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _strict_time() -> bool:
    return _env_flag("V21_STRICT_TIME")
