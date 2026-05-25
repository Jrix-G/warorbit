"""V22 agent: combo model search with V20/V15 fallback.

V22 is deliberately separate from V21 routing.  It always tries its own combo
search first, then falls back conservatively.
"""

from __future__ import annotations

import os
import time

import bot_v7
import v15_eval
import v15_search
import v20_search
import v21_policy_ranker
import v21_search
import v22_features
import v22_model
import v22_oracle


def agent(obs, config=None, *, time_budget: float | None = None, horizon: int | None = None):
    if _flag("V22_DISABLE"):
        return v20_search.fallback_v15(obs, config)
    try:
        return search(obs, config, time_budget=time_budget, horizon=horizon)
    except Exception:
        return v20_search.fallback_v15(obs, config)


def search(obs, config=None, *, time_budget: float | None = None, horizon: int | None = None) -> list:
    budget = float(time_budget if time_budget is not None else os.environ.get("V22_TIME_BUDGET", "0.45"))
    deadline = time.monotonic() + max(0.02, budget)
    h = int(horizon if horizon is not None else os.environ.get("V22_HORIZON", "14"))
    det_h = int(os.environ.get("V22_DET_HORIZON", str(max(4, h // 2))))
    top_k = max(1, int(os.environ.get("V22_TOP_K", "10")))
    beam_width = max(1, int(os.environ.get("V22_BEAM_WIDTH", "32")))
    max_combo = max(1, int(os.environ.get("V22_MAX_COMBO", "4")))
    model_weight = float(os.environ.get("V22_MODEL_WEIGHT", "0.0"))
    det_weight = float(os.environ.get("V22_DET_WEIGHT", "0.45"))
    max_loss = float(os.environ.get("V22_MAX_ESC_LOSS", "0.025"))
    min_gain = float(os.environ.get("V22_MIN_GAIN", "0.00005"))

    player = v21_search._player_from_obs(obs)
    fs = v21_search._state_from_obs(obs, config)
    try:
        v7_move = bot_v7.agent(obs, config)
    except Exception:
        v7_move = []
    atomic = v15_search._enumerate_shots(fs, player, v7_move if isinstance(v7_move, list) else [])
    ranked = v21_policy_ranker.rank_candidates(fs, player, atomic)
    if not ranked:
        return v20_search.fallback_v15(obs, config)
    shots = [row.shot for row in ranked[:top_k]]
    combos = v21_search._beam_combos(shots, max_combo=max_combo, beam_width=beam_width)
    if not combos:
        return []

    passive_baseline = float(v15_search._eval_combo(fs, player, [], h, False, v15_eval.ESC))
    det_baseline = v22_oracle._eval_combo_det(fs, player, [], det_h)
    baseline_obj = passive_baseline + det_weight * (det_baseline - passive_baseline)
    model = v22_model.LinearComboRanker.load(os.environ.get("V22_MODEL", ""))
    dim = len(v22_features.FEATURE_NAMES)

    best_combo: list = []
    best_obj = baseline_obj - max(0.0, max_loss)
    seen: set[tuple[tuple[int, int], ...]] = set()
    for combo in combos:
        if time.monotonic() > deadline:
            break
        if not v15_search._valid_combo(combo):
            continue
        key = _combo_key(combo)
        if key in seen:
            continue
        seen.add(key)
        passive = float(v15_search._eval_combo(fs, player, combo, h, False, v15_eval.ESC))
        det = v22_oracle._eval_combo_det(fs, player, combo, det_h)
        feat = v22_features.combo_features(
            fs,
            player,
            combo,
            passive_score=passive,
            passive_baseline=passive_baseline,
            det_score=det,
            det_baseline=det_baseline,
            max_combo=max_combo,
        )
        learned = model.score(feat) if model.ready_for(dim) else 0.0
        obj = passive + det_weight * (det - passive) + model_weight * learned
        if obj + max_loss < baseline_obj:
            continue
        if obj > best_obj + min_gain:
            best_obj = obj
            best_combo = combo
    clean = _clean_combo(best_combo)
    if clean:
        return clean
    if _flag("V22_EMPTY_RETURNS_PASS"):
        return []
    return v20_search.search(obs, config, time_budget=min(0.08, max(0.02, budget * 0.5)), horizon=h)


def _clean_combo(combo: list) -> list:
    out = []
    for shot in combo:
        if isinstance(shot, list) and len(shot) == 3:
            ships = int(shot[2])
            if ships > 0:
                out.append([int(shot[0]), float(shot[1]), ships])
    return out


def _combo_key(combo: list) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(s[0]), int(round(float(s[1]) * 1000))) for s in combo))


def _flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}
