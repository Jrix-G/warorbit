"""V14 hybrid bot: V12/V13 tactical candidates + supervised neural ranker."""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any

import numpy as np

import bot_v12
import v14_core


_CACHE: dict[str, v14_core.V14Scorer | None] = {}
_FOUR_PLAYER_AGENT: Any | None = None
_FOUR_PLAYER_MODULES: list[Any] | None = None
_FOUR_PLAYER_AGENT_LOADED = False


_FOUR_PLAYER_PROFILES: dict[str, dict[str, float | int | bool]] = {
    "base": {},
    "anti_pool": {
        "FOUR_PLAYER_ROTATING_REACTION_GAP": 1,
        "FOUR_PLAYER_ROTATING_SEND_RATIO": 0.72,
        "FOUR_PLAYER_ROTATING_TURN_LIMIT": 14,
        "FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT": 0.94,
        "FOUR_PLAYER_TARGET_MARGIN": 2,
        "MULTI_SOURCE_TOP_K": 10,
        "THREE_SOURCE_ETA_TOLERANCE": 2,
        "THREE_SOURCE_PLAN_PENALTY": 0.94,
        "FINISHING_HOSTILE_SEND_BONUS": 4,
        "WEAK_ENEMY_THRESHOLD": 70,
        "ELIMINATION_BONUS": 34.0,
        "AHEAD_ATTACK_MARGIN_BONUS": 0.12,
        "BEHIND_ATTACK_MARGIN_PENALTY": 0.04,
        "FINISHING_ATTACK_MARGIN_BONUS": 0.12,
    },
    "closer": {
        "FOUR_PLAYER_ROTATING_REACTION_GAP": 1,
        "FOUR_PLAYER_ROTATING_SEND_RATIO": 0.78,
        "FOUR_PLAYER_ROTATING_TURN_LIMIT": 16,
        "FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT": 1.02,
        "FOUR_PLAYER_TARGET_MARGIN": 1,
        "HOSTILE_MARGIN_BASE": 2,
        "HOSTILE_MARGIN_CAP": 10,
        "MULTI_SOURCE_TOP_K": 12,
        "THREE_SOURCE_ETA_TOLERANCE": 2,
        "THREE_SOURCE_PLAN_PENALTY": 0.96,
        "FOUR_SOURCE_PLAN_PENALTY": 0.94,
        "FINISHING_HOSTILE_SEND_BONUS": 6,
        "WEAK_ENEMY_THRESHOLD": 85,
        "ELIMINATION_BONUS": 48.0,
        "AHEAD_ATTACK_MARGIN_BONUS": 0.14,
        "BEHIND_ATTACK_MARGIN_PENALTY": 0.03,
        "FINISHING_ATTACK_MARGIN_BONUS": 0.16,
    },
    "eco": {
        "FOUR_PLAYER_ROTATING_REACTION_GAP": 1,
        "FOUR_PLAYER_ROTATING_SEND_RATIO": 0.68,
        "FOUR_PLAYER_ROTATING_TURN_LIMIT": 15,
        "FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT": 1.08,
        "FOUR_PLAYER_TARGET_MARGIN": 3,
        "SAFE_NEUTRAL_MARGIN": 1,
        "CONTESTED_NEUTRAL_MARGIN": 1,
        "WEAK_ENEMY_THRESHOLD": 60,
        "ELIMINATION_BONUS": 26.0,
    },
    "deep_eco": {
        "HORIZON": 180,
        "SIM_HORIZON": 180,
        "ROUTE_SEARCH_HORIZON": 80,
        "SOFT_ACT_DEADLINE": 1.45,
        "HEAVY_PHASE_MIN_TIME": 0.22,
        "OPTIONAL_PHASE_MIN_TIME": 0.12,
        "FOUR_PLAYER_ROTATING_REACTION_GAP": 1,
        "FOUR_PLAYER_ROTATING_SEND_RATIO": 0.70,
        "FOUR_PLAYER_ROTATING_TURN_LIMIT": 16,
        "FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT": 1.10,
        "FOUR_PLAYER_TARGET_MARGIN": 2,
        "SAFE_NEUTRAL_MARGIN": 1,
        "CONTESTED_NEUTRAL_MARGIN": 1,
        "MULTI_SOURCE_TOP_K": 10,
        "THREE_SOURCE_ETA_TOLERANCE": 2,
        "THREE_SOURCE_PLAN_PENALTY": 0.95,
        "FOUR_SOURCE_PLAN_PENALTY": 0.94,
        "WEAK_ENEMY_THRESHOLD": 65,
        "ELIMINATION_BONUS": 32.0,
    },
}


def _is_four_player(obs: dict[str, Any]) -> bool:
    planets = list(obs.get("planets", []) or [])
    owners = {int(p[1]) for p in planets if int(p[1]) >= 0}
    return len(owners) >= 4


def _dist(a: list, b: list) -> float:
    return math.hypot(float(a[2]) - float(b[2]), float(a[3]) - float(b[3]))


def _enemy_strengths(planets: list, fleets: list, me: int) -> dict[int, dict[str, float]]:
    ids = {int(p[1]) for p in planets if int(p[1]) not in (-1, me)}
    result: dict[int, dict[str, float]] = {}
    for eid in ids:
        ep = [p for p in planets if int(p[1]) == eid]
        ef = [f for f in fleets if int(f[1]) == eid]
        ships = sum(float(p[5]) for p in ep) + sum(float(f[6]) for f in ef)
        prod = sum(float(p[6]) for p in ep)
        result[eid] = {
            "planets": float(len(ep)),
            "ships": ships,
            "prod": prod,
            "strength": ships + 12.0 * prod + 35.0 * len(ep),
        }
    return result


def _candidate_target(obs: dict[str, Any], candidate: dict) -> list | None:
    planets = list(obs.get("planets", []) or [])
    moves = list(candidate.get("moves", []) or [])
    if not moves:
        return None
    return v14_core._infer_target(planets, moves[0])


def _candidate_sent(candidate: dict) -> float:
    return float(sum(max(0, int(m[2])) for m in candidate.get("moves", []) or [] if len(m) >= 3))


def _nearest_my_distance(planets: list, my_planets: list, target: list | None) -> float:
    if target is None or not my_planets:
        return 999.0
    return min(_dist(src, target) for src in my_planets)


def _four_player_heuristic_scores(obs: dict[str, Any], candidates: list[dict], scorer_scores: np.ndarray) -> np.ndarray:
    planets = list(obs.get("planets", []) or [])
    fleets = list(obs.get("fleets", []) or [])
    me = int(obs.get("player", 0) or 0)
    step = int(obs.get("step", 0) or 0)
    my_planets = [p for p in planets if int(p[1]) == me]
    my_ships = sum(float(p[5]) for p in my_planets) + sum(float(f[6]) for f in fleets if int(f[1]) == me)
    my_prod = sum(float(p[6]) for p in my_planets)
    strengths = _enemy_strengths(planets, fleets, me)
    ranked_enemies = sorted(strengths, key=lambda eid: strengths[eid]["strength"])
    weakest = ranked_enemies[0] if ranked_enemies else None
    strongest = ranked_enemies[-1] if ranked_enemies else None
    total_enemy_strength = sum(v["strength"] for v in strengths.values()) or 1.0
    my_strength = my_ships + 12.0 * my_prod + 35.0 * len(my_planets)
    leader_pressure = my_strength > 0.42 * (my_strength + total_enemy_strength)

    scores = np.full(len(candidates), -20.0, dtype=np.float32)
    scorer_scores = scorer_scores.astype(np.float32) if scorer_scores.size == len(candidates) else np.zeros(len(candidates), dtype=np.float32)

    for i, cand in enumerate(candidates):
        ctype = str(cand.get("type", "noop"))
        target = _candidate_target(obs, cand)
        owner = int(target[1]) if target is not None else -99
        sent = _candidate_sent(cand)
        nearest = _nearest_my_distance(planets, my_planets, target)
        hint = float(cand.get("score_hint", 0.0))
        score = 0.03 * float(scorer_scores[i]) + 0.15 * math.tanh(hint)

        if ctype == "noop":
            scores[i] = -8.0 if my_planets else 0.0
            continue

        if ctype == "defense":
            score += 8.0 + 2.0 * min(2.0, sent / max(1.0, my_ships * 0.08))
            scores[i] = score
            continue

        if ctype in ("staging",):
            score += 1.5
            if leader_pressure:
                score += 1.0
            if step < 70:
                score += 1.0
            scores[i] = score
            continue

        if owner == -1 or ctype in ("expand", "opportunistic_expand"):
            if target is None:
                continue
            prod = float(target[6])
            defenders = float(target[5])
            margin = sent - defenders
            close_bonus = max(0.0, (55.0 - nearest) / 55.0)
            robust = 1.0 if margin >= max(6.0, 1.5 * prod) else -1.0
            score += 3.0 + 1.8 * close_bonus + 0.55 * prod + 0.04 * margin + robust
            if step < 65 and nearest > 58.0:
                score -= 4.0
            if sent < 14.0 and step < 80:
                score -= 2.5
            if leader_pressure and step > 70:
                score -= 1.0
            scores[i] = score
            continue

        if owner in strengths:
            enemy = strengths[owner]
            is_weakest = owner == weakest
            is_strongest = owner == strongest
            # In 4p openings every enemy starts with very few planets; treating
            # "few planets" as finishable causes suicidal turn-0 rushes.
            almost_dead = (
                enemy["ships"] < 0.45 * max(1.0, my_ships)
                and enemy["prod"] < 0.75 * max(1.0, my_prod)
            ) or enemy["strength"] < 0.35 * max(1.0, my_strength)
            margin = sent - (float(target[5]) if target is not None else 0.0)

            if ctype == "focus_finish":
                if step < 70 and not almost_dead:
                    scores[i] = -14.0 + 0.01 * margin
                    continue
                score += 6.0 if is_weakest else -4.0
                score += 3.0 if almost_dead else -1.0
                score += 0.03 * margin
            else:
                if step < 70 and not almost_dead:
                    score -= 6.0
                score += 3.5 if is_weakest else -2.0
                score -= 3.5 if is_strongest and not leader_pressure else 0.0
                score += 2.0 if almost_dead else 0.0
                score += 0.025 * margin
                if nearest > 65.0:
                    score -= 2.0
            scores[i] = score
            continue

        scores[i] = score
    return scores


def _load_scorer() -> v14_core.V14Scorer | None:
    path = os.environ.get("V14_WEIGHTS", "evaluations/scorer_v14.npz")
    key = str(Path(path))
    if key in _CACHE:
        return _CACHE[key]
    p = Path(path)
    if not p.exists():
        _CACHE[key] = None
        return None
    try:
        scorer = v14_core.V14Scorer.load(p)
    except Exception:
        scorer = None
    _CACHE[key] = scorer
    return scorer


def _load_four_player_agent() -> Any | None:
    global _FOUR_PLAYER_AGENT, _FOUR_PLAYER_MODULES, _FOUR_PLAYER_AGENT_LOADED
    if _FOUR_PLAYER_AGENT_LOADED:
        return _FOUR_PLAYER_AGENT
    _FOUR_PLAYER_AGENT_LOADED = True
    runtime = os.environ.get("V14_4P_AGENT", "distance").strip().lower()
    module_name = {
        "orbitbotnext": "opponents.notebook_orbitbotnext",
        "distance": "opponents.notebook_distance_prioritized",
        "physics": "opponents.notebook_physics_accurate",
        "pascal": "opponents.notebook_pascalledesma_orbitwork_v14",
    }.get(runtime, "portfolio")
    try:
        if module_name == "portfolio":
            names = [
                "opponents.notebook_distance_prioritized",
                "opponents.notebook_orbitbotnext",
                "opponents.notebook_physics_accurate",
            ]
            modules = [__import__(name, fromlist=["agent"]) for name in names]
            four_player_agent = None
        else:
            modules = [__import__(module_name, fromlist=["agent"])]
            four_player_agent = getattr(modules[0], "agent")
    except Exception:
        _FOUR_PLAYER_AGENT = None
        _FOUR_PLAYER_MODULES = None
    else:
        profile = os.environ.get("V14_4P_PROFILE", "eco").strip().lower()
        patch = _FOUR_PLAYER_PROFILES.get(profile, _FOUR_PLAYER_PROFILES["eco"])
        for module in modules:
            for name, value in patch.items():
                if hasattr(module, name):
                    setattr(module, name, value)
        _FOUR_PLAYER_AGENT = four_player_agent
        _FOUR_PLAYER_MODULES = modules
    return _FOUR_PLAYER_AGENT


def _safe_agent_call(agent_fn: Any, obs: Any, config: Any = None) -> list[list] | None:
    try:
        actions = agent_fn(obs, config)
    except TypeError:
        try:
            actions = agent_fn(obs)
        except Exception:
            return None
    except Exception:
        return None
    return actions if isinstance(actions, list) else None


def _score_action_batch(obs: dict[str, Any], actions: list[list]) -> float:
    planets = list(obs.get("planets", []) or [])
    fleets = list(obs.get("fleets", []) or [])
    me = int(obs.get("player", 0) or 0)
    step = int(obs.get("step", 0) or 0)
    if not actions:
        my_planets = [p for p in planets if int(p[1]) == me]
        return -4.0 if my_planets else 0.0

    my_planets = [p for p in planets if int(p[1]) == me]
    my_ids = {int(p[0]) for p in my_planets}
    my_prod = sum(float(p[6]) for p in my_planets)
    my_ships = sum(float(p[5]) for p in my_planets) + sum(float(f[6]) for f in fleets if int(f[1]) == me)
    enemy_stats = _enemy_strengths(planets, fleets, me)
    weakest = min(enemy_stats, key=lambda eid: enemy_stats[eid]["strength"]) if enemy_stats else None
    score = 0.0

    for move in actions:
        if not isinstance(move, (list, tuple)) or len(move) < 3:
            continue
        src = next((p for p in planets if int(p[0]) == int(move[0])), None)
        if src is None or int(src[0]) not in my_ids:
            score -= 10.0
            continue
        sent = max(0.0, float(move[2]))
        target = v14_core._infer_target(planets, move)
        if target is None:
            score -= 1.0 + 0.02 * sent
            continue
        owner = int(target[1])
        prod = float(target[6])
        defenders = float(target[5])
        distance = _dist(src, target)
        margin = sent - defenders

        if owner == -1:
            local = 5.0 + 1.8 * prod - 0.10 * defenders - 0.035 * distance
            if margin >= 1.0:
                local += min(4.0, 0.12 * margin)
            else:
                local -= 5.0 + abs(margin)
            if step < 80:
                local += 2.0 + 0.9 * prod
            if sent > max(8.0, 0.85 * max(1.0, float(src[5]))) and step < 45:
                local -= 1.5
            score += local
            continue

        if owner == me:
            incoming = sum(float(f[6]) for f in fleets if int(f[1]) != me and len(f) > 5 and int(f[5]) == int(target[0]))
            score += 2.0 + 0.12 * incoming + 0.3 * prod - 0.02 * distance
            continue

        enemy = enemy_stats.get(owner, {"strength": 999.0, "ships": 999.0, "prod": 999.0})
        almost_dead = enemy["ships"] < 0.55 * max(1.0, my_ships) and enemy["prod"] <= max(1.0, 0.8 * my_prod)
        local = 1.0 + 1.2 * prod - 0.06 * defenders - 0.03 * distance + 0.08 * margin
        if owner == weakest:
            local += 2.2
        if almost_dead:
            local += 4.0
        if step < 70 and not almost_dead:
            local -= 5.0
        score += local

    return score


def _call_four_player_agent(obs: Any, config: Any = None) -> list[list] | None:
    agent_fn = _load_four_player_agent()
    if _FOUR_PLAYER_MODULES and agent_fn is None:
        obs_dict = v14_core.obs_as_dict(obs)
        best_actions: list[list] | None = None
        best_score = -1e9
        for module in _FOUR_PLAYER_MODULES:
            actions = _safe_agent_call(getattr(module, "agent", None), obs, config)
            if actions is None:
                continue
            score = _score_action_batch(obs_dict, actions)
            if score > best_score:
                best_score = score
                best_actions = actions
        return best_actions
    if agent_fn is None:
        return None
    return _safe_agent_call(agent_fn, obs, config)


def agent(obs: Any, config: Any = None) -> list[list]:
    try:
        return _agent_inner(obs, config)
    except Exception:
        return bot_v12.agent(obs, config)


def _agent_inner(obs: Any, config: Any = None) -> list[list]:
    scorer = _load_scorer()
    if scorer is None:
        return bot_v12.agent(obs, config)

    obs_dict = v14_core.obs_as_dict(obs)
    if _is_four_player(obs_dict) and os.environ.get("V14_4P_RUNTIME", "notebook").lower() != "ml":
        actions_4p = _call_four_player_agent(obs, config)
        if actions_4p is not None:
            return actions_4p

    candidates = v14_core.get_candidates(obs_dict)
    if not candidates:
        return bot_v12.agent(obs, config)
    feats = v14_core.candidate_matrix(obs_dict, candidates)
    scores = scorer.forward(feats)
    if _is_four_player(obs_dict):
        scores = _four_player_heuristic_scores(obs_dict, candidates, scores)
    actions = v14_core.select_actions(candidates, scores)

    # Conservative fallback: if the ranker wants to do nothing early while V12
    # sees a move, trust the tactical baseline.
    step = int(obs_dict.get("step", 0) or 0)
    if not actions and step < 160 and not _is_four_player(obs_dict):
        fallback = bot_v12.agent(obs, config)
        return fallback if isinstance(fallback, list) else []
    return actions


def get_candidates_and_scores(obs: Any) -> tuple[list[dict], np.ndarray]:
    scorer = _load_scorer()
    obs_dict = v14_core.obs_as_dict(obs)
    candidates = v14_core.get_candidates(obs_dict)
    if scorer is None or not candidates:
        return candidates, np.zeros(len(candidates), dtype=np.float32)
    scores = scorer.forward(v14_core.candidate_matrix(obs_dict, candidates))
    if _is_four_player(obs_dict):
        scores = _four_player_heuristic_scores(obs_dict, candidates, scores)
    return candidates, scores
