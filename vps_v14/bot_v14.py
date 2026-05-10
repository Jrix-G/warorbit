"""V14 hybrid bot: V12/V13 tactical candidates + supervised neural ranker."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

import bot_v12
import v14_core


_CACHE: dict[str, v14_core.V14Scorer | None] = {}


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
    candidates = v14_core.get_candidates(obs_dict)
    if not candidates:
        return bot_v12.agent(obs, config)
    feats = v14_core.candidate_matrix(obs_dict, candidates)
    scores = scorer.forward(feats)
    actions = v14_core.select_actions(candidates, scores)

    # Conservative fallback: if the ranker wants to do nothing early while V12
    # sees a move, trust the tactical baseline.
    step = int(obs_dict.get("step", 0) or 0)
    if not actions and step < 160:
        fallback = bot_v12.agent(obs, config)
        return fallback if isinstance(fallback, list) else []
    return actions


def get_candidates_and_scores(obs: Any) -> tuple[list[dict], np.ndarray]:
    scorer = _load_scorer()
    obs_dict = v14_core.obs_as_dict(obs)
    candidates = v14_core.get_candidates(obs_dict)
    if scorer is None or not candidates:
        return candidates, np.zeros(len(candidates), dtype=np.float32)
    return candidates, scorer.forward(v14_core.candidate_matrix(obs_dict, candidates))
