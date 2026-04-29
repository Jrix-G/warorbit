"""Orbit Wars V8 inference bot.

The learned part ranks a few complete candidate plans:
- V7 baseline plan,
- attack / expansion / defense / comet / reserve variants.

When the checkpoint is missing, the policy falls back to the V7 baseline.
"""

from __future__ import annotations

import os
from typing import Optional

import bot_v7

from v8_core import MODEL_PATH, LinearV8Model, build_candidate_plans, build_state_features, select_plan


_MODEL: Optional[LinearV8Model] = None


def get_model() -> LinearV8Model:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if os.path.exists(MODEL_PATH):
        _MODEL = LinearV8Model.load(MODEL_PATH)
    else:
        _MODEL = LinearV8Model.zero()
    return _MODEL


def set_model(model: Optional[LinearV8Model]) -> None:
    global _MODEL
    _MODEL = model


def select_candidate(obs) -> tuple:
    bot_v7.set_scorer(None)
    world = bot_v7._build_world(obs)
    model = get_model()
    plan, plans, state = select_plan(world, model)
    return plan, plans, state


def agent(obs, config=None):
    try:
        bot_v7.set_scorer(None)
        world = bot_v7._build_world(obs)
        if not world.my_planets:
            return []
        model = get_model()
        plan, _, _ = select_plan(world, model)
        return plan.actions or []
    except Exception:
        return []


def state_policy(game, player: int):
    """SimGame/FastState-compatible policy."""
    try:
        obs = game.observation(player)
        return agent(obs, None)
    except Exception:
        return []


def bootstrap_zero() -> None:
    """Force an explicit zero model in memory."""
    set_model(LinearV8Model.zero())


__all__ = [
    "agent",
    "bootstrap_zero",
    "get_model",
    "set_model",
    "select_candidate",
    "state_policy",
]
