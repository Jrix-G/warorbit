"""Kaggle submission entry point for the current V9 checkpoint.

This wrapper intentionally uses the best V9 checkpoint found so far in local
testing. It is an experimental submission, not a validated upgrade over V7.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from war_orbit.agents.v9.policy import V9Agent, get_weights, load_checkpoint
from war_orbit.config.v9_config import V9Config


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "evaluations" / "v9_backbone_60m_policy.npz"

_AGENT: Optional[V9Agent] = None
_LOADED = False


def _build_agent() -> V9Agent:
    global _LOADED
    if not _LOADED:
        load_checkpoint(str(CHECKPOINT))
        _LOADED = True
    config = V9Config(
        front_lock_turns=20,
        target_active_fronts=2.0,
        target_backbone_turn_frac=0.15,
        front_pressure_plan_bias=0.14,
        front_pressure_attack_penalty=0.12,
        search_width=8,
        simulation_depth=3,
        simulation_rollouts=2,
        opponent_samples=4,
        candidate_diversity=1.15,
        exploration_rate=0.0,
    )
    return V9Agent(config=config, weights=get_weights())


def agent(obs, config=None):
    global _AGENT
    if _AGENT is None:
        _AGENT = _build_agent()
    return _AGENT(obs, config)


__all__ = ["agent"]
