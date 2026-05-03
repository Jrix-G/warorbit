"""Auto-tuning helpers for V9 adaptation."""

from __future__ import annotations

from dataclasses import replace

from ..agents.v9.adaptation import AdaptationController, TrainingMetrics
from ..config.v9_config import V9Config


def apply_adaptation(config: V9Config, controller: AdaptationController) -> V9Config:
    overrides = controller.planning_overrides()
    return replace(
        config,
        exploration_rate=float(overrides["exploration_rate"]),
        candidate_diversity=float(overrides["candidate_diversity"]),
        rollout_weight=float(overrides["rollout_weight"]),
        uncertainty_penalty=float(overrides["uncertainty_penalty"]),
        finisher_bias=float(overrides["finisher_bias"]),
        sigma=float(config.sigma * controller.state.sigma_scale),
        learning_rate=float(config.learning_rate * controller.state.learning_rate_scale),
    )


def update_controller(
    controller: AdaptationController,
    *,
    generation: int,
    train_score: float,
    eval_score: float,
    eval_2p: float,
    eval_4p: float,
    promoted: bool,
):
    metrics = TrainingMetrics(
        generation=generation,
        train_score=float(train_score),
        eval_score=float(eval_score),
        eval_2p=float(eval_2p),
        eval_4p=float(eval_4p),
        promoted=bool(promoted),
    )
    return controller.observe(metrics)
