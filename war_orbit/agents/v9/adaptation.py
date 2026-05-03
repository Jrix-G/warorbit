"""Automatic adaptation for V9 training and runtime exploration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AdaptationState:
    generation: int = 0
    best_score: float = -1.0
    no_progress: int = 0
    exploration_rate: float = 0.08
    candidate_diversity: float = 1.0
    rollout_weight: float = 0.42
    uncertainty_penalty: float = 0.20
    finisher_bias: float = 1.0
    sigma_scale: float = 1.0
    learning_rate_scale: float = 1.0
    injected_plan_bias: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    generation: int
    train_score: float
    eval_score: float
    eval_2p: float
    eval_4p: float
    promoted: bool


class AdaptationController:
    """Detect stagnation and adjust exploration/search pressure."""

    def __init__(
        self,
        *,
        window: int = 4,
        min_improvement: float = 0.015,
        exploration_rate: float = 0.08,
        candidate_diversity: float = 1.0,
        rollout_weight: float = 0.42,
        uncertainty_penalty: float = 0.20,
        finisher_bias: float = 1.0,
    ):
        self.window = int(window)
        self.min_improvement = float(min_improvement)
        self.state = AdaptationState(
            exploration_rate=float(exploration_rate),
            candidate_diversity=float(candidate_diversity),
            rollout_weight=float(rollout_weight),
            uncertainty_penalty=float(uncertainty_penalty),
            finisher_bias=float(finisher_bias),
        )
        self.history: List[TrainingMetrics] = []

    def observe(self, metrics: TrainingMetrics) -> AdaptationState:
        self.history.append(metrics)
        self.state.generation = metrics.generation
        score = float(metrics.eval_score)
        improved = score >= self.state.best_score + self.min_improvement or metrics.promoted
        if improved:
            self.state.best_score = max(self.state.best_score, score)
            self.state.no_progress = 0
            self._anneal_after_progress()
        else:
            self.state.no_progress += 1
            if self.state.no_progress >= self.window:
                self._escape_stagnation()
        self._rebalance_mode_pressure(metrics)
        return self.state

    def _anneal_after_progress(self) -> None:
        self.state.exploration_rate = max(0.025, self.state.exploration_rate * 0.82)
        self.state.candidate_diversity = max(1.0, self.state.candidate_diversity * 0.92)
        self.state.sigma_scale = max(0.70, self.state.sigma_scale * 0.90)
        self.state.learning_rate_scale = min(1.15, self.state.learning_rate_scale * 1.03)
        self.state.uncertainty_penalty = min(0.35, self.state.uncertainty_penalty * 1.04)

    def _escape_stagnation(self) -> None:
        self.state.exploration_rate = min(0.35, self.state.exploration_rate * 1.45 + 0.025)
        self.state.candidate_diversity = min(2.25, self.state.candidate_diversity + 0.25)
        self.state.rollout_weight = min(0.72, self.state.rollout_weight + 0.06)
        self.state.uncertainty_penalty = max(0.08, self.state.uncertainty_penalty * 0.86)
        self.state.sigma_scale = min(1.75, self.state.sigma_scale * 1.18)
        self.state.learning_rate_scale = max(0.55, self.state.learning_rate_scale * 0.90)
        self.state.injected_plan_bias["multi_step_trap"] = self.state.injected_plan_bias.get("multi_step_trap", 0.0) + 0.04
        self.state.injected_plan_bias["resource_denial"] = self.state.injected_plan_bias.get("resource_denial", 0.0) + 0.04
        self.state.injected_plan_bias["endgame_finisher"] = self.state.injected_plan_bias.get("endgame_finisher", 0.0) + 0.05
        self.state.injected_plan_bias["staging_transfer"] = self.state.injected_plan_bias.get("staging_transfer", 0.0) + 0.04
        self.state.injected_plan_bias["defensive_consolidation"] = self.state.injected_plan_bias.get("defensive_consolidation", 0.0) + 0.03
        self.state.no_progress = max(0, self.window // 2)

    def observe_generalization_failure(self) -> AdaptationState:
        self.state.no_progress = self.window
        self.state.exploration_rate = min(0.45, self.state.exploration_rate * 1.75 + 0.04)
        self.state.candidate_diversity = min(2.50, self.state.candidate_diversity + 0.40)
        self.state.rollout_weight = min(0.80, self.state.rollout_weight + 0.10)
        self.state.uncertainty_penalty = min(0.50, self.state.uncertainty_penalty + 0.08)
        self.state.sigma_scale = min(2.00, self.state.sigma_scale * 1.30)
        self.state.learning_rate_scale = max(0.45, self.state.learning_rate_scale * 0.75)
        self.state.injected_plan_bias["balanced"] = self.state.injected_plan_bias.get("balanced", 0.0) + 0.06
        self.state.injected_plan_bias["defensive_consolidation"] = self.state.injected_plan_bias.get("defensive_consolidation", 0.0) + 0.10
        self.state.injected_plan_bias["staging_transfer"] = self.state.injected_plan_bias.get("staging_transfer", 0.0) + 0.12
        return self.state

    def _rebalance_mode_pressure(self, metrics: TrainingMetrics) -> None:
        if metrics.eval_4p + 0.08 < metrics.eval_2p:
            self.state.finisher_bias = min(1.65, self.state.finisher_bias + 0.08)
            self.state.candidate_diversity = min(2.0, self.state.candidate_diversity + 0.10)
            self.state.injected_plan_bias["staging_transfer"] = self.state.injected_plan_bias.get("staging_transfer", 0.0) + 0.03
            self.state.injected_plan_bias["endgame_finisher"] = self.state.injected_plan_bias.get("endgame_finisher", 0.0) + 0.03
        elif metrics.eval_2p + 0.08 < metrics.eval_4p:
            self.state.injected_plan_bias["aggressive_expansion"] = self.state.injected_plan_bias.get("aggressive_expansion", 0.0) + 0.025
            self.state.injected_plan_bias["resource_denial"] = self.state.injected_plan_bias.get("resource_denial", 0.0) + 0.025

    def planning_overrides(self) -> Dict[str, float]:
        return {
            "exploration_rate": self.state.exploration_rate,
            "candidate_diversity": self.state.candidate_diversity,
            "rollout_weight": self.state.rollout_weight,
            "uncertainty_penalty": self.state.uncertainty_penalty,
            "finisher_bias": self.state.finisher_bias,
        }
