"""V10 4p-first policy surface built on the V9 runtime."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ...config.v10_config import V10Config
from ..v9.policy import V9Agent, V9Policy, V9Weights, get_weights as _v9_get, load_checkpoint as _v9_load, save_checkpoint as _v9_save, set_weights as _v9_set
from ...features.plan_features import PlanCandidate
from ..v9.evaluator import SimulationEstimate


DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[3] / "evaluations" / "v10_4p_policy.npz"


class V10Weights(V9Weights):
    @classmethod
    def defaults(cls) -> "V10Weights":
        base = V9Weights.defaults()
        plan_bias = base.plan_bias.copy()
        state_plan_w = base.state_plan_w.copy()
        plan_w = base.plan_w.copy()
        # Favor consolidation and transfers for 4p before tactical snipes.
        from ...features.plan_features import PLAN_FEATURE_NAMES, PLAN_TYPE_TO_INDEX
        from ...features.state_features import STATE_FEATURE_NAMES
        plan_feature_index = {name: i for i, name in enumerate(PLAN_FEATURE_NAMES)}
        state_feature_index = {name: i for i, name in enumerate(STATE_FEATURE_NAMES)}

        plan_bias[PLAN_TYPE_TO_INDEX["staging_transfer"]] = 0.11
        plan_bias[PLAN_TYPE_TO_INDEX["defensive_consolidation"]] = 0.26
        plan_bias[PLAN_TYPE_TO_INDEX["resource_denial"]] = -0.02
        plan_bias[PLAN_TYPE_TO_INDEX["delayed_strike"]] = -0.04
        plan_bias[PLAN_TYPE_TO_INDEX["aggressive_expansion"]] = 0.02
        plan_bias[PLAN_TYPE_TO_INDEX["opportunistic_snipe"]] = -0.09
        plan_bias[PLAN_TYPE_TO_INDEX["probe"]] = -0.10
        plan_bias[PLAN_TYPE_TO_INDEX["reserve_hold"]] = -0.01

        # 4p: when fronts/threat rise, push consolidation and reduce tactical spread.
        state_plan_w[PLAN_TYPE_TO_INDEX["defensive_consolidation"], state_feature_index["is_4p"]] += 0.28
        state_plan_w[PLAN_TYPE_TO_INDEX["defensive_consolidation"], state_feature_index["active_front_ratio"]] += 0.24
        state_plan_w[PLAN_TYPE_TO_INDEX["staging_transfer"], state_feature_index["active_front_ratio"]] -= 0.10
        state_plan_w[PLAN_TYPE_TO_INDEX["resource_denial"], state_feature_index["is_4p"]] -= 0.16
        state_plan_w[PLAN_TYPE_TO_INDEX["delayed_strike"], state_feature_index["is_4p"]] -= 0.14
        state_plan_w[PLAN_TYPE_TO_INDEX["opportunistic_snipe"], state_feature_index["is_4p"]] -= 0.24

        # Slightly prefer robust posture in 4p.
        plan_w[plan_feature_index["defense_move_frac"]] += 0.08
        plan_w[plan_feature_index["transfer_ship_frac"]] += 0.05
        plan_w[plan_feature_index["attack_move_frac"]] -= 0.06

        return cls(state_plan_w, plan_w, plan_bias, base.interaction_w.copy())


class V10Policy(V9Policy):
    """Named V10 policy wrapper used by validation and downstream imports."""

    def score_candidates(
        self,
        world,
        candidates: Iterable[PlanCandidate],
        *,
        estimates: Optional[Dict[str, SimulationEstimate]] = None,
        rollout_weight: float = 0.42,
        uncertainty_penalty: float = 0.20,
        injected_plan_bias: Optional[Dict[str, float]] = None,
        front_pressure_plan_bias: float = 0.12,
        front_pressure_attack_penalty: float = 0.12,
        four_p_front_budget: float = 2.7,
        front_open_penalty_weight: float = 0.10,
        front_close_bonus_weight: float = 0.08,
        front_overlap_penalty_weight: float = 0.08,
    ) -> List[Tuple[PlanCandidate, float, object]]:
        scored = super().score_candidates(
            world,
            candidates,
            estimates=estimates,
            rollout_weight=rollout_weight,
            uncertainty_penalty=uncertainty_penalty,
            injected_plan_bias=injected_plan_bias,
            front_pressure_plan_bias=front_pressure_plan_bias,
            front_pressure_attack_penalty=front_pressure_attack_penalty,
            four_p_front_budget=four_p_front_budget,
            front_open_penalty_weight=front_open_penalty_weight,
            front_close_bonus_weight=front_close_bonus_weight,
            front_overlap_penalty_weight=front_overlap_penalty_weight,
        )
        if not getattr(world, "is_four_player", False):
            return scored

        my_planets = len(getattr(world, "my_planets", []) or [])
        adjusted: List[Tuple[PlanCandidate, float, object]] = []
        for candidate, score, features in scored:
            metadata = candidate.metadata or {}
            backbone = float(metadata.get("backbone", 0.0))
            active_fronts = float(metadata.get("active_fronts", 0.0))
            front_budget = max(1.0, float(metadata.get("front_phase_budget", four_p_front_budget)))
            delta = 0.0
            if backbone > 0.0:
                delta += 0.18
                if 6 <= my_planets < 15 and not getattr(world, "is_late", False):
                    delta += 0.08
                if active_fronts <= front_budget + 0.50:
                    delta += 0.06
            if candidate.plan_type == "defensive_consolidation":
                delta += 0.10
            if active_fronts > front_budget + 0.50 and candidate.plan_type in ("resource_denial", "delayed_strike", "opportunistic_snipe", "aggressive_expansion"):
                delta -= 0.06
            adjusted.append((candidate, score + delta, features))
        return adjusted


class V10Agent(V9Agent):
    def __init__(self, config: Optional[V10Config] = None, weights: Optional[V10Weights] = None, *, injected_plan_bias: Optional[Dict[str, float]] = None):
        super().__init__(config=config or V10Config(), weights=weights or V10Weights.defaults(), injected_plan_bias=injected_plan_bias)
        self.policy = V10Policy(weights or V10Weights.defaults())


_GLOBAL_AGENT: Optional[V10Agent] = None


def get_weights() -> V10Weights:
    return V10Weights.from_flat(_v9_get().flatten())


def save_checkpoint(path: Optional[str] = None, weights: Optional[V10Weights] = None, meta: Optional[dict] = None) -> str:
    return _v9_save(str(Path(path) if path else DEFAULT_CHECKPOINT), weights or get_weights(), meta)


def load_checkpoint(path: Optional[str] = None) -> bool:
    ok = _v9_load(str(Path(path) if path else DEFAULT_CHECKPOINT))
    if not ok:
        return False
    return True


def agent(obs, config=None):
    global _GLOBAL_AGENT
    if _GLOBAL_AGENT is None:
        cfg = V10Config()
        _v9_set(get_weights())
        _GLOBAL_AGENT = V10Agent(cfg, get_weights())
    return _GLOBAL_AGENT(obs, config)


if os.environ.get("BOT_V10_NO_AUTOLOAD") != "1":
    if not load_checkpoint():
        _v9_set(V10Weights.defaults())
