"""Hybrid beam/rollout search for V9."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..agents.v9.evaluator import SimulationEstimate, V9Evaluator
from ..agents.v9.planner import PlanningParameters, V9Planner
from ..agents.v9.policy import V9Policy
from ..core.game import build_world
from ..features.plan_features import PlanCandidate


@dataclass
class SearchResult:
    chosen: PlanCandidate
    candidates: List[PlanCandidate]
    scores: Dict[str, float]
    estimates: Dict[str, SimulationEstimate]


def _diverse_top(candidates: List[PlanCandidate], width: int) -> List[PlanCandidate]:
    out: List[PlanCandidate] = []
    seen = set()
    for candidate in candidates:
        if candidate.plan_type in seen and len(out) < max(2, width // 2):
            continue
        out.append(candidate)
        seen.add(candidate.plan_type)
        if len(out) >= width:
            return out
    for candidate in candidates:
        if candidate not in out:
            out.append(candidate)
            if len(out) >= width:
                break
    return out


class HybridSearch:
    """Generate candidates, beam-filter by ranker, then rollout top plans."""

    def __init__(
        self,
        planner: Optional[V9Planner] = None,
        policy: Optional[V9Policy] = None,
        evaluator: Optional[V9Evaluator] = None,
        *,
        search_width: int = 7,
        exploration_rate: float = 0.08,
        rollout_weight: float = 0.42,
        uncertainty_penalty: float = 0.20,
        seed: int = 9009,
    ):
        self.planner = planner or V9Planner(PlanningParameters())
        self.policy = policy or V9Policy()
        self.evaluator = evaluator or V9Evaluator(seed=seed)
        self.search_width = int(search_width)
        self.exploration_rate = float(exploration_rate)
        self.rollout_weight = float(rollout_weight)
        self.uncertainty_penalty = float(uncertainty_penalty)
        self.rng = random.Random(seed)

    def decide(self, obs, *, injected_plan_bias: Optional[Dict[str, float]] = None) -> SearchResult:
        world = build_world(obs)
        candidates = self.planner.generate(world, self.rng)
        if not candidates:
            empty = PlanCandidate("empty", [], "reserve_hold")
            return SearchResult(empty, [empty], {"empty": 0.0}, {})
        raw = self.policy.score_candidates(
            world,
            candidates,
            rollout_weight=0.0,
            uncertainty_penalty=self.uncertainty_penalty,
            injected_plan_bias=injected_plan_bias,
        )
        raw.sort(key=lambda item: item[1], reverse=True)
        top = _diverse_top([c for c, _s, _f in raw], self.search_width)
        estimates = self.evaluator.evaluate(obs, top)
        chosen, final = self.policy.choose(
            world,
            candidates,
            estimates=estimates,
            exploration_rate=self.exploration_rate,
            rollout_weight=self.rollout_weight,
            uncertainty_penalty=self.uncertainty_penalty,
            injected_plan_bias=injected_plan_bias,
            rng=self.rng,
        )
        return SearchResult(chosen, candidates, {c.name: s for c, s, _f in final}, estimates)
