"""Adaptive nonlinear ranking policy for V9."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ...config.v9_config import V9Config
from ...core.game import build_world
from ...features.plan_features import FEATURE_DIM as PLAN_DIM
from ...features.plan_features import PLAN_FEATURE_NAMES, PLAN_TYPE_TO_INDEX, PLAN_TYPES, PlanCandidate, extract_plan_features, match_move_target
from ...features.state_features import FEATURE_DIM as STATE_DIM
from ...features.state_features import STATE_FEATURE_NAMES, extract_state_features
from .evaluator import SimulationEstimate, V9Evaluator
from .planner import PlanningParameters, V9Planner


PLAN_FEATURE_INDEX = {name: i for i, name in enumerate(PLAN_FEATURE_NAMES)}
STATE_FEATURE_INDEX = {name: i for i, name in enumerate(STATE_FEATURE_NAMES)}

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[3] / "evaluations" / "v9_policy.npz"


@dataclass
class V9Weights:
    state_plan_w: np.ndarray
    plan_w: np.ndarray
    plan_bias: np.ndarray
    interaction_w: np.ndarray

    @classmethod
    def defaults(cls) -> "V9Weights":
        state_plan_w = np.zeros((len(PLAN_TYPES), STATE_DIM), dtype=np.float32)
        plan_w = np.zeros(PLAN_DIM, dtype=np.float32)
        plan_bias = np.zeros(len(PLAN_TYPES), dtype=np.float32)
        interaction_w = np.array([0.55, 0.44, 0.50, 0.35, -0.62, -0.35, 0.38, 0.30], dtype=np.float32)

        def pw(name: str, value: float) -> None:
            plan_w[PLAN_FEATURE_INDEX[name]] = value

        pw("move_count", 0.04)
        pw("ship_commitment", 0.20)
        pw("attack_move_frac", 0.14)
        pw("expand_move_frac", 0.12)
        pw("defense_move_frac", 0.06)
        pw("transfer_ship_frac", 0.24)
        pw("avg_eta", -0.08)
        pw("target_prod_gain", 0.28)
        pw("target_ship_cost", -0.08)
        pw("distinct_target_frac", 0.10)
        pw("overcommit_risk", -0.70)
        pw("undercommit_risk", -0.35)
        pw("finisher_flag", 0.10)
        pw("denial_flag", 0.08)
        pw("trap_flag", 0.07)
        pw("probe_flag", -0.03)
        pw("frontier_pressure", -0.08)
        pw("garrison_after_commit", 0.24)
        pw("weak_enemy_focus", 0.20)
        pw("high_prod_focus", 0.16)
        pw("neutral_focus", 0.08)
        pw("enemy_focus", 0.10)
        pw("base_score", 0.055)

        plan_bias[PLAN_TYPE_TO_INDEX["balanced"]] = 0.05
        plan_bias[PLAN_TYPE_TO_INDEX["aggressive_expansion"]] = 0.12
        plan_bias[PLAN_TYPE_TO_INDEX["delayed_strike"]] = 0.08
        plan_bias[PLAN_TYPE_TO_INDEX["multi_step_trap"]] = 0.06
        plan_bias[PLAN_TYPE_TO_INDEX["resource_denial"]] = 0.11
        plan_bias[PLAN_TYPE_TO_INDEX["endgame_finisher"]] = 0.04
        plan_bias[PLAN_TYPE_TO_INDEX["defensive_consolidation"]] = 0.09
        plan_bias[PLAN_TYPE_TO_INDEX["staging_transfer"]] = 0.20
        plan_bias[PLAN_TYPE_TO_INDEX["opportunistic_snipe"]] = 0.08
        plan_bias[PLAN_TYPE_TO_INDEX["probe"]] = -0.02
        plan_bias[PLAN_TYPE_TO_INDEX["reserve_hold"]] = -0.10

        state_plan_w[PLAN_TYPE_TO_INDEX["aggressive_expansion"], STATE_FEATURE_INDEX["is_opening"]] = 0.22
        state_plan_w[PLAN_TYPE_TO_INDEX["aggressive_expansion"], STATE_FEATURE_INDEX["neutral_softness"]] = 0.22
        state_plan_w[PLAN_TYPE_TO_INDEX["resource_denial"], STATE_FEATURE_INDEX["enemy_prod_share"]] = 0.18
        state_plan_w[PLAN_TYPE_TO_INDEX["resource_denial"], STATE_FEATURE_INDEX["prod_lead_vs_strongest"]] = 0.12
        state_plan_w[PLAN_TYPE_TO_INDEX["endgame_finisher"], STATE_FEATURE_INDEX["finish_pressure"]] = 0.42
        state_plan_w[PLAN_TYPE_TO_INDEX["endgame_finisher"], STATE_FEATURE_INDEX["is_late"]] = 0.22
        state_plan_w[PLAN_TYPE_TO_INDEX["delayed_strike"], STATE_FEATURE_INDEX["is_4p"]] = 0.15
        state_plan_w[PLAN_TYPE_TO_INDEX["staging_transfer"], STATE_FEATURE_INDEX["is_4p"]] = 0.48
        state_plan_w[PLAN_TYPE_TO_INDEX["staging_transfer"], STATE_FEATURE_INDEX["active_front_ratio"]] = 0.24
        state_plan_w[PLAN_TYPE_TO_INDEX["multi_step_trap"], STATE_FEATURE_INDEX["active_front_ratio"]] = 0.18
        state_plan_w[PLAN_TYPE_TO_INDEX["defensive_consolidation"], STATE_FEATURE_INDEX["threatened_ratio"]] = 0.42
        state_plan_w[PLAN_TYPE_TO_INDEX["defensive_consolidation"], STATE_FEATURE_INDEX["doomed_ratio"]] = 0.26
        state_plan_w[PLAN_TYPE_TO_INDEX["defensive_consolidation"], STATE_FEATURE_INDEX["active_front_ratio"]] = 0.20
        state_plan_w[PLAN_TYPE_TO_INDEX["reserve_hold"], STATE_FEATURE_INDEX["comeback_pressure"]] = 0.10
        return cls(state_plan_w, plan_w, plan_bias, interaction_w)

    def clone(self) -> "V9Weights":
        return V9Weights(
            self.state_plan_w.copy(),
            self.plan_w.copy(),
            self.plan_bias.copy(),
            self.interaction_w.copy(),
        )

    def flatten(self) -> np.ndarray:
        return np.concatenate([
            self.state_plan_w.ravel(),
            self.plan_w.ravel(),
            self.plan_bias.ravel(),
            self.interaction_w.ravel(),
        ]).astype(np.float32)

    @classmethod
    def from_flat(cls, flat: Sequence[float]) -> "V9Weights":
        vec = np.asarray(flat, dtype=np.float32).ravel()
        n_state = len(PLAN_TYPES) * STATE_DIM
        n_plan = PLAN_DIM
        n_bias = len(PLAN_TYPES)
        n_inter = 8
        expected = n_state + n_plan + n_bias + n_inter
        if vec.size != expected:
            raise ValueError(f"V9 weight vector size {vec.size} != {expected}")
        off = 0
        state_plan_w = vec[off: off + n_state].reshape(len(PLAN_TYPES), STATE_DIM)
        off += n_state
        plan_w = vec[off: off + n_plan]
        off += n_plan
        plan_bias = vec[off: off + n_bias]
        off += n_bias
        interaction_w = vec[off: off + n_inter]
        return cls(state_plan_w.copy(), plan_w.copy(), plan_bias.copy(), interaction_w.copy())


def _sim_score(estimate: Optional[SimulationEstimate], rollout_weight: float, uncertainty_penalty: float) -> float:
    if estimate is None or estimate.rollouts <= 0:
        return 0.0
    useful = (
        estimate.mean_delta
        + 0.35 * estimate.worst_delta
        + 0.58 * estimate.margin_delta
        + 0.12 * estimate.production_delta
        + 0.08 * estimate.planet_delta
        + 1.25 * estimate.finish_bonus
    )
    return rollout_weight * useful - uncertainty_penalty * estimate.uncertainty


class V9Policy:
    """Nonlinear ranker plus hybrid filtering."""

    def __init__(self, weights: Optional[V9Weights] = None):
        self.weights = weights.clone() if weights is not None else V9Weights.defaults()

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
    ) -> List[Tuple[PlanCandidate, float, np.ndarray]]:
        state_feat = extract_state_features(world)
        scored: List[Tuple[PlanCandidate, float, np.ndarray]] = []
        injected_plan_bias = injected_plan_bias or {}
        for candidate in candidates:
            plan_feat = extract_plan_features(candidate, world)
            pidx = PLAN_TYPE_TO_INDEX.get(candidate.plan_type, 0)
            linear = float(self.weights.state_plan_w[pidx] @ state_feat)
            linear += float(self.weights.plan_w @ plan_feat)
            linear += float(self.weights.plan_bias[pidx])
            linear += float(injected_plan_bias.get(candidate.plan_type, 0.0))

            opening = state_feat[STATE_FEATURE_INDEX["is_opening"]]
            late = max(state_feat[STATE_FEATURE_INDEX["is_late"]], state_feat[STATE_FEATURE_INDEX["is_very_late"]])
            finish_pressure = state_feat[STATE_FEATURE_INDEX["finish_pressure"]]
            comeback = state_feat[STATE_FEATURE_INDEX["comeback_pressure"]]
            threatened = state_feat[STATE_FEATURE_INDEX["threatened_ratio"]]
            fronts = state_feat[STATE_FEATURE_INDEX["active_front_ratio"]]
            four_p = state_feat[STATE_FEATURE_INDEX["is_4p"]]

            attack = plan_feat[PLAN_FEATURE_INDEX["attack_move_frac"]]
            expand = plan_feat[PLAN_FEATURE_INDEX["expand_move_frac"]]
            defense = plan_feat[PLAN_FEATURE_INDEX["defense_move_frac"]]
            transfer = plan_feat[PLAN_FEATURE_INDEX["transfer_ship_frac"]]
            overcommit = plan_feat[PLAN_FEATURE_INDEX["overcommit_risk"]]
            undercommit = plan_feat[PLAN_FEATURE_INDEX["undercommit_risk"]]
            weak_focus = plan_feat[PLAN_FEATURE_INDEX["weak_enemy_focus"]]
            high_prod = plan_feat[PLAN_FEATURE_INDEX["high_prod_focus"]]

            nonlinear = 0.0
            iw = self.weights.interaction_w
            nonlinear += float(iw[0]) * math.tanh(float(finish_pressure * attack + late * weak_focus))
            nonlinear += float(iw[1]) * math.tanh(float(opening * expand + plan_feat[PLAN_FEATURE_INDEX["neutral_focus"]]))
            nonlinear += float(iw[2]) * math.tanh(float(four_p * (transfer + high_prod)))
            nonlinear += float(iw[3]) * math.tanh(float((threatened + fronts) * defense))
            nonlinear += float(iw[4]) * float(overcommit) * (1.0 + float(threatened + fronts))
            nonlinear += float(iw[5]) * float(undercommit) * (1.0 + float(late))
            nonlinear += float(iw[6]) * math.tanh(float(comeback * (defense + expand)))
            nonlinear += float(iw[7]) * math.tanh(float(high_prod + weak_focus))

            safety = 0.0
            metadata_bonus = 0.0
            if four_p > 0.5:
                metadata = candidate.metadata or {}
                backbone = float(metadata.get("backbone", 0.0))
                front_lock = float(metadata.get("front_lock", 0.0))
                consolidation_threshold = float(metadata.get("consolidation_threshold", 0.0))
                staged_finisher = float(metadata.get("staged_finisher", 0.0))
                metadata_bonus += 0.36 * backbone
                metadata_bonus += (0.12 + 0.14 * float(fronts)) * front_lock
                metadata_bonus += 0.20 * consolidation_threshold
                if candidate.plan_type == "staging_transfer":
                    metadata_bonus += 0.10 + 0.16 * backbone + 0.08 * front_lock
                if backbone > 0.0 and transfer >= 0.30 and attack < 0.35:
                    metadata_bonus += 0.08
                metadata_bonus += (0.12 + 0.30 * float(finish_pressure)) * staged_finisher
                if 6 <= len(world.my_planets) < 15 and not world.is_late:
                    metadata_bonus += 0.18 * float(transfer + defense)
                    if attack > 0.45 and transfer < 0.12:
                        metadata_bonus -= 0.12
                if fronts > 0.36 and not world.is_late:
                    front_pressure = float((fronts - 0.36) / 0.64)
                    if candidate.plan_type in ("defensive_consolidation", "staging_transfer", "reserve_hold"):
                        metadata_bonus += float(front_pressure_plan_bias) * (0.75 + front_pressure)
                    if backbone > 0.0 or front_lock > 0.0 or consolidation_threshold > 0.0:
                        metadata_bonus += 0.08 * front_pressure
                    finisher_ready = candidate.plan_type == "endgame_finisher" and (staged_finisher > 0.0 or finish_pressure > 1.0)
                    if not finisher_ready and candidate.plan_type in (
                        "resource_denial",
                        "delayed_strike",
                        "multi_step_trap",
                        "opportunistic_snipe",
                        "aggressive_expansion",
                    ):
                        metadata_bonus -= float(front_pressure_attack_penalty) * (0.60 + front_pressure)
                    if attack > 0.35 and transfer < 0.18 and not finisher_ready:
                        metadata_bonus -= 0.10 + 0.10 * front_pressure
                if fronts > 0.42 and candidate.plan_type not in ("defensive_consolidation", "staging_transfer", "reserve_hold"):
                    metadata_bonus -= 0.10 * float(fronts)
                focus = metadata.get("focus_enemy_id")
                if focus is not None and world.weakest_enemy_id is not None and int(focus) == int(world.weakest_enemy_id):
                    metadata_bonus += 0.05
            if candidate.plan_type == "reserve_hold" and not world.threatened_candidates and not world.doomed_candidates:
                safety -= 0.22
            if candidate.plan_type == "probe" and (world.is_late or finish_pressure > 0.8):
                safety -= 0.20
            if not candidate.moves and candidate.plan_type != "reserve_hold":
                safety -= 0.55

            estimate = estimates.get(candidate.name) if estimates else None
            score = linear + nonlinear + safety + metadata_bonus + _sim_score(estimate, rollout_weight, uncertainty_penalty)
            scored.append((candidate, float(score), plan_feat))
        return scored

    def choose(
        self,
        world,
        candidates: List[PlanCandidate],
        *,
        estimates: Optional[Dict[str, SimulationEstimate]] = None,
        exploration_rate: float = 0.0,
        rollout_weight: float = 0.42,
        uncertainty_penalty: float = 0.20,
        injected_plan_bias: Optional[Dict[str, float]] = None,
        front_pressure_plan_bias: float = 0.12,
        front_pressure_attack_penalty: float = 0.12,
        rng: Optional[random.Random] = None,
    ) -> Tuple[PlanCandidate, List[Tuple[PlanCandidate, float, np.ndarray]]]:
        scored = self.score_candidates(
            world,
            candidates,
            estimates=estimates,
            rollout_weight=rollout_weight,
            uncertainty_penalty=uncertainty_penalty,
            injected_plan_bias=injected_plan_bias,
            front_pressure_plan_bias=front_pressure_plan_bias,
            front_pressure_attack_penalty=front_pressure_attack_penalty,
        )
        if not scored:
            return PlanCandidate("empty", [], "reserve_hold"), []
        scored.sort(key=lambda item: item[1], reverse=True)
        if scored[0][0].moves:
            best = scored[0][0]
        else:
            best = next((c for c, _s, _f in scored if c.moves), scored[0][0])
        rng = rng or random.Random()
        if exploration_rate > 0 and len(scored) > 1 and rng.random() < exploration_rate:
            pool = scored[: min(5, len(scored))]
            vals = np.array([s for _c, s, _f in pool], dtype=np.float32)
            vals = vals - float(np.max(vals))
            probs = np.exp(vals / 0.28)
            probs = probs / max(1e-8, float(np.sum(probs)))
            pick = int(rng.choices(range(len(pool)), weights=probs.tolist(), k=1)[0])
            best = pool[pick][0]
        return best, scored


def _dist_planets(a, b) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def _select_focus_enemy(world, current: Optional[int]) -> Optional[int]:
    if current is not None and any(int(p.owner) == int(current) for p in world.enemy_planets):
        return int(current)
    owners = [
        int(o)
        for o in world.owner_strength
        if o not in (-1, world.player) and any(int(p.owner) == int(o) for p in world.enemy_planets)
    ]
    if not owners:
        return None
    preferred = getattr(world, "weakest_enemy_id", None)
    if preferred is not None and int(preferred) in owners:
        return int(preferred)
    return min(owners, key=lambda o: float(world.owner_strength.get(o, 0)) + 18.0 * float(world.owner_production.get(o, 0)))


def _select_front_anchor(world, focus_enemy_id: Optional[int], current: Optional[int]) -> Optional[int]:
    if current is not None:
        planet = world.planet_by_id.get(int(current))
        if planet is not None and planet.owner == world.player and planet.id not in world.doomed_candidates:
            return int(planet.id)
    if not world.my_planets:
        return None
    targets = [p for p in world.enemy_planets if focus_enemy_id is None or int(p.owner) == int(focus_enemy_id)]
    if not targets:
        targets = list(world.enemy_planets or world.neutral_planets)
    if not targets:
        return int(max(world.my_planets, key=lambda p: p.ships).id)
    safe = [p for p in world.my_planets if p.id not in world.doomed_candidates] or list(world.my_planets)
    anchor = min(
        safe,
        key=lambda p: (
            min(_dist_planets(p, t) for t in targets),
            p.id in world.threatened_candidates,
            -float(p.production),
        ),
    )
    return int(anchor.id)


def _active_front_count_runtime(world, focus_enemy_id: Optional[int]) -> int:
    enemies = [p for p in world.enemy_planets if focus_enemy_id is None or int(p.owner) == int(focus_enemy_id)]
    if not enemies:
        enemies = list(world.enemy_planets)
    count = 0
    for mine in world.my_planets:
        if any(_dist_planets(mine, enemy) <= 36.0 + float(mine.radius) + float(enemy.radius) for enemy in enemies):
            count += 1
    return count


class V9Agent:
    """Callable Kaggle-compatible V9 agent."""

    def __init__(
        self,
        config: Optional[V9Config] = None,
        weights: Optional[V9Weights] = None,
        *,
        injected_plan_bias: Optional[Dict[str, float]] = None,
    ):
        self.config = config or V9Config()
        self.policy = V9Policy(weights)
        self.injected_plan_bias = dict(injected_plan_bias or {})
        self.rng = random.Random(self.config.seed)
        self.plan_history: List[str] = []
        self.plan_stats: Dict[str, float] = {}
        self.front_lock_owner: Optional[int] = None
        self.front_lock_anchor: Optional[int] = None
        self.front_lock_until: int = -1
        self.front_lock_switches: int = 0
        self.last_step: int = -1

    def __call__(self, obs, config=None):
        try:
            return self.act(obs)
        except Exception:
            return []

    def act(self, obs) -> List[List]:
        world = build_world(obs)
        if not world.my_planets:
            return []
        self._reset_if_new_game(world)
        self._update_front_lock(world)
        planner = V9Planner(PlanningParameters(
            max_candidates=self.config.max_candidates,
            candidate_diversity=self.config.candidate_diversity,
            finisher_bias=self.config.finisher_bias,
            min_source_ships=self.config.min_source_ships,
            max_moves_per_plan=self.config.max_moves_per_plan,
            focus_enemy_id=self.front_lock_owner,
            front_anchor_id=self.front_lock_anchor,
        ))
        candidates = planner.generate(world, self.rng)
        if not candidates:
            return []

        raw_scored = self.policy.score_candidates(
            world,
            candidates,
            rollout_weight=0.0,
            uncertainty_penalty=self.config.uncertainty_penalty,
            injected_plan_bias=self.injected_plan_bias,
            front_pressure_plan_bias=self.config.front_pressure_plan_bias,
            front_pressure_attack_penalty=self.config.front_pressure_attack_penalty,
        )
        raw_scored.sort(key=lambda item: item[1], reverse=True)
        top = _diverse_top([c for c, _s, _f in raw_scored], self.config.search_width)
        evaluator = V9Evaluator(
            depth=self.config.simulation_depth,
            rollouts=self.config.simulation_rollouts,
            opponent_samples=self.config.opponent_samples,
            seed=self.config.seed + world.step,
        )
        estimates = evaluator.evaluate(obs, top)
        best, _ = self.policy.choose(
            world,
            candidates,
            estimates=estimates,
            exploration_rate=self.config.exploration_rate,
            rollout_weight=self.config.rollout_weight,
            uncertainty_penalty=self.config.uncertainty_penalty,
            injected_plan_bias=self.injected_plan_bias,
            front_pressure_plan_bias=self.config.front_pressure_plan_bias,
            front_pressure_attack_penalty=self.config.front_pressure_attack_penalty,
            rng=self.rng,
        )
        self.plan_history.append(best.plan_type)
        self._record_plan_diagnostics(world, best)
        self.last_step = int(world.step)
        return best.moves

    def _reset_if_new_game(self, world) -> None:
        step = int(world.step)
        if self.last_step >= 0 and step < self.last_step:
            self.plan_history.clear()
            self.plan_stats.clear()
            self.front_lock_owner = None
            self.front_lock_anchor = None
            self.front_lock_until = -1
            self.front_lock_switches = 0

    def _update_front_lock(self, world) -> None:
        if not world.is_four_player or not world.enemy_planets:
            self.front_lock_owner = None
            self.front_lock_anchor = None
            self.front_lock_until = -1
            return
        step = int(world.step)
        keep_lock = (
            self.front_lock_owner is not None
            and step <= self.front_lock_until
            and any(int(p.owner) == int(self.front_lock_owner) for p in world.enemy_planets)
        )
        focus = _select_focus_enemy(world, self.front_lock_owner if keep_lock else None)
        if focus is None:
            self.front_lock_owner = None
            self.front_lock_anchor = None
            self.front_lock_until = -1
            return
        if self.front_lock_owner is not None and int(focus) != int(self.front_lock_owner):
            self.front_lock_switches += 1
        self.front_lock_owner = int(focus)
        self.front_lock_anchor = _select_front_anchor(world, self.front_lock_owner, self.front_lock_anchor if keep_lock else None)
        if not keep_lock:
            self.front_lock_until = step + int(getattr(self.config, "front_lock_turns", 24))

    def _record_plan_diagnostics(self, world, candidate: PlanCandidate) -> None:
        stats = self.plan_stats
        stats["turns"] = stats.get("turns", 0.0) + 1.0
        if world.is_four_player:
            stats["four_p_turns"] = stats.get("four_p_turns", 0.0) + 1.0
            stats["front_lock_turns"] = stats.get("front_lock_turns", 0.0) + (1.0 if self.front_lock_owner is not None else 0.0)
            stats["active_front_sum"] = stats.get("active_front_sum", 0.0) + float(_active_front_count_runtime(world, self.front_lock_owner))
        metadata = candidate.metadata or {}
        if metadata.get("backbone", 0.0):
            stats["backbone_turns"] = stats.get("backbone_turns", 0.0) + 1.0
        if metadata.get("staged_finisher", 0.0):
            stats["staged_finisher_turns"] = stats.get("staged_finisher_turns", 0.0) + 1.0
        if metadata.get("consolidation_threshold", 0.0):
            stats["consolidation_threshold_turns"] = stats.get("consolidation_threshold_turns", 0.0) + 1.0
        stats["focus_switches"] = float(self.front_lock_switches)

        for move in candidate.moves or []:
            if len(move) != 3:
                continue
            stats["total_moves"] = stats.get("total_moves", 0.0) + 1.0
            ships = max(0, int(move[2]))
            stats["total_ships_sent"] = stats.get("total_ships_sent", 0.0) + float(ships)
            match = match_move_target(move, world)
            if match is None:
                continue
            target, _eta = match
            if target.owner == world.player:
                stats["transfer_moves"] = stats.get("transfer_moves", 0.0) + 1.0
                stats["transfer_ships"] = stats.get("transfer_ships", 0.0) + float(ships)
            elif target.owner == -1:
                stats["expand_moves"] = stats.get("expand_moves", 0.0) + 1.0
            else:
                stats["attack_moves"] = stats.get("attack_moves", 0.0) + 1.0


def _diverse_top(candidates: Sequence[PlanCandidate], width: int) -> List[PlanCandidate]:
    out: List[PlanCandidate] = []
    seen_types = set()
    for candidate in candidates:
        if candidate.plan_type in seen_types and len(out) < max(2, width // 2):
            continue
        out.append(candidate)
        seen_types.add(candidate.plan_type)
        if len(out) >= width:
            return out
    for candidate in candidates:
        if candidate not in out:
            out.append(candidate)
            if len(out) >= width:
                break
    return out


_GLOBAL_WEIGHTS = V9Weights.defaults()
_GLOBAL_AGENT: Optional[V9Agent] = None


def set_weights(weights: V9Weights) -> None:
    global _GLOBAL_WEIGHTS, _GLOBAL_AGENT
    _GLOBAL_WEIGHTS = weights.clone()
    if _GLOBAL_AGENT is not None:
        _GLOBAL_AGENT.policy = V9Policy(_GLOBAL_WEIGHTS)


def get_weights() -> V9Weights:
    return _GLOBAL_WEIGHTS.clone()


def save_checkpoint(path: Optional[str] = None, weights: Optional[V9Weights] = None, meta: Optional[dict] = None) -> str:
    p = Path(path) if path else DEFAULT_CHECKPOINT
    p.parent.mkdir(parents=True, exist_ok=True)
    w = weights.clone() if weights is not None else _GLOBAL_WEIGHTS.clone()
    payload = {
        "flat": w.flatten(),
        "state_plan_w": w.state_plan_w,
        "plan_w": w.plan_w,
        "plan_bias": w.plan_bias,
        "interaction_w": w.interaction_w,
        "meta_json": np.asarray(json.dumps(meta or {}, sort_keys=True, default=str)),
    }
    np.savez_compressed(str(p), **payload)
    return str(p)


def load_checkpoint(path: Optional[str] = None) -> bool:
    p = Path(path) if path else DEFAULT_CHECKPOINT
    if not p.exists():
        return False
    try:
        data = np.load(str(p), allow_pickle=False)
        if "flat" in data:
            weights = V9Weights.from_flat(data["flat"])
        else:
            weights = V9Weights(
                np.asarray(data["state_plan_w"], dtype=np.float32),
                np.asarray(data["plan_w"], dtype=np.float32),
                np.asarray(data["plan_bias"], dtype=np.float32),
                np.asarray(data["interaction_w"], dtype=np.float32),
            )
        set_weights(weights)
        return True
    except Exception:
        return False


def agent(obs, config=None):
    global _GLOBAL_AGENT
    if _GLOBAL_AGENT is None:
        cfg = V9Config()
        _GLOBAL_AGENT = V9Agent(cfg, _GLOBAL_WEIGHTS)
    return _GLOBAL_AGENT(obs, config)


if os.environ.get("BOT_V9_NO_AUTOLOAD") != "1":
    load_checkpoint()


__all__ = [
    "V9Agent",
    "V9Policy",
    "V9Weights",
    "agent",
    "get_weights",
    "set_weights",
    "save_checkpoint",
    "load_checkpoint",
]
