"""Orbit Wars V8 core.

This module keeps the learned part at the plan level:
- generate a small set of full candidate plans,
- featurize each plan against the current state,
- rank plans with a linear model,
- train with DAgger-style aggregated labels.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import bot_v7


MODEL_PATH = os.path.join("evaluations", "v8_policy.npz")

PLAN_STYLES = ["v7", "attack", "expand", "defense", "comet", "reserve", "all_in", "split", "hold"]
STYLE_TO_ID = {name: idx for idx, name in enumerate(PLAN_STYLES)}
N_STYLES = len(PLAN_STYLES)

STATE_FEATURE_DIM = 33
# Plan features = action stats (32) + style one-hot (N_STYLES). The one-hot is
# how the ranker tells "attack" from "expand" when both produce the same actions.
PLAN_FEATURE_DIM = 32 + N_STYLES
INTERACTION_DIM = 8
INPUT_DIM = STATE_FEATURE_DIM + PLAN_FEATURE_DIM + INTERACTION_DIM


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_max(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(max(values))


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    if logits.size == 0:
        return logits
    shifted = logits - float(np.max(logits))
    exps = np.exp(shifted)
    total = float(np.sum(exps))
    if total <= 1e-12:
        return np.full_like(logits, 1.0 / max(len(logits), 1), dtype=np.float32)
    return (exps / total).astype(np.float32)


def _plan_distance(src, tgt) -> float:
    return math.hypot(float(tgt.x) - float(src.x), float(tgt.y) - float(src.y))


def _is_enemy(owner: int, player: int) -> bool:
    return owner not in (-1, player)


def _domination(world) -> float:
    return (float(world.my_total) - float(world.max_enemy_strength)) / max(
        1.0, float(world.my_total + world.max_enemy_strength)
    )


def _action_angle_to_target(src, tgt) -> float:
    return math.atan2(float(tgt.y) - float(src.y), float(tgt.x) - float(src.x))


def _infer_target(world, action) -> Optional[object]:
    """Infer the target planet from an action ray.

    The action format does not carry the target id, so we use the current ray
    direction and pick the planet most aligned with it.
    """
    if action is None or len(action) != 3:
        return None
    src_id = int(action[0])
    angle = float(action[1])
    src = world.planet_by_id.get(src_id)
    if src is None:
        return None
    dir_x = math.cos(angle)
    dir_y = math.sin(angle)

    best = None
    best_score = -1.0e18
    for tgt in world.planets:
        if tgt.id == src_id:
            continue
        vx = float(tgt.x) - float(src.x)
        vy = float(tgt.y) - float(src.y)
        dist = math.hypot(vx, vy)
        if dist <= 1e-9:
            continue
        dot = vx * dir_x + vy * dir_y
        if dot <= 0.0:
            continue
        align = dot / dist
        score = align - 0.008 * dist
        if score > best_score:
            best_score = score
            best = tgt
    return best


def _normalized_entropy(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    total = float(sum(values))
    if total <= 1e-9:
        return 0.0
    probs = [v / total for v in values if v > 0]
    if not probs:
        return 0.0
    ent = -sum(p * math.log(p + 1e-12) for p in probs)
    return float(ent / math.log(max(len(values), 2)))


@dataclass
class CandidatePlan:
    name: str
    actions: List[List[float]]
    features: np.ndarray = field(default_factory=lambda: np.zeros(INPUT_DIM, dtype=np.float32))
    stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingExample:
    state_features: np.ndarray
    candidate_features: np.ndarray
    label: int
    value_target: float


class LinearV8Model:
    """Plan-level linear ranker with a value head."""

    def __init__(self, score_w=None, score_b=0.0, value_w=None, value_b=0.0):
        self.score_w = np.zeros(INPUT_DIM, dtype=np.float32) if score_w is None else np.asarray(score_w, dtype=np.float32)
        self.score_b = float(score_b)
        self.value_w = np.zeros(STATE_FEATURE_DIM, dtype=np.float32) if value_w is None else np.asarray(value_w, dtype=np.float32)
        self.value_b = float(value_b)

    @classmethod
    def zero(cls) -> "LinearV8Model":
        return cls()

    def copy(self) -> "LinearV8Model":
        return LinearV8Model(
            score_w=self.score_w.copy(),
            score_b=self.score_b,
            value_w=self.value_w.copy(),
            value_b=self.value_b,
        )

    def score_plan_features(self, candidate_features: np.ndarray) -> float:
        return float(np.dot(self.score_w, candidate_features) + self.score_b)

    def score_candidates(self, candidate_features: np.ndarray) -> np.ndarray:
        if len(candidate_features) == 0:
            return np.zeros(0, dtype=np.float32)
        return (candidate_features @ self.score_w + self.score_b).astype(np.float32)

    def value_state(self, state_features: np.ndarray) -> float:
        return float(np.dot(self.value_w, state_features) + self.value_b)

    def select_index(self, candidate_features: np.ndarray) -> int:
        if len(candidate_features) == 0:
            return 0
        scores = self.score_candidates(candidate_features)
        return int(np.argmax(scores))

    def train_batch(
        self,
        batch: Sequence[TrainingExample],
        lr: float = 0.05,
        value_lr: float = 0.03,
        l2: float = 1e-4,
        rank_margin: float = 0.15,
        rank_weight: float = 0.75,
        hard_negatives: int = 3,
        clip_abs: float = 4.0,
    ) -> Dict[str, float]:
        if not batch:
            return {"loss": 0.0, "acc": 0.0, "value_loss": 0.0, "rank_loss": 0.0}

        grad_score_w = np.zeros_like(self.score_w)
        grad_score_b = 0.0
        grad_value_w = np.zeros_like(self.value_w)
        grad_value_b = 0.0

        loss = 0.0
        value_loss = 0.0
        rank_loss = 0.0
        acc = 0.0

        for ex in batch:
            X = np.asarray(ex.candidate_features, dtype=np.float32)
            if X.ndim != 2 or len(X) == 0:
                continue
            y = int(max(0, min(ex.label, len(X) - 1)))
            logits = X @ self.score_w + self.score_b
            probs = _softmax(logits)
            loss -= float(math.log(max(float(probs[y]), 1e-12)))
            grad = probs.copy()
            grad[y] -= 1.0

            if len(logits) > 1:
                neg_idx = [i for i in range(len(logits)) if i != y]
                if neg_idx:
                    neg_logits = logits[neg_idx]
                    k = min(int(max(1, hard_negatives)), len(neg_idx))
                    if k < len(neg_idx):
                        hard_local = np.argpartition(neg_logits, -k)[-k:]
                        chosen_idx = [neg_idx[i] for i in hard_local]
                    else:
                        chosen_idx = neg_idx
                    pos_logit = float(logits[y])
                    for neg_i in chosen_idx:
                        gap = float(logits[neg_i] - pos_logit + rank_margin)
                        if gap <= -10.0:
                            continue
                        # Smooth pairwise hinge on the hardest competing plans.
                        if gap >= 0.0:
                            exp_neg = math.exp(-gap)
                            sig = 1.0 / (1.0 + exp_neg)
                        else:
                            exp_gap = math.exp(gap)
                            sig = exp_gap / (1.0 + exp_gap)
                        rank_loss += float(np.logaddexp(0.0, gap))
                        grad[y] -= rank_weight * sig
                        grad[neg_i] += rank_weight * sig

            grad_score_w += X.T @ grad
            grad_score_b += float(np.sum(grad))
            acc += float(int(np.argmax(probs)) == y)

            state = np.asarray(ex.state_features, dtype=np.float32)
            v_pred = self.value_state(state)
            v_err = v_pred - float(ex.value_target)
            value_loss += float(v_err * v_err)
            grad_value_w += 2.0 * v_err * state
            grad_value_b += 2.0 * v_err

        n = float(max(len(batch), 1))
        if lr != 0.0:
            self.score_w -= lr * (grad_score_w / n + l2 * self.score_w)
            self.score_b -= lr * (grad_score_b / n)
            np.clip(self.score_w, -clip_abs, clip_abs, out=self.score_w)
        if value_lr != 0.0:
            self.value_w -= value_lr * (grad_value_w / n + l2 * self.value_w)
            self.value_b -= value_lr * (grad_value_b / n)
            np.clip(self.value_w, -clip_abs, clip_abs, out=self.value_w)

        return {
            "loss": float(loss / n),
            "acc": float(acc / n),
            "value_loss": float(value_loss / n),
            "rank_loss": float(rank_loss / n),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            score_w=self.score_w.astype(np.float32),
            score_b=np.array([self.score_b], dtype=np.float32),
            value_w=self.value_w.astype(np.float32),
            value_b=np.array([self.value_b], dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str) -> "LinearV8Model":
        data = np.load(path, allow_pickle=False)
        return cls(
            score_w=data["score_w"],
            score_b=float(np.asarray(data["score_b"]).reshape(-1)[0]),
            value_w=data["value_w"],
            value_b=float(np.asarray(data["value_b"]).reshape(-1)[0]),
        )


def build_state_features(world) -> np.ndarray:
    planets = list(world.planets)
    my_planets = list(world.my_planets)
    enemy_planets = list(world.enemy_planets)
    neutral_planets = list(world.neutral_planets)
    total_planets = max(1, len(planets))
    total_strength = max(1.0, float(world.my_total + world.enemy_total))
    total_prod = max(1.0, float(world.my_prod + world.enemy_prod))

    reserve_sum = float(sum(world.reserve.values())) if getattr(world, "reserve", None) else 0.0
    avail_sum = float(sum(world.available.values())) if getattr(world, "available", None) else 0.0

    threat_deficit = 0.0
    threat_turns = []
    for info in world.threatened_candidates.values():
        threat_deficit += float(info.get("deficit_hint", 0))
        if info.get("fall_turn") is not None:
            threat_turns.append(float(info["fall_turn"]))

    reaction_gaps = []
    neutral_gaps = []
    for p in neutral_planets[:12]:
        my_t, enemy_t = world.reaction_times(p.id)
        reaction_gaps.append(float(enemy_t - my_t))
        neutral_gaps.append(float(min(my_t, enemy_t)))

    my_center = []
    enemy_center = []
    for p in my_planets:
        my_center.append(math.hypot(float(p.x) - 50.0, float(p.y) - 50.0))
    for p in enemy_planets:
        enemy_center.append(math.hypot(float(p.x) - 50.0, float(p.y) - 50.0))

    comet_lives = []
    for pid in list(world.comet_ids)[:12]:
        try:
            comet_lives.append(float(world.comet_life(pid)))
        except Exception:
            continue

    domination = (float(world.my_total) - float(world.max_enemy_strength)) / max(
        1.0, float(world.my_total + world.max_enemy_strength)
    )
    prod_dom = (float(world.my_prod) - float(world.enemy_prod)) / max(1.0, total_prod)
    ship_dom = (float(world.my_total) - float(world.enemy_total)) / max(1.0, total_strength)

    state = np.array(
        [
            float(world.step) / 500.0,
            float(world.remaining_steps) / 500.0,
            float(len(world.my_planets)) / total_planets,
            float(len(world.enemy_planets)) / total_planets,
            float(len(world.neutral_planets)) / total_planets,
            float(world.my_total) / total_strength,
            float(world.enemy_total) / total_strength,
            float(world.my_prod) / total_prod,
            float(world.enemy_prod) / total_prod,
            domination,
            prod_dom,
            ship_dom,
            reserve_sum / max(1.0, float(world.my_total)),
            avail_sum / max(1.0, float(world.my_total)),
            float(len(world.doomed_candidates)) / total_planets,
            float(len(world.threatened_candidates)) / total_planets,
            threat_deficit / max(1.0, float(world.my_total)),
            _safe_mean(threat_turns) / 100.0,
            _safe_mean(reaction_gaps) / 20.0,
            _safe_mean(neutral_gaps) / 20.0,
            _safe_mean(my_center) / 100.0,
            _safe_mean(enemy_center) / 100.0,
            float(len(world.comet_ids)) / total_planets,
            _safe_mean(comet_lives) / 100.0,
            float(len(world.static_neutral_planets)) / max(1, len(world.neutral_planets) or 1),
            1.0 if world.is_early else 0.0,
            1.0 if world.is_opening else 0.0,
            1.0 if world.is_late else 0.0,
            1.0 if world.is_very_late else 0.0,
            1.0 if world.is_four_player else 0.0,
            _safe_mean([float(p.production) for p in my_planets]) / 5.0,
            _safe_mean([float(p.production) for p in enemy_planets]) / 5.0,
            _safe_mean([float(p.ships) for p in my_planets]) / 100.0,
        ],
        dtype=np.float32,
    )
    if len(state) != STATE_FEATURE_DIM:
        raise AssertionError(f"state feature dim mismatch: {len(state)} != {STATE_FEATURE_DIM}")
    return state


def _style_one_hot(style: str) -> np.ndarray:
    vec = np.zeros(len(PLAN_STYLES), dtype=np.float32)
    vec[STYLE_TO_ID[style]] = 1.0
    return vec


# Note: an earlier draft included a `_collect_action_stats` helper that was
# never wired up. It has been removed -- the live feature builder is
# `build_full_features` below.


def _target_score(world, src, tgt, style: str) -> float:
    dist = _plan_distance(src, tgt)
    my_t, enemy_t = world.reaction_times(tgt.id)
    reaction_gap = float(enemy_t - my_t)
    is_enemy = _is_enemy(int(tgt.owner), world.player)
    is_neutral = int(tgt.owner) == -1
    is_friendly = int(tgt.owner) == world.player
    is_comet = tgt.id in world.comet_ids
    is_static = world.is_static(tgt.id)

    base = 0.0
    if style == "attack":
        base = 1.7 * float(tgt.production) - 0.38 * float(tgt.ships) - 0.43 * dist + 0.14 * reaction_gap
        if is_enemy:
            base += 2.2
        if is_static:
            base += 0.35
        if world.is_late:
            base += 0.5
    elif style == "expand":
        base = 2.8 * float(tgt.production) - 0.24 * float(tgt.ships) - 0.55 * dist + 0.10 * reaction_gap
        if is_neutral:
            base += 2.4
        if is_static:
            base += 0.8
        if world.is_opening:
            base += 0.8
        if world.is_early:
            base += 0.5
    elif style == "comet":
        base = 1.9 * float(tgt.production) - 0.30 * float(tgt.ships) - 0.42 * dist + 0.12 * reaction_gap
        if is_comet:
            base += 3.8
        if is_static:
            base += 0.35
        if is_neutral:
            base += 0.8
    elif style == "reserve":
        base = 1.2 * float(tgt.production) - 0.30 * float(tgt.ships) - 0.35 * dist
        if is_enemy:
            base += 1.1
        if is_neutral:
            base += 0.6
        if _domination(world) < 0:
            base += 0.4
    else:  # balanced fallback
        base = 1.6 * float(tgt.production) - 0.33 * float(tgt.ships) - 0.45 * dist + 0.10 * reaction_gap
        if is_neutral:
            base += 1.2
        if is_enemy:
            base += 0.7
        if is_static:
            base += 0.25
        if is_friendly:
            base -= 2.0
    if tgt.id in world.threatened_candidates:
        base += 0.25
    return float(base)


def _source_priority(world, src, style: str) -> float:
    budget = float(world.source_attack_left(src.id, defaultdict(int)))
    reserve = float(world.reserve.get(src.id, 0))
    center_dist = math.hypot(float(src.x) - 50.0, float(src.y) - 50.0)
    threat = 1.0 if src.id in world.threatened_candidates else 0.0
    doom = 1.0 if src.id in world.doomed_candidates else 0.0

    if style == "defense":
        return 0.45 * budget + 0.2 * float(src.production) + 0.15 * float(src.ships) - 0.15 * center_dist - 1.2 * threat + 1.0 * doom
    if style == "attack":
        return 0.55 * budget + 0.25 * float(src.production) + 0.10 * float(src.ships) - 0.05 * center_dist - 0.15 * threat
    if style == "expand":
        return 0.50 * budget + 0.35 * float(src.production) + 0.08 * float(src.ships) - 0.06 * center_dist
    if style == "comet":
        return 0.50 * budget + 0.20 * float(src.production) + 0.08 * float(src.ships) - 0.04 * center_dist
    if style == "reserve":
        return 0.25 * budget + 0.25 * reserve + 0.15 * float(src.ships) - 0.18 * threat
    return 0.45 * budget + 0.25 * float(src.production) + 0.10 * float(src.ships) - 0.05 * center_dist


def _desired_ships(world, src, tgt, style: str, budget: int, eta: float) -> int:
    if budget <= 0:
        return 0
    if style == "defense":
        need = int(world.reinforcement_needed_for(tgt.id, eta))
        if need <= 0:
            info = world.threatened_candidates.get(tgt.id, {})
            need = int(max(1, info.get("deficit_hint", 1)))
        return min(budget, max(need, int(0.45 * float(src.ships))))
    need = int(world.ships_needed_to_capture(tgt.id, eta))
    if tgt.owner == -1:
        need = max(need, 3)
    if _is_enemy(int(tgt.owner), world.player):
        need += 2
    if style == "attack":
        need += 1
    elif style == "expand":
        need += 0
    elif style == "comet":
        need += 1
    elif style == "reserve":
        need = max(1, min(budget, int(float(src.ships) * 0.35)))
    return min(budget, max(need, 1))


def _build_plan_from_mode(world, style: str) -> CandidatePlan:
    actions: List[List[float]] = []
    spent = defaultdict(int)

    if style == "v7":
        try:
            actions = [list(m) for m in bot_v7.plan_moves(world) or []]
        except Exception:
            actions = []
        return CandidatePlan(name="v7", actions=actions)

    if style == "defense":
        threats = sorted(
            world.threatened_candidates.items(),
            key=lambda item: (float(item[1].get("fall_turn", 10**9)), -float(item[1].get("deficit_hint", 0))),
        )
        for tgt_id, info in threats:
            tgt = world.planet_by_id.get(tgt_id)
            if tgt is None:
                continue
            best = None
            best_score = -1.0e18
            best_eta = None
            for src in sorted(world.my_planets, key=lambda p: -_source_priority(world, p, style)):
                budget = int(world.source_attack_left(src.id, spent))
                if budget <= 0 or src.id == tgt_id:
                    continue
                guess = max(1, min(budget, 16))
                shot = world.plan_shot(src.id, tgt.id, guess)
                if shot is None:
                    continue
                eta = float(shot[1])
                score = 0.8 * float(info.get("deficit_hint", 1)) - 0.35 * _plan_distance(src, tgt) + 0.18 * float(src.production) + 0.08 * float(src.ships)
                if score > best_score:
                    best_score = score
                    best = src
                    best_eta = eta
            if best is None:
                continue
            budget = int(world.source_attack_left(best.id, spent))
            desired = _desired_ships(world, best, tgt, style, budget, best_eta or 1.0)
            if desired <= 0:
                continue
            shot = world.plan_shot(best.id, tgt.id, desired)
            if shot is None:
                continue
            actions.append([int(best.id), float(shot[0]), int(desired)])
            spent[best.id] += int(desired)
        return CandidatePlan(name=style, actions=actions)

    if style == "reserve":
        source_order = sorted(world.my_planets, key=lambda p: -_source_priority(world, p, style))
    elif style == "attack":
        source_order = sorted(world.my_planets, key=lambda p: -_source_priority(world, p, style))
    elif style == "expand":
        source_order = sorted(world.my_planets, key=lambda p: -_source_priority(world, p, style))
    elif style == "comet":
        source_order = sorted(world.my_planets, key=lambda p: -_source_priority(world, p, style))
    else:
        source_order = list(world.my_planets)

    for src in source_order:
        budget = int(world.source_attack_left(src.id, spent))
        if budget <= 0:
            continue
        if style == "reserve" and budget < 10 and src.id not in world.threatened_candidates:
            continue
        if style == "attack":
            targets = [p for p in world.enemy_planets] or [p for p in world.neutral_planets]
        elif style == "expand":
            targets = [p for p in world.neutral_planets] or [p for p in world.enemy_planets]
        elif style == "comet":
            comet_targets = [world.planet_by_id[pid] for pid in world.comet_ids if pid in world.planet_by_id]
            targets = comet_targets or [p for p in world.neutral_planets] or [p for p in world.enemy_planets]
        else:
            targets = [p for p in world.neutral_planets] + [p for p in world.enemy_planets] + [p for p in world.my_planets if p.id in world.threatened_candidates]

        best = None
        best_score = -1.0e18
        best_eta = None
        for tgt in targets:
            if tgt.id == src.id:
                continue
            score = _target_score(world, src, tgt, style)
            if style == "reserve" and tgt.owner == world.player:
                score -= 1.0
            if score > best_score:
                guess = max(1, min(budget, 16))
                shot = world.plan_shot(src.id, tgt.id, guess)
                if shot is None:
                    continue
                best = tgt
                best_score = score
                best_eta = float(shot[1])

        if best is None:
            continue

        desired = _desired_ships(world, src, best, style, budget, best_eta or 1.0)
        if desired <= 0:
            continue
        if style == "reserve" and desired < max(1, int(0.25 * budget)) and best.owner == world.player:
            continue
        shot = world.plan_shot(src.id, best.id, desired)
        if shot is None:
            continue
        actions.append([int(src.id), float(shot[0]), int(desired)])
        spent[src.id] += int(desired)

    return CandidatePlan(name=style, actions=actions)


def _build_hold_plan(world) -> CandidatePlan:
    """A do-nothing plan; gives the ranker a meaningful 'no-op' alternative."""
    return CandidatePlan(name="hold", actions=[])


def _build_all_in_plan(world) -> CandidatePlan:
    """Send the maximum legal ships from the strongest source to the best target.

    Distinct from attack/expand because it commits a single decisive thrust
    rather than spreading. Forces a high-variance branch into the candidate set.
    """
    actions: List[List[float]] = []
    spent: Dict[int, int] = defaultdict(int)
    candidates = sorted(
        world.my_planets,
        key=lambda p: -(0.5 * float(p.ships) + 0.3 * float(p.production) + 0.2 * float(world.source_attack_left(p.id, spent))),
    )
    for src in candidates[:1]:
        budget = int(world.source_attack_left(src.id, spent))
        if budget <= 0:
            continue
        targets = [p for p in world.enemy_planets] or [p for p in world.neutral_planets]
        if not targets:
            break
        best = max(targets, key=lambda t: _target_score(world, src, t, "attack"))
        shot = world.plan_shot(src.id, best.id, budget)
        if shot is None:
            continue
        actions.append([int(src.id), float(shot[0]), int(budget)])
        spent[src.id] += budget
    return CandidatePlan(name="all_in", actions=actions)


def _build_split_plan(world) -> CandidatePlan:
    """Distribute attacks across up to 3 distinct targets per source.

    Acts as a counter-candidate to the v7/attack/expand branches that tend to
    converge on the single highest-scoring target.
    """
    actions: List[List[float]] = []
    spent: Dict[int, int] = defaultdict(int)
    target_pool = [p for p in world.neutral_planets] + [p for p in world.enemy_planets]
    if not target_pool:
        return CandidatePlan(name="split", actions=[])
    for src in sorted(world.my_planets, key=lambda p: -float(p.production)):
        budget = int(world.source_attack_left(src.id, spent))
        if budget < 6:
            continue
        ranked = sorted(
            target_pool,
            key=lambda t: -_target_score(world, src, t, "expand") if t.id != src.id else 1e18,
        )
        picks = [t for t in ranked if t.id != src.id][:3]
        if not picks:
            continue
        share = max(1, budget // max(1, len(picks)))
        remaining = budget
        for tgt in picks:
            if remaining <= 0:
                break
            send = min(remaining, share)
            shot = world.plan_shot(src.id, tgt.id, send)
            if shot is None:
                continue
            actions.append([int(src.id), float(shot[0]), int(send)])
            spent[src.id] += send
            remaining -= send
    return CandidatePlan(name="split", actions=actions)


def build_candidate_plans(world) -> List[CandidatePlan]:
    plans = [_build_plan_from_mode(world, "v7")]
    for style in ["attack", "expand", "defense", "comet", "reserve"]:
        plans.append(_build_plan_from_mode(world, style))
    plans.append(_build_all_in_plan(world))
    plans.append(_build_split_plan(world))
    plans.append(_build_hold_plan(world))

    state = build_state_features(world)
    for plan in plans:
        plan.features = build_full_features(world, state, plan)
        plan.stats = summarize_plan(world, plan)
    return plans


def summarize_plan(world, plan: CandidatePlan) -> Dict[str, float]:
    actions = plan.actions or []
    stats = {
        "n_actions": float(len(actions)),
        "total_ships": 0.0,
        "enemy_ratio": 0.0,
        "neutral_ratio": 0.0,
        "friendly_ratio": 0.0,
        "comet_ratio": 0.0,
        "defense_ratio": 0.0,
        "reserve_ratio": 0.0,
        "target_entropy": 0.0,
        "source_entropy": 0.0,
    }
    if not actions:
        return stats

    target_ids = []
    source_ids = []
    total_ships = 0.0
    enemy = neutral = friendly = comet = defense = 0
    for action in actions:
        if action is None or len(action) != 3:
            continue
        ships = float(action[2])
        total_ships += ships
        source_ids.append(int(action[0]))
        tgt = _infer_target(world, action)
        if tgt is None:
            continue
        target_ids.append(int(tgt.id))
        if tgt.owner == world.player:
            friendly += 1
            defense += 1
        elif tgt.owner == -1:
            neutral += 1
        else:
            enemy += 1
        if tgt.id in world.comet_ids:
            comet += 1

    stats["total_ships"] = total_ships
    stats["enemy_ratio"] = float(enemy) / max(1, len(actions))
    stats["neutral_ratio"] = float(neutral) / max(1, len(actions))
    stats["friendly_ratio"] = float(friendly) / max(1, len(actions))
    stats["comet_ratio"] = float(comet) / max(1, len(actions))
    stats["defense_ratio"] = float(defense) / max(1, len(actions))
    stats["reserve_ratio"] = 1.0 - min(1.0, total_ships / max(1.0, float(world.my_total)))
    stats["target_entropy"] = _normalized_entropy(target_ids)
    stats["source_entropy"] = _normalized_entropy(source_ids)
    return stats


def build_full_features(world, state_features: np.ndarray, plan: CandidatePlan) -> np.ndarray:
    inferred = []
    for action in plan.actions:
        tgt = _infer_target(world, action)
        if action is not None and len(action) == 3 and tgt is not None:
            inferred.append((action, tgt))

    source_prods = [float(world.planet_by_id[int(a[0])].production) for a, _ in inferred]
    source_ships = [float(world.planet_by_id[int(a[0])].ships) for a, _ in inferred]
    source_dists = [_plan_distance(world.planet_by_id[int(a[0])], tgt) for a, tgt in inferred]
    source_eta = []
    for action, tgt in inferred:
        shot = world.plan_shot(int(action[0]), tgt.id, max(1, int(action[2])))
        if shot is not None:
            source_eta.append(float(shot[1]))

    style_vec = _style_one_hot(plan.name if plan.name in STYLE_TO_ID else "v7")
    n_actions_safe = max(1, len(plan.actions))
    action_stats = np.array(
        [
            *style_vec.tolist(),
            float(plan.stats.get("n_actions", 0.0)) / max(1, len(world.my_planets)),
            float(plan.stats.get("total_ships", 0.0)) / max(1.0, float(world.my_total)),
            float(plan.stats.get("enemy_ratio", 0.0)),
            float(plan.stats.get("neutral_ratio", 0.0)),
            float(plan.stats.get("friendly_ratio", 0.0)),
            float(plan.stats.get("comet_ratio", 0.0)),
            float(plan.stats.get("defense_ratio", 0.0)),
            float(plan.stats.get("reserve_ratio", 0.0)),
            float(plan.stats.get("target_entropy", 0.0)),
            float(plan.stats.get("source_entropy", 0.0)),
            _safe_mean(source_prods) / 5.0,
            _safe_mean(source_ships) / 100.0,
            _safe_mean(source_dists) / 100.0,
            _safe_mean(source_eta) / 100.0,
            float(plan.stats.get("enemy_ratio", 0.0)) * float(state_features[9]),
            float(plan.stats.get("neutral_ratio", 0.0)) * float(state_features[25]),
            float(plan.stats.get("defense_ratio", 0.0)) * float(state_features[15]),
            float(plan.stats.get("comet_ratio", 0.0)) * float(state_features[22]),
            float(plan.stats.get("enemy_ratio", 0.0)) * float(state_features[27]),
            float(plan.stats.get("neutral_ratio", 0.0)) * float(state_features[26]),
            float(plan.stats.get("reserve_ratio", 0.0)) * float(state_features[13]),
            float(plan.stats.get("reserve_ratio", 0.0)) * float(state_features[7]),
            float(plan.stats.get("enemy_ratio", 0.0)) * float(state_features[5]),
            float(plan.stats.get("neutral_ratio", 0.0)) * float(state_features[1]),
            float(plan.stats.get("defense_ratio", 0.0)) * float(state_features[15]),
            float(plan.stats.get("comet_ratio", 0.0)) * float(state_features[23]),
            float(plan.stats.get("enemy_ratio", 0.0)) * float(state_features[28]),
            float(plan.stats.get("neutral_ratio", 0.0)) * float(state_features[24]),
            float(plan.stats.get("reserve_ratio", 0.0)) * float(state_features[12]),
            float(plan.stats.get("total_ships", 0.0)) / max(1.0, float(world.my_total)),
            float(len(plan.actions)) / max(1, len(world.my_planets)),
            float(sum(1 for a in plan.actions if a and len(a) == 3 and int(a[0]) in world.doomed_candidates)) / n_actions_safe,
        ],
        dtype=np.float32,
    )
    if len(action_stats) != PLAN_FEATURE_DIM:
        raise AssertionError(f"plan feature dim mismatch: {len(action_stats)} != {PLAN_FEATURE_DIM}")

    interaction = np.array(
        [
            float(state_features[9] * plan.stats.get("enemy_ratio", 0.0)),
            float(state_features[25] * plan.stats.get("neutral_ratio", 0.0)),
            float(state_features[27] * plan.stats.get("enemy_ratio", 0.0)),
            float(state_features[15] * plan.stats.get("defense_ratio", 0.0)),
            float(state_features[22] * plan.stats.get("comet_ratio", 0.0)),
            float(state_features[13] * plan.stats.get("reserve_ratio", 0.0)),
            float(state_features[7] * plan.stats.get("reserve_ratio", 0.0)),
            float(state_features[4] * plan.stats.get("neutral_ratio", 0.0)),
        ],
        dtype=np.float32,
    )
    x = np.concatenate([state_features, action_stats, interaction]).astype(np.float32)
    if len(x) != INPUT_DIM:
        raise AssertionError(f"input dim mismatch: {len(x)} != {INPUT_DIM}")
    return x


def select_plan(world, model: Optional[LinearV8Model] = None) -> Tuple[CandidatePlan, List[CandidatePlan], np.ndarray]:
    state = build_state_features(world)
    plans = build_candidate_plans(world)
    if model is None:
        model = LinearV8Model.zero()
    feats = np.stack([plan.features for plan in plans], axis=0) if plans else np.zeros((0, INPUT_DIM), dtype=np.float32)
    scores = model.score_candidates(feats) if len(feats) else np.zeros(0, dtype=np.float32)
    if len(scores):
        idx = int(np.argmax(scores))
    else:
        idx = 0
    return plans[idx], plans, state


def candidate_scores(model: LinearV8Model, plans: Sequence[CandidatePlan]) -> np.ndarray:
    if not plans:
        return np.zeros(0, dtype=np.float32)
    feats = np.stack([plan.features for plan in plans], axis=0)
    return model.score_candidates(feats)


def one_hot_label(n: int, idx: int) -> np.ndarray:
    y = np.zeros(n, dtype=np.float32)
    if n > 0:
        y[int(max(0, min(idx, n - 1)))] = 1.0
    return y


def clone_game(game):
    from SimGame import SimGame

    return SimGame(game.state.copy(), n_players=game.n_players)


def step_policy_episode(game, our_player: int, our_actions: List[List[float]], opponent_action_fn, steps: int = 3) -> float:
    """Short rollout score from a candidate plan."""
    from SimGame import SimGame

    sim = SimGame(game.state.copy(), n_players=game.n_players)
    if sim.is_terminal():
        scores = sim.scores()
        return float(scores[our_player] - max(s for i, s in enumerate(scores) if i != our_player))

    # First step: apply candidate plan.
    actions_by_player = {}
    for player in range(sim.n_players):
        if player == our_player:
            actions_by_player[player] = our_actions
        else:
            actions_by_player[player] = opponent_action_fn(sim, player)
    sim.step(actions_by_player)

    # Roll out a few more steps with the reference policy.
    for _ in range(max(0, int(steps) - 1)):
        if sim.is_terminal():
            break
        actions_by_player = {}
        for player in range(sim.n_players):
            actions_by_player[player] = opponent_action_fn(sim, player)
        sim.step(actions_by_player)

    scores = sim.scores()
    if not scores:
        return 0.0
    our_score = float(scores[our_player])
    best_other = max(float(s) for i, s in enumerate(scores) if i != our_player)
    winner = sim.winner()
    win_term = 1.0 if winner == our_player else (-1.0 if winner >= 0 else 0.0)
    margin = our_score - best_other
    scale = max(1.0, sum(abs(float(s)) for s in scores))
    return float(0.7 * win_term + 0.3 * math.tanh(2.5 * margin / scale))


def build_training_example(world, candidate_plans: Sequence[CandidatePlan], label: int, value_target: float) -> TrainingExample:
    state = build_state_features(world)
    feats = np.stack([plan.features for plan in candidate_plans], axis=0)
    return TrainingExample(
        state_features=state,
        candidate_features=feats,
        label=int(label),
        value_target=float(value_target),
    )
