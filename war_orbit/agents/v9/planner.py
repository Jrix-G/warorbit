"""Strategically diverse candidate generation for V9."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from ...features.plan_features import PlanCandidate


@dataclass
class PlanningParameters:
    max_candidates: int = 18
    candidate_diversity: float = 1.0
    finisher_bias: float = 1.0
    min_source_ships: int = 8
    max_moves_per_plan: int = 12
    focus_enemy_id: Optional[int] = None
    front_anchor_id: Optional[int] = None


class MoveBuilder:
    """Per-candidate move accumulator with source budgets and commitments."""

    def __init__(
        self,
        world,
        *,
        reserve_scale: float = 1.0,
        allow_reserve: bool = False,
        max_moves: int = 12,
    ):
        self.world = world
        self.reserve_scale = float(reserve_scale)
        self.allow_reserve = bool(allow_reserve)
        self.max_moves = int(max_moves)
        self.moves: List[List] = []
        self.spent: Dict[int, int] = defaultdict(int)
        self.commitments: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)

    def inventory_left(self, src_id: int) -> int:
        src = self.world.planet_by_id.get(int(src_id))
        if src is None:
            return 0
        return max(0, int(src.ships) - self.spent[int(src_id)])

    def attack_left(self, src_id: int) -> int:
        src = self.world.planet_by_id.get(int(src_id))
        if src is None:
            return 0
        if self.allow_reserve:
            budget = int(src.ships * 0.92)
        else:
            base = int(self.world.available.get(src_id, max(0, int(src.ships * 0.56))))
            if src_id in self.world.threatened_candidates:
                base = min(base, max(0, int(src.ships * 0.35)))
            budget = int(base * self.reserve_scale)
        return max(0, min(int(src.ships), budget) - self.spent[int(src_id)])

    def add_move(self, src_id: int, angle: float, ships: int, *, attack_budget: bool = True) -> int:
        if len(self.moves) >= self.max_moves:
            return 0
        left = self.attack_left(src_id) if attack_budget else self.inventory_left(src_id)
        send = max(0, min(int(ships), int(left)))
        if send <= 0:
            return 0
        self.moves.append([int(src_id), float(angle), int(send)])
        self.spent[int(src_id)] += int(send)
        return int(send)

    def add_commitment(self, target_id: int, turns: int, ships: int) -> None:
        self.commitments[int(target_id)].append((int(turns), int(self.world.player), int(ships)))


def _dist(a, b) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def _ship_lead(world) -> float:
    strongest = max((s for o, s in world.owner_strength.items() if o not in (-1, world.player)), default=0)
    return float(world.my_total) / max(1.0, float(strongest))


def _prod_lead(world) -> float:
    strongest = max((p for o, p in world.owner_production.items() if o not in (-1, world.player)), default=0)
    return float(world.my_prod) / max(1.0, float(strongest))


def _focus_enemy_id(world, preferred: Optional[int] = None) -> Optional[int]:
    if preferred is not None and any(int(p.owner) == int(preferred) for p in world.enemy_planets):
        return int(preferred)
    if world.weakest_enemy_id is not None and any(int(p.owner) == int(world.weakest_enemy_id) for p in world.enemy_planets):
        return int(world.weakest_enemy_id)
    owners = [
        int(o)
        for o in world.owner_strength
        if o not in (-1, world.player) and any(int(p.owner) == int(o) for p in world.enemy_planets)
    ]
    if not owners:
        return None
    return min(owners, key=lambda o: float(world.owner_strength.get(o, 0)) + 18.0 * float(world.owner_production.get(o, 0)))


def _frontier_objectives(world, focus_enemy_id: Optional[int] = None, *, include_neutrals: bool = True) -> List[object]:
    objectives: List[object] = []
    if focus_enemy_id is not None:
        objectives.extend([p for p in world.enemy_planets if int(p.owner) == int(focus_enemy_id)])
    if not objectives:
        objectives.extend(world.enemy_planets)
    if include_neutrals:
        objectives.extend(world.neutral_planets)
    return objectives


def _frontier_anchor(world, focus_enemy_id: Optional[int] = None, preferred_anchor_id: Optional[int] = None):
    if not world.my_planets:
        return None
    if preferred_anchor_id is not None:
        preferred = world.planet_by_id.get(int(preferred_anchor_id))
        if preferred is not None and preferred.owner == world.player and preferred.id not in world.doomed_candidates:
            return preferred
    objectives = _frontier_objectives(world, focus_enemy_id, include_neutrals=True)
    if not objectives:
        return max(world.my_planets, key=lambda p: p.ships)
    safe = [p for p in world.my_planets if p.id not in world.doomed_candidates] or list(world.my_planets)
    return min(safe, key=lambda p: (min(_dist(p, t) for t in objectives), p.id in world.threatened_candidates, -float(p.production)))


def _active_front_count(world, focus_enemy_id: Optional[int] = None) -> int:
    enemies = [p for p in world.enemy_planets if focus_enemy_id is None or int(p.owner) == int(focus_enemy_id)]
    if not enemies:
        enemies = list(world.enemy_planets)
    count = 0
    for mine in world.my_planets:
        if any(_dist(mine, enemy) <= 36.0 + float(mine.radius) + float(enemy.radius) for enemy in enemies):
            count += 1
    return count


def _candidate_metadata(world, focus_enemy_id: Optional[int], anchor, **extra: float) -> Dict[str, float]:
    meta: Dict[str, float] = {
        "active_fronts": float(_active_front_count(world, focus_enemy_id)),
    }
    if focus_enemy_id is not None:
        meta["focus_enemy_id"] = float(focus_enemy_id)
    if anchor is not None:
        meta["front_anchor_id"] = float(anchor.id)
    meta.update({str(k): float(v) for k, v in extra.items()})
    return meta


def _target_score(target, turns: int, needed: int, world, family: str) -> float:
    prod = float(target.production)
    ships = float(target.ships)
    time_tax = 1.0 + 0.055 * max(1, int(turns))
    need_tax = 1.0 + 0.035 * max(1, int(needed))
    owner_bonus = 0.0
    if target.owner == -1:
        owner_bonus = 11.0
    elif target.owner != world.player:
        owner_bonus = 18.0
    score = owner_bonus + 20.0 * prod - 0.25 * ships
    if family == "aggressive_expansion":
        score += 20.0 if target.owner == -1 else -7.0
        score += max(0.0, 18.0 - ships)
    elif family == "resource_denial":
        score += 28.0 * prod if target.owner not in (-1, world.player) else -15.0
        enemy_prod = world.owner_production.get(target.owner, 0)
        score += 3.5 * enemy_prod
    elif family == "endgame_finisher":
        if target.owner == world.weakest_enemy_id:
            score += 35.0
        if world.is_late or world.is_very_late:
            score += 24.0
        score += max(0.0, 24.0 - ships)
    elif family == "opportunistic_snipe":
        score += max(0.0, 22.0 - ships) + 9.0 * prod
    elif family == "probe":
        score += 8.0 * prod - 0.10 * ships
    return score / (time_tax * need_tax)


def _shot_option(
    builder: MoveBuilder,
    src,
    target,
    *,
    family: str,
    aggression: float = 1.0,
    min_send: int = 1,
) -> Optional[dict]:
    left = builder.attack_left(src.id)
    if left < min_send:
        return None
    rough = max(min_send, min(left, int(target.ships) + 1))
    aim = builder.world.plan_shot(src.id, target.id, rough)
    if aim is None:
        return None
    turns = int(max(1, math.ceil(aim[1])))
    if builder.world.is_very_late and turns > builder.world.remaining_steps - 1:
        return None
    if builder.world.is_late and turns > builder.world.remaining_steps - 4:
        return None
    needed = builder.world.ships_needed_to_capture(target.id, turns, builder.commitments)
    if needed <= 0:
        return None
    send = int(math.ceil(max(float(min_send), needed * aggression)))
    send = min(left, send)
    if send < needed and left < needed:
        return None
    final_aim = builder.world.plan_shot(src.id, target.id, max(1, send))
    if final_aim is None:
        return None
    turns = int(max(1, math.ceil(final_aim[1])))
    needed = builder.world.ships_needed_to_capture(target.id, turns, builder.commitments)
    send = min(left, int(math.ceil(max(float(min_send), needed * aggression))))
    if send < needed:
        return None
    score = _target_score(target, turns, needed, builder.world, family)
    score /= 1.0 + 0.012 * _dist(src, target)
    return {
        "score": score,
        "src": src,
        "target": target,
        "angle": float(final_aim[0]),
        "turns": turns,
        "needed": int(needed),
        "send": int(send),
        "left": int(left),
    }


def _commit_target(
    builder: MoveBuilder,
    target,
    *,
    family: str,
    aggression: float,
    max_sources: int,
    min_send: int,
) -> float:
    options = []
    for src in builder.world.my_planets:
        opt = _shot_option(builder, src, target, family=family, aggression=aggression, min_send=min_send)
        if opt is not None:
            options.append(opt)
    if not options:
        return 0.0
    options.sort(key=lambda o: -o["score"])

    for opt in options:
        if opt["send"] >= opt["needed"]:
            sent = builder.add_move(opt["src"].id, opt["angle"], opt["send"], attack_budget=True)
            if sent >= opt["needed"]:
                builder.add_commitment(target.id, opt["turns"], sent)
                return float(opt["score"])
            return 0.0

    chosen = options[:max(2, int(max_sources))]
    if sum(o["left"] for o in chosen) < min(o["needed"] for o in chosen):
        return 0.0
    joint_turn = max(o["turns"] for o in chosen)
    missing = builder.world.ships_needed_to_capture(target.id, joint_turn, builder.commitments)
    if missing <= 0:
        return 0.0
    if sum(o["left"] for o in chosen) < missing:
        return 0.0

    remaining = int(math.ceil(missing * aggression))
    committed = 0
    score = 0.0
    for idx, opt in enumerate(chosen):
        other_left = sum(o["left"] for o in chosen[idx + 1 :])
        send = min(opt["left"], max(1, remaining - other_left))
        if send <= 0:
            continue
        sent = builder.add_move(opt["src"].id, opt["angle"], send, attack_budget=True)
        committed += sent
        remaining -= sent
        score += opt["score"]
        if remaining <= 0:
            break
    if committed >= missing:
        builder.add_commitment(target.id, joint_turn, committed)
        return float(score / max(1, len(chosen)))
    return 0.0


def _commit_reinforcements(builder: MoveBuilder, *, force: bool = False) -> float:
    score = 0.0
    threatened = list(builder.world.threatened_candidates)
    if force:
        threatened.extend(pid for pid in builder.world.doomed_candidates if pid not in threatened)
    for target_id in threatened:
        target = builder.world.planet_by_id.get(target_id)
        if target is None or target.owner != builder.world.player:
            continue
        best = None
        for src in builder.world.my_planets:
            if src.id == target.id:
                continue
            left = builder.inventory_left(src.id)
            if left < 3:
                continue
            probe = min(left, max(3, int(target.production) * 4))
            aim = builder.world.plan_shot(src.id, target.id, probe)
            if aim is None:
                continue
            turns = int(max(1, math.ceil(aim[1])))
            need = builder.world.reinforcement_needed_for(target.id, turns, builder.commitments)
            if need <= 0 and not force:
                continue
            send = min(left, max(need + 2, int(src.ships * 0.28)))
            if send <= 0:
                continue
            option_score = (40.0 + 8.0 * target.production + 0.2 * need) / (1.0 + turns)
            if best is None or option_score > best[0]:
                best = (option_score, src, aim[0], turns, send)
        if best is None:
            continue
        opt_score, src, angle, turns, send = best
        sent = builder.add_move(src.id, angle, send, attack_budget=False)
        if sent > 0:
            builder.add_commitment(target.id, turns, sent)
            score += opt_score
    return score


def _stage_to_front(
    builder: MoveBuilder,
    *,
    ratio: float,
    min_send: int,
    focus_enemy_id: Optional[int] = None,
    preferred_anchor_id: Optional[int] = None,
    max_transfers: Optional[int] = None,
    allow_threatened_sources: bool = False,
) -> float:
    front = _frontier_anchor(builder.world, focus_enemy_id, preferred_anchor_id)
    if front is None:
        return 0.0
    objectives = _frontier_objectives(builder.world, focus_enemy_id, include_neutrals=True)
    if not objectives:
        return 0.0
    frontier_distance = {
        p.id: min(_dist(p, obj) for obj in objectives)
        for p in builder.world.my_planets
    }
    front_dist = frontier_distance.get(front.id, min(_dist(front, obj) for obj in objectives))
    safe_fronts = [
        p for p in builder.world.my_planets
        if p.id not in builder.world.doomed_candidates
        and (allow_threatened_sources or p.id not in builder.world.threatened_candidates)
    ]
    if not safe_fronts:
        safe_fronts = [p for p in builder.world.my_planets if p.id not in builder.world.doomed_candidates]
    score = 0.0
    transfers = 0
    for src in sorted(builder.world.my_planets, key=lambda p: (frontier_distance.get(p.id, 0.0), float(p.ships)), reverse=True):
        if max_transfers is not None and transfers >= int(max_transfers):
            break
        if src.id == front.id:
            continue
        if src.id in builder.world.doomed_candidates:
            continue
        if not allow_threatened_sources and src.id in builder.world.threatened_candidates:
            continue
        left = builder.attack_left(src.id)
        if left < min_send:
            continue
        src_dist = frontier_distance.get(src.id, min(_dist(src, obj) for obj in objectives))
        if src_dist <= max(front_dist + 6.0, front_dist * 1.22):
            continue
        stage_candidates = [
            p for p in safe_fronts
            if p.id != src.id
            and frontier_distance.get(p.id, src_dist) < src_dist * 0.86
        ]
        if stage_candidates:
            stage_target = min(stage_candidates, key=lambda p: (_dist(src, p), frontier_distance.get(p.id, 999.0)))
        else:
            stage_target = front
        send = int(left * ratio)
        if send < min_send:
            continue
        aim = builder.world.plan_shot(src.id, stage_target.id, send)
        if aim is None:
            continue
        if int(math.ceil(aim[1])) > 45:
            continue
        sent = builder.add_move(src.id, aim[0], send, attack_budget=True)
        if sent > 0:
            transfers += 1
            anchor_bonus = 1.12 if stage_target.id == front.id else 1.0
            score += anchor_bonus * sent / max(1.0, aim[1])
    return score


def _commit_target_from_sources(
    builder: MoveBuilder,
    sources: Iterable[object],
    target,
    *,
    family: str,
    aggression: float,
    min_send: int,
) -> float:
    options = []
    for src in sources:
        opt = _shot_option(builder, src, target, family=family, aggression=aggression, min_send=min_send)
        if opt is not None:
            options.append(opt)
    if not options:
        return 0.0
    options.sort(key=lambda o: -o["score"])
    for opt in options:
        if opt["send"] < opt["needed"]:
            continue
        sent = builder.add_move(opt["src"].id, opt["angle"], opt["send"], attack_budget=True)
        if sent >= opt["needed"]:
            builder.add_commitment(target.id, opt["turns"], sent)
            return float(opt["score"])
    return 0.0


class V9Planner:
    """Candidate generator with explicit macro-strategy families."""

    def __init__(self, params: Optional[PlanningParameters] = None):
        self.params = params or PlanningParameters()

    def generate(self, world, rng: Optional[random.Random] = None) -> List[PlanCandidate]:
        candidates: List[PlanCandidate] = []
        p = self.params
        families: List[Callable[[object], Optional[PlanCandidate]]] = [
            self._balanced,
            self._aggressive_expansion,
            self._four_player_backbone,
            self._front_lock_consolidation,
            self._delayed_strike,
            self._multi_step_trap,
            self._resource_denial,
            self._staged_finisher,
            self._endgame_finisher,
            self._defensive_consolidation,
            self._opportunistic_snipe,
            self._probe,
            self._reserve_hold,
        ]
        if p.candidate_diversity >= 1.35:
            families.extend([self._wide_expansion, self._all_in_finisher, self._deep_staging])
        for fn in families:
            try:
                candidate = fn(world)
            except Exception:
                candidate = None
            if candidate is not None:
                candidates.append(candidate.clipped(p.max_moves_per_plan))

        seen = set()
        unique: List[PlanCandidate] = []
        for cand in candidates:
            key = (cand.name, tuple((m[0], round(float(m[1]), 3), int(m[2])) for m in cand.moves))
            if key in seen:
                continue
            seen.add(key)
            unique.append(cand)

        if rng is not None and p.candidate_diversity > 1.0 and unique:
            rng.shuffle(unique)
            unique.sort(key=lambda c: (len(c.moves) == 0, -c.base_score))
        else:
            unique.sort(key=lambda c: (len(c.moves) == 0, -c.base_score))
        return unique[: p.max_candidates]

    def _balanced(self, world) -> PlanCandidate:
        b = MoveBuilder(world, max_moves=self.params.max_moves_per_plan)
        base = _commit_reinforcements(b)
        targets = sorted(
            [t for t in world.planets if t.owner != world.player],
            key=lambda t: (
                t.owner != -1,
                -float(t.production),
                float(t.ships),
                min((_dist(src, t) for src in world.my_planets), default=999.0),
            ),
        )
        for target in targets[:6]:
            base += _commit_target(
                b,
                target,
                family="balanced",
                aggression=1.10 if target.owner != -1 else 1.04,
                max_sources=2,
                min_send=self.params.min_source_ships,
            )
            if len(b.moves) >= self.params.max_moves_per_plan // 2:
                break
        return PlanCandidate("v9_balanced", b.moves, "balanced", base_score=base)

    def _aggressive_expansion(self, world) -> Optional[PlanCandidate]:
        if not world.neutral_planets:
            return None
        b = MoveBuilder(world, reserve_scale=0.92, max_moves=self.params.max_moves_per_plan)
        targets = sorted(
            world.neutral_planets,
            key=lambda t: (
                -float(t.production) / max(1.0, float(t.ships)),
                min((_dist(src, t) for src in world.my_planets), default=999.0),
            ),
        )
        score = 0.0
        limit = 5 if world.is_opening or world.is_four_player else 3
        for target in targets[:limit]:
            score += _commit_target(
                b,
                target,
                family="aggressive_expansion",
                aggression=1.02,
                max_sources=2,
                min_send=max(3, self.params.min_source_ships - 2),
            )
        if not b.moves:
            return None
        return PlanCandidate("v9_aggressive_expansion", b.moves, "aggressive_expansion", base_score=score)

    def _wide_expansion(self, world) -> Optional[PlanCandidate]:
        if not world.neutral_planets:
            return None
        b = MoveBuilder(world, reserve_scale=0.82, max_moves=self.params.max_moves_per_plan)
        targets = sorted(world.neutral_planets, key=lambda t: (float(t.ships), -float(t.production)))
        score = 0.0
        for target in targets[:8]:
            score += _commit_target(b, target, family="aggressive_expansion", aggression=1.0, max_sources=1, min_send=4)
        if not b.moves:
            return None
        return PlanCandidate("v9_wide_expansion", b.moves, "aggressive_expansion", base_score=score * 0.92)

    def _four_player_backbone(self, world) -> Optional[PlanCandidate]:
        if not world.is_four_player or len(world.my_planets) < 2:
            return None
        focus = _focus_enemy_id(world, self.params.focus_enemy_id)
        anchor = _frontier_anchor(world, focus, self.params.front_anchor_id)
        active_fronts = _active_front_count(world, focus)
        b = MoveBuilder(world, reserve_scale=1.06, max_moves=self.params.max_moves_per_plan)
        score = _stage_to_front(
            b,
            ratio=0.80,
            min_send=max(6, self.params.min_source_ships - 1),
            focus_enemy_id=focus,
            preferred_anchor_id=self.params.front_anchor_id,
            max_transfers=9,
        )
        if not b.moves:
            return None
        threshold_gap = max(0, 13 - len(world.my_planets))
        score += 13.0 + 1.4 * threshold_gap + 2.2 * active_fronts
        return PlanCandidate(
            "v9_4p_backbone",
            b.moves,
            "staging_transfer",
            base_score=score,
            metadata=_candidate_metadata(
                world,
                focus,
                anchor,
                backbone=1.0,
                front_lock=1.0,
                consolidation_threshold=1.0 if 6 <= len(world.my_planets) < 15 and world.step < 130 else 0.0,
            ),
        )

    def _front_lock_consolidation(self, world) -> Optional[PlanCandidate]:
        if not world.is_four_player or len(world.my_planets) < 2:
            return None
        focus = _focus_enemy_id(world, self.params.focus_enemy_id)
        anchor = _frontier_anchor(world, focus, self.params.front_anchor_id)
        active_fronts = _active_front_count(world, focus)
        front_pressure = max(0, active_fronts - 2)
        under_threshold = len(world.my_planets) < 15 and world.step < 140
        if active_fronts <= 1 and not world.threatened_candidates and not world.doomed_candidates and not under_threshold:
            return None
        b = MoveBuilder(world, reserve_scale=0.78, max_moves=self.params.max_moves_per_plan)
        score = _commit_reinforcements(b, force=True)
        score += _stage_to_front(
            b,
            ratio=min(0.66, 0.48 + 0.06 * front_pressure),
            min_send=max(5, self.params.min_source_ships - 2),
            focus_enemy_id=focus,
            preferred_anchor_id=self.params.front_anchor_id,
            max_transfers=min(8, 6 + front_pressure),
        )
        if not b.moves:
            return None
        score += 5.0 + 2.0 * active_fronts + 4.0 * front_pressure + (6.0 if under_threshold else 0.0)
        return PlanCandidate(
            "v9_front_lock_consolidation",
            b.moves,
            "defensive_consolidation",
            base_score=score,
            metadata=_candidate_metadata(
                world,
                focus,
                anchor,
                front_lock=1.0,
                consolidation_threshold=1.0 if under_threshold else 0.0,
            ),
        )

    def _delayed_strike(self, world) -> Optional[PlanCandidate]:
        if len(world.my_planets) < 2:
            return None
        b = MoveBuilder(world, reserve_scale=1.05, max_moves=self.params.max_moves_per_plan)
        focus = _focus_enemy_id(world, self.params.focus_enemy_id)
        high_front_pressure = world.is_four_player and _active_front_count(world, focus) > 2
        score = _stage_to_front(
            b,
            ratio=0.64 if high_front_pressure else 0.58,
            min_send=max(8, self.params.min_source_ships),
            focus_enemy_id=focus,
            preferred_anchor_id=self.params.front_anchor_id,
            max_transfers=6 if high_front_pressure else 5 if world.is_four_player else None,
        )
        front = _frontier_anchor(world, focus, self.params.front_anchor_id)
        if front is not None and b.attack_left(front.id) >= self.params.min_source_ships:
            targets = sorted(
                [t for t in world.enemy_planets if t.owner != world.player],
                key=lambda t: (focus is not None and int(t.owner) != int(focus), -float(t.production), float(t.ships), _dist(front, t)),
            )
            for target in targets[:1 if high_front_pressure else 2]:
                score += _commit_target(b, target, family="delayed_strike", aggression=1.02 if high_front_pressure else 1.08, max_sources=1, min_send=self.params.min_source_ships)
        if not b.moves:
            return None
        return PlanCandidate(
            "v9_delayed_strike",
            b.moves,
            "delayed_strike",
            base_score=score,
            metadata=_candidate_metadata(world, focus, front, front_lock=1.0 if world.is_four_player else 0.0),
        )

    def _deep_staging(self, world) -> Optional[PlanCandidate]:
        if len(world.my_planets) < 3:
            return None
        b = MoveBuilder(world, reserve_scale=1.10, max_moves=self.params.max_moves_per_plan)
        focus = _focus_enemy_id(world, self.params.focus_enemy_id)
        anchor = _frontier_anchor(world, focus, self.params.front_anchor_id)
        score = _stage_to_front(
            b,
            ratio=0.82,
            min_send=max(10, self.params.min_source_ships),
            focus_enemy_id=focus,
            preferred_anchor_id=self.params.front_anchor_id,
            max_transfers=10 if world.is_four_player else None,
        )
        if not b.moves:
            return None
        return PlanCandidate(
            "v9_deep_staging",
            b.moves,
            "staging_transfer",
            base_score=score,
            metadata=_candidate_metadata(world, focus, anchor, backbone=1.0, front_lock=1.0 if world.is_four_player else 0.0),
        )

    def _multi_step_trap(self, world) -> Optional[PlanCandidate]:
        if not world.enemy_planets and not world.threatened_candidates:
            return None
        b = MoveBuilder(world, reserve_scale=0.86, max_moves=self.params.max_moves_per_plan)
        score = _commit_reinforcements(b, force=True)
        front = _frontier_anchor(world)
        if front is not None:
            nearby = sorted(
                [t for t in world.planets if t.owner != world.player],
                key=lambda t: (_dist(front, t), float(t.ships) - 3.0 * float(t.production)),
            )
            for target in nearby[:2]:
                left = b.attack_left(front.id)
                if left < 4:
                    continue
                probe = min(left, max(3, min(14, int(target.ships * 0.45) + 2)))
                aim = world.plan_shot(front.id, target.id, probe)
                if aim is None:
                    continue
                sent = b.add_move(front.id, aim[0], probe, attack_budget=True)
                if sent > 0:
                    score += 3.5 + float(target.production)
        if not b.moves:
            return None
        return PlanCandidate("v9_multi_step_trap", b.moves, "multi_step_trap", base_score=score)

    def _resource_denial(self, world) -> Optional[PlanCandidate]:
        if not world.enemy_planets:
            return None
        b = MoveBuilder(world, reserve_scale=1.0, max_moves=self.params.max_moves_per_plan)
        focus = _focus_enemy_id(world, self.params.focus_enemy_id)
        high_front_pressure = world.is_four_player and _active_front_count(world, focus) > 2
        targets = sorted(
            world.enemy_planets,
            key=lambda t: (
                focus is not None and int(t.owner) != int(focus),
                -float(t.production),
                world.owner_production.get(t.owner, 0),
                float(t.ships),
            ),
        )
        score = 0.0
        for target in targets[:2 if high_front_pressure else 4]:
            score += _commit_target(
                b,
                target,
                family="resource_denial",
                aggression=1.07 if high_front_pressure else 1.15,
                max_sources=2 if high_front_pressure else 3 if world.is_four_player else 2,
                min_send=self.params.min_source_ships,
            )
        if not b.moves:
            return None
        return PlanCandidate(
            "v9_resource_denial",
            b.moves,
            "resource_denial",
            base_score=score,
            metadata=_candidate_metadata(world, focus, _frontier_anchor(world, focus, self.params.front_anchor_id), front_lock=1.0 if world.is_four_player else 0.0),
        )

    def _staged_finisher(self, world) -> Optional[PlanCandidate]:
        if not world.is_four_player or not world.enemy_planets or len(world.my_planets) < 2:
            return None
        focus = _focus_enemy_id(world, self.params.focus_enemy_id)
        focus_targets = [t for t in world.enemy_planets if focus is None or int(t.owner) == int(focus)]
        if not focus_targets:
            return None
        focus_strength = sum(float(t.ships) for t in focus_targets)
        focus_prod = sum(float(t.production) for t in focus_targets)
        early_clear_lead = world.step >= 55 and _ship_lead(world) >= 1.25 and _prod_lead(world) >= 1.10
        if world.step < 70 and not early_clear_lead:
            return None
        ready = (
            world.is_late
            or world.is_very_late
            or _ship_lead(world) >= 1.12
            or _prod_lead(world) >= 1.12
            or len(focus_targets) <= 4
            or focus_strength <= max(1.0, float(world.my_total) * 0.72)
            or focus_prod <= max(1.0, float(world.my_prod) * 0.45)
        )
        if not ready:
            return None

        anchor = _frontier_anchor(world, focus, self.params.front_anchor_id)
        if anchor is None:
            return None
        b = MoveBuilder(world, reserve_scale=1.10, allow_reserve=True, max_moves=self.params.max_moves_per_plan)
        score = _stage_to_front(
            b,
            ratio=0.62,
            min_send=max(8, self.params.min_source_ships),
            focus_enemy_id=focus,
            preferred_anchor_id=anchor.id,
            max_transfers=4,
        )
        objectives = focus_targets
        front_sources = sorted(
            [p for p in world.my_planets if p.id not in world.doomed_candidates],
            key=lambda p: (min(_dist(p, t) for t in objectives), p.id in world.threatened_candidates, -float(p.ships)),
        )[:2]
        targets = sorted(focus_targets, key=lambda t: (float(t.ships), -float(t.production), _dist(anchor, t)))
        for target in targets[:4]:
            if len(b.moves) >= self.params.max_moves_per_plan:
                break
            score += _commit_target_from_sources(
                b,
                front_sources,
                target,
                family="endgame_finisher",
                aggression=1.18 * self.params.finisher_bias,
                min_send=5,
            )
        if not b.moves:
            return None
        score += 10.0 + max(0.0, 5.0 - len(focus_targets)) * 1.7
        return PlanCandidate(
            "v9_staged_finisher",
            b.moves,
            "endgame_finisher",
            base_score=score,
            metadata=_candidate_metadata(world, focus, anchor, staged_finisher=1.0, front_lock=1.0),
        )

    def _endgame_finisher(self, world) -> Optional[PlanCandidate]:
        lead = _ship_lead(world)
        prod = _prod_lead(world)
        if not world.enemy_planets:
            return None
        if not (world.is_late or world.is_very_late or lead >= 1.18 or prod >= 1.25):
            return None
        b = MoveBuilder(world, reserve_scale=1.25, allow_reserve=True, max_moves=self.params.max_moves_per_plan)
        weakest = _focus_enemy_id(world, self.params.focus_enemy_id)
        targets = [t for t in world.enemy_planets if weakest is None or int(t.owner) == int(weakest)]
        if not targets:
            targets = list(world.enemy_planets)
        targets.sort(key=lambda t: (float(t.ships), -float(t.production)))
        score = 0.0
        aggression = 1.22 * self.params.finisher_bias
        for target in targets[:6]:
            score += _commit_target(b, target, family="endgame_finisher", aggression=aggression, max_sources=4, min_send=5)
        if not b.moves:
            return None
        return PlanCandidate(
            "v9_endgame_finisher",
            b.moves,
            "endgame_finisher",
            base_score=score,
            metadata=_candidate_metadata(world, weakest, _frontier_anchor(world, weakest, self.params.front_anchor_id), front_lock=1.0 if world.is_four_player else 0.0),
        )

    def _all_in_finisher(self, world) -> Optional[PlanCandidate]:
        if not world.enemy_planets or (_ship_lead(world) < 1.35 and not world.is_very_late):
            return None
        b = MoveBuilder(world, reserve_scale=1.45, allow_reserve=True, max_moves=self.params.max_moves_per_plan)
        focus = _focus_enemy_id(world, self.params.focus_enemy_id)
        targets = sorted(world.enemy_planets, key=lambda t: (focus is not None and int(t.owner) != int(focus), float(t.ships)))
        score = 0.0
        for target in targets[:8]:
            score += _commit_target(b, target, family="endgame_finisher", aggression=1.45, max_sources=5, min_send=4)
        if not b.moves:
            return None
        return PlanCandidate(
            "v9_all_in_finisher",
            b.moves,
            "endgame_finisher",
            base_score=score * 0.88,
            metadata=_candidate_metadata(world, focus, _frontier_anchor(world, focus, self.params.front_anchor_id), front_lock=1.0 if world.is_four_player else 0.0),
        )

    def _defensive_consolidation(self, world) -> Optional[PlanCandidate]:
        b = MoveBuilder(world, reserve_scale=0.82 if world.is_four_player else 0.72, max_moves=self.params.max_moves_per_plan)
        focus = _focus_enemy_id(world, self.params.focus_enemy_id)
        active_fronts = _active_front_count(world, focus) if world.is_four_player else 0
        front_pressure = max(0, active_fronts - 2)
        score = _commit_reinforcements(b, force=True)
        score += _stage_to_front(
            b,
            ratio=min(0.70, 0.48 + 0.07 * front_pressure),
            min_send=max(7, self.params.min_source_ships),
            focus_enemy_id=focus,
            preferred_anchor_id=self.params.front_anchor_id,
            max_transfers=min(9, 6 + front_pressure) if world.is_four_player else None,
        )
        if not b.moves:
            return None
        score += 4.0 * front_pressure + (4.0 if world.is_four_player else 0.0)
        return PlanCandidate(
            "v9_defensive_consolidation",
            b.moves,
            "defensive_consolidation",
            base_score=score,
            metadata=_candidate_metadata(world, focus, _frontier_anchor(world, focus, self.params.front_anchor_id), front_lock=1.0 if world.is_four_player else 0.0),
        )

    def _opportunistic_snipe(self, world) -> Optional[PlanCandidate]:
        targets = [
            t for t in world.planets
            if t.owner != world.player and float(t.ships) <= 10.0 + 2.8 * float(t.production)
        ]
        if not targets:
            return None
        b = MoveBuilder(world, reserve_scale=0.94, max_moves=self.params.max_moves_per_plan)
        targets.sort(key=lambda t: (-float(t.production), float(t.ships)))
        score = 0.0
        for target in targets[:5]:
            score += _commit_target(
                b,
                target,
                family="opportunistic_snipe",
                aggression=1.01,
                max_sources=1,
                min_send=max(3, min(self.params.min_source_ships, 6)),
            )
        if not b.moves:
            return None
        return PlanCandidate("v9_opportunistic_snipe", b.moves, "opportunistic_snipe", base_score=score)

    def _probe(self, world) -> Optional[PlanCandidate]:
        if not world.enemy_planets:
            return None
        b = MoveBuilder(world, reserve_scale=0.62, max_moves=min(6, self.params.max_moves_per_plan))
        targets = sorted(world.enemy_planets, key=lambda t: (-float(t.production), float(t.ships)))
        score = 0.0
        for src in sorted(world.my_planets, key=lambda p: -b.attack_left(p.id)):
            if len(b.moves) >= 3:
                break
            left = b.attack_left(src.id)
            if left < 5:
                continue
            target = min(targets[:5], key=lambda t: _dist(src, t)) if targets else None
            if target is None:
                continue
            ships = min(left, max(3, min(12, int(0.18 * left))))
            aim = world.plan_shot(src.id, target.id, ships)
            if aim is None:
                continue
            sent = b.add_move(src.id, aim[0], ships, attack_budget=True)
            if sent > 0:
                score += 2.0 + float(target.production)
        if not b.moves:
            return None
        return PlanCandidate("v9_probe", b.moves, "probe", base_score=score)

    def _reserve_hold(self, world) -> PlanCandidate:
        b = MoveBuilder(world, reserve_scale=0.55, max_moves=self.params.max_moves_per_plan)
        score = _commit_reinforcements(b, force=True)
        return PlanCandidate("v9_reserve_hold", b.moves, "reserve_hold", base_score=score)
