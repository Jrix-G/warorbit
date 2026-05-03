"""Orbit Wars V8.5 — V7 fixes + candidate plan ranker + 4p specialised plans.

Key changes vs V7:
  Fix #0  already_in_flight check        — skip targets with enough own ships in transit
  Fix #1  garrison_floor                  — production-proportional minimum garrison
  Fix #2  garrison_ratio cap              — ensure >= GARRISON_FLOOR_RATIO on planets
  Fix #3  ThreatModel latent capacity     — raise keep_needed by latent enemy reach
  Fix #4  4p early expansion thresholds   — lower neutral margin, delay rear staging
  Fix #5  4p offense throttle             — defensive mode when 2+ active fronts
  + 4p candidate plans: opportunistic, eliminate_weakest, conservation, late_blitz, conversion_push
  + opening candidate plans: opening_fortify, opening_relay
  + probe candidate plan: probe_focus
  + Friendly staging transfers as first-class missions (transfer_push plan)
  + Robust intercept solver with local ETA search and damped fixed-point refinement
  + Linear plan ranker (zero-init = V7 baseline tie-break = safe warm-start)
  + Value head (linear regressor on state features)

The bot reuses V7 machinery (WorldModel, scoring helpers, planning primitives)
through *imports*. No copy of large blocks. The V8.5 plan loop is implemented
as `plan_moves_v8_5`. The V7 baseline plan is always candidate index 0; with
zero ranker weights the model falls back to V7 — a deliberately safe warm-start.

Public entry point: `agent(obs, config=None)`.
Training hooks (used by train_v8_5.py):
  set_ranker_weights(state_w, candidate_w, value_w)
  set_candidate_log_callback(callback)   # called as callback(log_dict)
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import bot_v7 as v7
from bot_v7 import (
    Planet, Fleet, Mission, ShotOption,
    WorldModel, _build_world,
    _build_modes, _target_value, _preferred_send, _apply_score_mods,
    _build_snipe_mission, _build_reinforcement_missions,
    _opening_filter, _is_safe_neutral, _is_contested_neutral,
    _dist, _travel_time, _estimate_arrival, _predict_target_pos, _search_safe_intercept,
    HORIZON, AIM_ITERATIONS, INTERCEPT_TOLERANCE, PARTIAL_SOURCE_MIN_SHIPS, MULTI_SOURCE_TOP_K, MULTI_SOURCE_ETA_TOLERANCE,
    HOSTILE_SWARM_ETA_TOLERANCE, MULTI_SOURCE_PLAN_PENALTY,
    THREE_SOURCE_SWARM_ENABLED, THREE_SOURCE_MIN_TARGET_SHIPS,
    THREE_SOURCE_ETA_TOLERANCE, THREE_SOURCE_PLAN_PENALTY,
    REAR_SOURCE_MIN_SHIPS, REAR_DISTANCE_RATIO, REAR_STAGE_PROGRESS,
    REAR_SEND_RATIO_TWO_PLAYER, REAR_SEND_RATIO_FOUR_PLAYER,
    REAR_SEND_MIN_SHIPS, REAR_MAX_TRAVEL_TURNS,
    AOW_HOARD_PROD_RATIO, AOW_HOARD_SEND_RATIO,
    DOOMED_EVAC_TURN_LIMIT, DOOMED_MIN_SHIPS,
    LATE_CAPTURE_BUFFER, VERY_LATE_CAPTURE_BUFFER,
    FOLLOWUP_MIN_SHIPS, COMET_MAX_CHASE_TURNS, LOW_VALUE_COMET_PRODUCTION,
    ATTACK_COST_TURN_WEIGHT, REINFORCE_SAFETY_MARGIN,
)


# ---------------------------------------------------------------------------
# V8.5 constants (Fix #0..#5, 4p plans)
# ---------------------------------------------------------------------------

# Fix #0 — already-in-flight thresholds
ENEMY_PARTIAL_COMMITMENT_RATIO = 0.5

# Fix #1 — garrison floor multipliers (ships per production unit)
GARRISON_FLOOR_NEAR_DIST = 20.0
GARRISON_FLOOR_MID_DIST = 35.0
GARRISON_FLOOR_NEAR_MULT = 8
GARRISON_FLOOR_MID_MULT = 5
GARRISON_FLOOR_FAR_MULT = 3
GARRISON_FLOOR_MIN_ABSOLUTE = 5
GARRISON_FLOOR_4P_EARLY_RATIO = 0.55  # halve floor before t60 in 4p so expansion isn't strangled

# Fix #2 — garrison ratio cap
GARRISON_FLOOR_RATIO = 0.32        # at least 32% of total ships on planets

# Fix #3 — threat model
THREAT_LATENT_HORIZON = 12         # turns ahead to estimate latent enemy strikes
THREAT_LATENT_DISCOUNT = 0.30      # weight applied to top enemy reachable
THREAT_LATENT_TOP_K = 1            # only the strongest reachable enemy contributes

# Fix #4 — 4p early expansion overrides
FOUR_P_EARLY_TURN_LIMIT = 60
FOUR_P_EARLY_NEUTRAL_MARGIN = 1
FOUR_P_REAR_STAGING_MIN_TURN = 60

# Fix #5 — 4p offense throttle
ACTIVE_FRONT_THRESHOLD = 2
THROTTLE_GARRISON_RATIO = 0.28
THROTTLE_TOP2_PRESSURE_RATIO = 0.75
THROTTLE_RELEASE_TURNS = 8         # consecutive low-pressure turns to exit throttle

# 4p plan triggers
LATE_BLITZ_TURN = 280
LATE_BLITZ_EARLY_TURN = 90       # expose the finisher inside 120-turn ES games
LATE_BLITZ_MIN_GARRISON_RATIO = 0.36
LATE_BLITZ_MAX_ACTIVE_FRONTS = 1
LATE_BLITZ_MIN_SHIP_LEAD = 1.05
LATE_BLITZ_MIN_PROD_LEAD = 1.05
ELIMINATE_WEAKEST_FRACTION = 0.40  # weakest enemy < this fraction of my ships
OPPORTUNISTIC_DEPLETED_PROD_MULT = 2.25  # target.ships < production * this

# Ranker dimensions
N_STATE_FEATURES = 24
N_CANDIDATE_FEATURES = 14
N_PLANS_MAX = 14                   # baseline + attack + expand + defense + reserve + transfer + probe + 5 4p variants + 2 opening variants
RANKER_TEMPERATURE = 1.0

# State feature indices.
STATE_MY_SHIP_SHARE = 0
STATE_ENEMY_SHIP_SHARE = 1
STATE_MY_PROD_SHARE = 2
STATE_ENEMY_PROD_SHARE = 3
STATE_MY_PLANET_SHARE = 4
STATE_ENEMY_PLANET_SHARE = 5
STATE_NEUTRAL_PLANET_SHARE = 6
STATE_STEP_FRACTION = 7
STATE_REMAINING_FRACTION = 8
STATE_IS_4P = 9
STATE_IS_OPENING = 10
STATE_IS_LATE = 11
STATE_ACTIVE_FRONTS = 12
STATE_WEAKEST_FRAC = 13
STATE_STRONGEST_FRAC = 14
STATE_CONTESTED_RATIO = 15
STATE_INTER_ENEMY_RATIO = 16
STATE_GARRISON_RATIO = 17
STATE_MIN_GARRISON_RATIO = 18
STATE_MAX_ENEMY_STRENGTH = 19
STATE_DOOMED_RATIO = 20
STATE_THREATENED_RATIO = 21
STATE_FLEET_DENSITY = 22
STATE_BIAS = 23

# Candidate feature indices.
CAND_N_MOVES = 0
CAND_SHIP_FRAC = 1
CAND_ATTACK_FRAC = 2
CAND_EXPAND_FRAC = 3
CAND_DEFENSE_FRAC = 4
CAND_AVG_ETA = 5
CAND_PLAN_IDX = 6
CAND_IS_BASELINE = 7
CAND_IS_4P = 8
CAND_IS_DEFENSE = 9
CAND_IS_ATTACK = 10
CAND_TRANSFER_FRAC = 11
CAND_IS_TRANSFER = 12
CAND_BIAS = 13
EARLY_4P_EXPAND_PLANET_TARGET = 6
EARLY_4P_EXPAND_OVERRIDE_TURN = 70
EARLY_4P_EXPAND_MIN_GARRISON_RATIO = 0.28
OPENING_FORTIFY_TURN_LIMIT = 48
OPENING_FORTIFY_MIN_GARRISON_RATIO = 0.36
OPENING_FORTIFY_MAX_ACTIVE_FRONTS = 1
OPENING_RELAY_TURN_LIMIT = 42
OPENING_RELAY_MIN_PLANETS = 3
OPENING_RELAY_MIN_GARRISON_RATIO = 0.34
OPENING_RELAY_MIN_TRANSFER_FRAC = 0.06
OPENING_RELAY_MAX_ACTIVE_FRONTS = 1
PROBE_MIN_SHIPS = 2
PROBE_MAX_SHIPS = 16
PROBE_TARGET_SHIPS_MULT = 2.25
PROBE_MIN_PROD = 1
PROBE_MIN_GARRISON_RATIO = 0.22
PROBE_MIN_SOURCE_AVAILABLE = 10
PROBE_SCORE_BONUS = 1.20
CONVERSION_PUSH_MIN_TURN = 60
CONVERSION_PUSH_MIN_GARRISON_RATIO = 0.30
CONVERSION_PUSH_MAX_ACTIVE_FRONTS = 2
CONVERSION_PUSH_MIN_SHIP_LEAD = 1.03
CONVERSION_PUSH_MIN_PROD_LEAD = 1.03

# Friendly staging transfers. Kovi-style strong bots use friendly moves heavily
# to consolidate rear production into a small number of launch platforms.
TRANSFER_SOURCE_MIN_SHIPS = 14
TRANSFER_DEST_MIN_PRODUCTION = 1
TRANSFER_MAX_TRAVEL_TURNS = 42
TRANSFER_FRONT_PROGRESS = 0.88
TRANSFER_REAR_DISTANCE_RATIO = 1.12
TRANSFER_BASE_SEND_RATIO = 0.55
TRANSFER_PUSH_SEND_RATIO = 0.78
TRANSFER_MIN_SEND_SHIPS = 8
TRANSFER_FRONT_STOCK_PROD_MULT = 15
TRANSFER_FRONT_STOCK_ABS = 35
TRANSFER_ACTIVE_FRONT_BONUS = 1.35
TRANSFER_PLAN_BONUS = 1.45
TRANSFER_OVERRIDE_MIN_STEP = 25
TRANSFER_OVERRIDE_MIN_FRAC = 0.08
TRANSFER_OVERRIDE_MAX_ACTIVE_FRONTS = 3


# ---------------------------------------------------------------------------
# Module-level mutable state for ranker
# ---------------------------------------------------------------------------

_STATE_W = np.zeros((N_PLANS_MAX, N_STATE_FEATURES), dtype=np.float32)
_CAND_W = np.zeros(N_CANDIDATE_FEATURES, dtype=np.float32)
_VALUE_W = np.zeros(N_STATE_FEATURES, dtype=np.float32)
_VALUE_B = np.float32(0.0)
_CANDIDATE_LOG_CB: Optional[Callable[[dict], None]] = None
_THROTTLE_RELEASE_COUNTER = 0


# Plan id -> human label (also drives the ranker rows)
PLAN_NAMES = (
    "v7_baseline",          # 0
    "expand_focus",         # 1
    "attack_focus",         # 2
    "defense_focus",        # 3
    "reserve_hold",         # 4
    "transfer_push",        # 5  (friendly staging emphasis)
    "probe_focus",          # 6  (harass borderline enemy targets)
    "4p_opportunistic",     # 7  (4p only)
    "4p_eliminate_weakest", # 8  (4p only)
    "4p_conservation",      # 9  (4p only)
    "4p_late_blitz",        # 10 (4p only, t > LATE_BLITZ_TURN)
    "4p_conversion_push",   # 11 (4p only, favorable midgame conversion)
    "opening_fortify",      # 12 (opening safety override)
    "opening_relay",        # 13 (opening ship-exchange / staging override)
)


# ---------------------------------------------------------------------------
# Public training hooks
# ---------------------------------------------------------------------------

def set_ranker_weights(state_w=None, candidate_w=None, value_w=None, value_b=None):
    """Override ranker weights. Pass None to leave unchanged."""
    global _STATE_W, _CAND_W, _VALUE_W, _VALUE_B
    if state_w is not None:
        arr = np.asarray(state_w, dtype=np.float32)
        if arr.shape != (N_PLANS_MAX, N_STATE_FEATURES):
            raise ValueError(f"state_w shape {arr.shape} != {(N_PLANS_MAX, N_STATE_FEATURES)}")
        _STATE_W = arr.copy()
    if candidate_w is not None:
        arr = np.asarray(candidate_w, dtype=np.float32).ravel()
        if arr.size != N_CANDIDATE_FEATURES:
            raise ValueError(f"candidate_w size {arr.size} != {N_CANDIDATE_FEATURES}")
        _CAND_W = arr.copy()
    if value_w is not None:
        arr = np.asarray(value_w, dtype=np.float32).ravel()
        if arr.size != N_STATE_FEATURES:
            raise ValueError(f"value_w size {arr.size} != {N_STATE_FEATURES}")
        _VALUE_W = arr.copy()
    if value_b is not None:
        _VALUE_B = np.float32(value_b)


def get_ranker_weights():
    return (_STATE_W.copy(), _CAND_W.copy(), _VALUE_W.copy(), float(_VALUE_B))


def set_candidate_log_callback(cb):
    global _CANDIDATE_LOG_CB
    _CANDIDATE_LOG_CB = cb


def reset_throttle_state():
    global _THROTTLE_RELEASE_COUNTER
    _THROTTLE_RELEASE_COUNTER = 0


# ---------------------------------------------------------------------------
# Fix #0 — already-in-flight check
# ---------------------------------------------------------------------------

def my_ships_en_route_to(target_id: int, world: WorldModel) -> int:
    """Sum of own ships already in flight toward this target."""
    total = 0
    for entry in world.arrivals_by_planet.get(target_id, []):
        eta, owner, ships = entry
        if owner == world.player and ships > 0:
            total += int(ships)
    return total


def already_committed_enough(target_id: int, eta: int, world: WorldModel,
                              planned_commitments) -> bool:
    """True if current in-flight (+ planned) is enough — caller should skip new commit."""
    target = world.planet_by_id[target_id]
    in_flight = my_ships_en_route_to(target_id, world)
    if in_flight <= 0:
        return False
    needed = world.ships_needed_to_capture(target_id, eta, planned_commitments)
    if needed <= 0:
        return True
    if in_flight >= needed:
        return True
    if target.owner not in (-1, world.player) and in_flight > needed * ENEMY_PARTIAL_COMMITMENT_RATIO:
        # Enemy target with substantial commitment already — wait for resolution
        return True
    return False


# ---------------------------------------------------------------------------
# Fix #1 — garrison floor
# ---------------------------------------------------------------------------

def nearest_enemy_dist(world: WorldModel, planet) -> float:
    if not world.enemy_planets:
        return 1e6
    return min(_dist(planet.x, planet.y, e.x, e.y) for e in world.enemy_planets)


def garrison_floor(planet, world: WorldModel) -> int:
    prod = max(1, int(planet.production))
    d = nearest_enemy_dist(world, planet)
    if d < GARRISON_FLOOR_NEAR_DIST:
        base = prod * GARRISON_FLOOR_NEAR_MULT
    elif d < GARRISON_FLOOR_MID_DIST:
        base = prod * GARRISON_FLOOR_MID_MULT
    else:
        base = prod * GARRISON_FLOOR_FAR_MULT
    if world.is_four_player and world.step < FOUR_P_EARLY_TURN_LIMIT:
        base = int(base * GARRISON_FLOOR_4P_EARLY_RATIO)
    return max(GARRISON_FLOOR_MIN_ABSOLUTE, base)


# ---------------------------------------------------------------------------
# Fix #3 — latent threat per planet
# ---------------------------------------------------------------------------

def latent_threat(planet, world: WorldModel) -> int:
    """Estimated extra enemy ships that could reach this planet within horizon.

    Uses only the top-K reachable enemy planet(s) instead of summing the whole
    enemy roster — otherwise in 4p we add three rosters and lock garrisons so
    high that expansion stalls.
    """
    if not world.enemy_planets:
        return 0
    candidates = []
    for e in world.enemy_planets:
        if e.ships <= 0:
            continue
        eta = _travel_time(e.x, e.y, e.radius, planet.x, planet.y, planet.radius, max(1, int(e.ships)))
        if eta > THREAT_LATENT_HORIZON:
            continue
        candidates.append(int(e.ships))
    if not candidates:
        return 0
    candidates.sort(reverse=True)
    top = candidates[:max(1, THREAT_LATENT_TOP_K)]
    return int(sum(top) * THREAT_LATENT_DISCOUNT)


# ---------------------------------------------------------------------------
# Active fronts & garrison ratio (state-level signals)
# ---------------------------------------------------------------------------

def compute_active_fronts(world: WorldModel) -> Dict[int, int]:
    """{enemy_id: count_of_my_planets_targeted} for enemies currently inbound."""
    fronts: Dict[int, set] = defaultdict(set)
    for planet in world.my_planets:
        for entry in world.arrivals_by_planet.get(planet.id, []):
            eta, owner, ships = entry
            if owner not in (-1, world.player) and ships > 0:
                fronts[owner].add(planet.id)
    return {oid: len(s) for oid, s in fronts.items()}


def garrison_ratio_now(world: WorldModel) -> float:
    if world.my_total <= 0:
        return 1.0
    on_planets = sum(int(p.ships) for p in world.my_planets)
    return on_planets / max(1, world.my_total)


# ---------------------------------------------------------------------------
# State features (24-D) — used by ranker + value head
# ---------------------------------------------------------------------------

def build_state_features(world: WorldModel) -> np.ndarray:
    n_my = max(1, len(world.my_planets))
    n_enemy = max(1, len(world.enemy_planets))
    total_planets = max(1, n_my + n_enemy + len(world.neutral_planets))
    total_ships = max(1, world.my_total + world.enemy_total)
    total_prod = max(1, world.my_prod + world.enemy_prod)

    fronts = compute_active_fronts(world)
    n_active_fronts = len(fronts)
    n_contested = sum(1 for s in fronts.values() if s > 0)

    weakest_strength = 0
    strongest_strength = 0
    if world.enemy_planets:
        strengths = [s for o, s in world.owner_strength.items() if o not in (-1, world.player)]
        if strengths:
            weakest_strength = min(strengths)
            strongest_strength = max(strengths)

    weakest_frac = weakest_strength / max(1, world.my_total)
    strongest_frac = strongest_strength / max(1, world.my_total)

    g_ratio = garrison_ratio_now(world)
    avg_prod = world.my_prod / n_my
    if world.my_planets:
        min_garr = min(int(p.ships) for p in world.my_planets)
        min_garr_norm = min_garr / max(1, avg_prod)
    else:
        min_garr_norm = 0.0

    inter_enemy = 0
    for f in world.fleets:
        if f.owner in (-1, world.player):
            continue
        # Targeting any non-player non-neutral planet?
        for p in world.planets:
            if p.owner not in (-1, world.player, f.owner):
                inter_enemy += int(f.ships)
                break

    feat = np.array([
        world.my_total / total_ships,                              # STATE_MY_SHIP_SHARE
        world.enemy_total / total_ships,                           # STATE_ENEMY_SHIP_SHARE
        world.my_prod / total_prod,                                # STATE_MY_PROD_SHARE
        world.enemy_prod / total_prod,                             # STATE_ENEMY_PROD_SHARE
        n_my / total_planets,                                      # STATE_MY_PLANET_SHARE
        n_enemy / total_planets,                                   # STATE_ENEMY_PLANET_SHARE
        len(world.neutral_planets) / total_planets,                # STATE_NEUTRAL_PLANET_SHARE
        world.step / 500.0,                                        # STATE_STEP_FRACTION
        world.remaining_steps / 500.0,                             # STATE_REMAINING_FRACTION
        1.0 if world.is_four_player else 0.0,                      # STATE_IS_4P
        1.0 if world.is_opening else 0.0,                          # STATE_IS_OPENING
        1.0 if world.is_late else 0.0,                             # STATE_IS_LATE
        n_active_fronts / 3.0,                                     # STATE_ACTIVE_FRONTS
        weakest_frac,                                              # STATE_WEAKEST_FRAC
        strongest_frac,                                            # STATE_STRONGEST_FRAC
        n_contested / max(1, n_my),                                # STATE_CONTESTED_RATIO
        inter_enemy / max(1, total_ships),                         # STATE_INTER_ENEMY_RATIO
        g_ratio,                                                   # STATE_GARRISON_RATIO
        min(min_garr_norm / 10.0, 2.0),                            # STATE_MIN_GARRISON_RATIO
        min(world.max_enemy_strength / max(1, world.my_total), 3.0),# STATE_MAX_ENEMY_STRENGTH
        len(world.doomed_candidates) / max(1, n_my),               # STATE_DOOMED_RATIO
        len(world.threatened_candidates) / max(1, n_my),           # STATE_THREATENED_RATIO
        min(len(world.fleets) / 30.0, 2.0),                        # STATE_FLEET_DENSITY
        1.0,                                                       # STATE_BIAS
    ], dtype=np.float32)
    return feat


# ---------------------------------------------------------------------------
# V7-equivalent plan generator (used as candidate 0 + as workhorse for variants)
# ---------------------------------------------------------------------------

# We re-implement plan_moves locally so we can inject:
#   - Fix #0: skip targets already in-flight
#   - Fix #1+#3: keep_needed = max(timeline, garrison_floor, latent_threat)
#   - Fix #2: garrison_ratio cap before final emit
#   - Fix #4: 4p early expansion overrides
#   - Fix #5: 4p offense throttle
#   - Variant biases (attack/expand/defense/reserve, 4p_*)
# Code mirrors v7.plan_moves but with hooks. Kept compact.

@dataclass
class PlanContext:
    variant: str = "v7_baseline"
    attack_boost: float = 1.0
    expand_boost: float = 1.0
    defense_boost: float = 1.0
    probe_boost: float = 1.0
    only_targets_owner: Optional[int] = None      # restrict offensive targets to this owner
    only_depleted_targets: bool = False           # opportunistic
    block_offense: bool = False                   # conservation
    suppress_garrison_floor: bool = False         # late_blitz
    suppress_garrison_ratio_cap: bool = False
    suppress_rear_staging: bool = False
    transfer_boost: float = 1.0
    force_transfer_staging: bool = False


def _adjust_keep_needed(world: WorldModel, planet, ctx: PlanContext, base_keep: int) -> int:
    if ctx.suppress_garrison_floor:
        return base_keep
    floor = garrison_floor(planet, world)
    latent = latent_threat(planet, world)
    return max(base_keep, floor, latent)


def _allow_4p_late_blitz(world: WorldModel, n_fronts: int, g_ratio: float) -> bool:
    """Late blitz stays late by default, but can activate earlier in a clearly favorable 4p endgame."""
    if world.step > LATE_BLITZ_TURN:
        return True
    if world.step < LATE_BLITZ_EARLY_TURN:
        return False
    if n_fronts > LATE_BLITZ_MAX_ACTIVE_FRONTS:
        return False
    if g_ratio < LATE_BLITZ_MIN_GARRISON_RATIO:
        return False
    strongest_enemy = max(
        (s for owner, s in world.owner_strength.items() if owner not in (-1, world.player)),
        default=0,
    )
    strongest_enemy_prod = max(
        (p for owner, p in world.owner_production.items() if owner not in (-1, world.player)),
        default=0,
    )
    ship_lead = world.my_total / max(1, strongest_enemy)
    prod_lead = world.my_prod / max(1, strongest_enemy_prod)
    return ship_lead >= LATE_BLITZ_MIN_SHIP_LEAD or prod_lead >= LATE_BLITZ_MIN_PROD_LEAD


def _aim_with_prediction_v8_5(src, target, ships, initial_by_id, ang_vel, comets, comet_ids):
    """More robust intercept solve than the V7 helper.

    We probe a small neighborhood around the initial ETA and use a damped
    fixed-point update so one-turn oscillations do not lock in a bad aim.
    """
    seed = _estimate_arrival(src.x, src.y, src.radius, target.x, target.y, target.radius, ships)
    if seed is None:
        return _search_safe_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids)

    base_turns = int(seed[1])
    best = None
    for offset in range(-2, 3):
        turns = max(1, base_turns + offset)
        prev_pos = None
        for _ in range(AIM_ITERATIONS + 3):
            pos = _predict_target_pos(target, turns, initial_by_id, ang_vel, comets, comet_ids)
            if pos is None:
                break
            nxt = _estimate_arrival(src.x, src.y, src.radius, pos[0], pos[1], target.radius, ships)
            if nxt is None:
                break
            angle, next_turns = nxt
            residual = abs(next_turns - turns)
            drift = 0.0 if prev_pos is None else math.hypot(pos[0] - prev_pos[0], pos[1] - prev_pos[1])
            score = (residual, drift, next_turns, abs(offset))
            if best is None or score < best[0]:
                best = (score, angle, next_turns, pos[0], pos[1])
            if residual <= INTERCEPT_TOLERANCE:
                confirm_pos = _predict_target_pos(target, next_turns, initial_by_id, ang_vel, comets, comet_ids)
                if confirm_pos is not None:
                    confirm = _estimate_arrival(src.x, src.y, src.radius, confirm_pos[0], confirm_pos[1], target.radius, ships)
                    if confirm is not None and abs(confirm[1] - next_turns) <= INTERCEPT_TOLERANCE:
                        return confirm[0], confirm[1], confirm_pos[0], confirm_pos[1]
            damped = int(round(0.65 * turns + 0.35 * next_turns))
            if damped == turns:
                prev_pos = pos
                break
            prev_pos = pos
            turns = max(1, damped)

    if best is not None:
        _, angle, turns, px, py = best
        return angle, turns, px, py
    return _search_safe_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids)


def _plan_shot_v8_5(world: WorldModel, src_id: int, target_id: int, ships: int):
    bucket = max(1, int(ships))
    if bucket <= 8:
        key_ships = bucket
    elif bucket <= 32:
        key_ships = (bucket // 4) * 4
    elif bucket <= 128:
        key_ships = (bucket // 8) * 8
    else:
        key_ships = (bucket // 16) * 16
    cache_key = (src_id, target_id, key_ships)
    cached = world.aim_cache.get(cache_key)
    if cached is not None:
        return cached if cached != "MISS" else None
    src = world.planet_by_id[src_id]
    target = world.planet_by_id[target_id]
    result = _aim_with_prediction_v8_5(src, target, ships, world.initial_by_id,
                                        world.ang_vel, world.comets, world.comet_ids)
    world.aim_cache[cache_key] = result if result is not None else "MISS"
    return result


def _conversion_push_owner(world: WorldModel, n_fronts: Optional[int] = None,
                           g_ratio: Optional[float] = None) -> Optional[int]:
    """Return the enemy owner to pressure when a stable 4p lead should convert."""
    if not world.is_four_player or world.step < CONVERSION_PUSH_MIN_TURN:
        return None
    if n_fronts is None:
        n_fronts = len(compute_active_fronts(world))
    if g_ratio is None:
        g_ratio = garrison_ratio_now(world)
    if n_fronts > CONVERSION_PUSH_MAX_ACTIVE_FRONTS or g_ratio < CONVERSION_PUSH_MIN_GARRISON_RATIO:
        return None

    enemy_strengths = [
        (owner, strength)
        for owner, strength in world.owner_strength.items()
        if owner not in (-1, world.player)
    ]
    if not enemy_strengths:
        return None
    strongest_owner, strongest_strength = max(enemy_strengths, key=lambda item: item[1])
    strongest_prod = max(
        (prod for owner, prod in world.owner_production.items() if owner not in (-1, world.player)),
        default=0,
    )
    ship_lead = world.my_total / max(1, strongest_strength)
    prod_lead = world.my_prod / max(1, strongest_prod)
    max_enemy_planets = max(
        (sum(1 for p in world.planets if p.owner == owner) for owner, _ in enemy_strengths),
        default=0,
    )
    planet_lead = len(world.my_planets) - max_enemy_planets
    if (ship_lead >= CONVERSION_PUSH_MIN_SHIP_LEAD
            or prod_lead >= CONVERSION_PUSH_MIN_PROD_LEAD
            or planet_lead >= 2):
        return int(strongest_owner)
    return None


def _has_4p_opportunistic_target(world: WorldModel) -> bool:
    """Only expose opportunistic mode when there is an actual depleted contested target."""
    for target in world.planets:
        if target.owner == world.player:
            continue
        is_depleted = target.ships < target.production * OPPORTUNISTIC_DEPLETED_PROD_MULT
        if not is_depleted:
            continue
        has_enemy_inbound = any(
            owner not in (-1, world.player)
            for _, owner, ships in world.arrivals_by_planet.get(target.id, [])
            if ships > 0
        )
        if target.owner not in (-1, world.player) or has_enemy_inbound:
            return True
    return False


def _is_probe_target(target, world: WorldModel) -> bool:
    if target.owner == world.player:
        return False
    if target.owner == -1:
        # Prefer contested neutrals over soft neutrals.
        return _is_contested_neutral(target, world)
    if target.production < PROBE_MIN_PROD:
        return False
    if int(target.ships) > int(target.production * PROBE_TARGET_SHIPS_MULT):
        return False
    if target.id in world.comet_ids:
        return False
    return True


def _probe_score(target, turns: int, send: int, world: WorldModel, modes, probe_boost: float = 1.0) -> float:
    base_value = _target_value(target, turns, "capture", world, modes)
    if base_value <= 0:
        return -1.0
    if target.owner == -1:
        base_value *= 0.55
    else:
        base_value *= 0.85
    if target.id in world.doomed_candidates:
        base_value *= 1.05
    # Probes are about pressure and information, so small, fast sends are preferred.
    return _apply_score_mods(
        (base_value * PROBE_SCORE_BONUS * probe_boost) / (send + turns * 0.25 + 1.0),
        target, "snipe", world,
    )


def _build_probe_missions(world, planned_commitments, modes, source_budget_fn, probe_boost: float = 1.0):
    missions = []
    for src in world.my_planets:
        budget = source_budget_fn(src.id)
        if budget < PROBE_MIN_SHIPS:
            continue
        src_best = None
        src_available = budget
        for target in world.planets:
            if target.id == src.id or not _is_probe_target(target, world):
                continue
            if already_committed_enough(target.id, 1, world, planned_commitments):
                continue
            probe_cap = min(PROBE_MAX_SHIPS, max(PROBE_MIN_SHIPS, int(target.production * 2), int(src_available * 0.15)))
            if probe_cap < PROBE_MIN_SHIPS:
                continue
            rough_aim = _plan_shot_v8_5(world, src.id, target.id, probe_cap)
            if rough_aim is None:
                continue
            turns = rough_aim[1]
            if turns > max(18, world.remaining_steps - 4):
                continue
            if target.owner not in (-1, world.player):
                # Only probe enemies if we can keep the send small.
                if int(target.ships) > int(target.production * PROBE_TARGET_SHIPS_MULT):
                    continue
            send = min(src_available, probe_cap)
            if send < PROBE_MIN_SHIPS:
                continue
            score = _probe_score(target, turns, send, world, modes, probe_boost=probe_boost)
            if score <= 0:
                continue
            option = ShotOption(score=score, src_id=src.id, target_id=target.id,
                                angle=rough_aim[0], turns=turns, needed=send,
                                send_cap=send, mission="probe")
            if src_best is None or score > src_best[0]:
                src_best = (score, target.id, turns, option)
        if src_best is not None:
            score, target_id, turns, option = src_best
            missions.append(Mission(kind="probe", score=score, target_id=target_id,
                                    turns=turns, options=[option]))
    return missions


def _transfer_frontier_targets(world: WorldModel) -> List[Planet]:
    return world.enemy_planets or world.static_neutral_planets or world.neutral_planets


def _frontier_distance_map(world: WorldModel, targets: Sequence[Planet]) -> Dict[int, float]:
    if not targets:
        return {p.id: 1e9 for p in world.my_planets}
    return {
        p.id: min(_dist(p.x, p.y, t.x, t.y) for t in targets)
        for p in world.my_planets
    }


def _emit_transfer_staging(world: WorldModel, ctx: PlanContext, atk_left, append_move) -> None:
    """Move safe rear ships to friendly launch platforms near the action."""
    if ctx.suppress_rear_staging or world.is_late or len(world.my_planets) < 2:
        return
    if world.is_four_player and world.step < FOUR_P_REAR_STAGING_MIN_TURN and not ctx.force_transfer_staging:
        return

    frontier_targets = _transfer_frontier_targets(world)
    if not frontier_targets:
        return

    frontier_distance = _frontier_distance_map(world, frontier_targets)
    safe_planets = [p for p in world.my_planets if p.id not in world.doomed_candidates]
    if len(safe_planets) < 2:
        return

    active_front_ids = {p.id for p in compute_active_fronts(world)}
    destinations = [
        p for p in safe_planets
        if p.production >= TRANSFER_DEST_MIN_PRODUCTION
    ]
    if not destinations:
        return

    send_ratio = TRANSFER_PUSH_SEND_RATIO if ctx.force_transfer_staging else TRANSFER_BASE_SEND_RATIO
    send_ratio = min(0.92, send_ratio * max(0.2, ctx.transfer_boost))

    for src in sorted(safe_planets, key=lambda p: -frontier_distance[p.id]):
        available = atk_left(src.id)
        if available < TRANSFER_SOURCE_MIN_SHIPS:
            continue

        src_frontier_dist = frontier_distance[src.id]
        best = None
        for dst in destinations:
            if dst.id == src.id:
                continue
            dst_frontier_dist = frontier_distance[dst.id]
            if dst_frontier_dist >= src_frontier_dist * TRANSFER_FRONT_PROGRESS:
                continue
            if src_frontier_dist < dst_frontier_dist * TRANSFER_REAR_DISTANCE_RATIO:
                continue

            stock_limit = max(TRANSFER_FRONT_STOCK_ABS, int(dst.production * TRANSFER_FRONT_STOCK_PROD_MULT))
            if int(dst.ships) >= stock_limit and dst.id not in active_front_ids and not ctx.force_transfer_staging:
                continue

            probe_send = max(TRANSFER_MIN_SEND_SHIPS, int(available * send_ratio))
            aim = _plan_shot_v8_5(world, src.id, dst.id, probe_send)
            if aim is None:
                continue
            angle, turns, _, _ = aim
            if turns > TRANSFER_MAX_TRAVEL_TURNS:
                continue

            progress = max(0.0, src_frontier_dist - dst_frontier_dist)
            stock_need = max(0, stock_limit - int(dst.ships))
            active_bonus = TRANSFER_ACTIVE_FRONT_BONUS if dst.id in active_front_ids else 1.0
            score = (progress / max(1.0, turns)) * active_bonus
            score += min(1.5, stock_need / max(1.0, stock_limit))
            if ctx.force_transfer_staging:
                score *= TRANSFER_PLAN_BONUS
            if best is None or score > best[0]:
                best = (score, dst, angle, turns)

        if best is None:
            continue
        _, dst, angle, _turns = best
        stock_limit = max(TRANSFER_FRONT_STOCK_ABS, int(dst.production * TRANSFER_FRONT_STOCK_PROD_MULT))
        stock_need = max(TRANSFER_MIN_SEND_SHIPS, stock_limit - int(dst.ships))
        stock_send = min(stock_need, int(available * 0.85))
        send = min(available, max(TRANSFER_MIN_SEND_SHIPS, int(available * send_ratio), stock_send))
        if send < TRANSFER_MIN_SEND_SHIPS:
            continue
        append_move(src.id, angle, send)


def _v8_5_plan(world: WorldModel, ctx: PlanContext) -> Tuple[List[List], List[int]]:
    """Returns (moves, sources_ships_sent). Pure heuristic plan with V8.5 fixes + variant bias."""
    modes = _build_modes(world)

    # Fix #4: 4p early expansion override (mutates modes via attack margin trick + adjusted preferred_send)
    is_4p_early = world.is_four_player and world.step < FOUR_P_EARLY_TURN_LIMIT

    # Fix #5: throttle decision
    fronts = compute_active_fronts(world)
    n_fronts = len(fronts)
    enemy_strengths = sorted((s for o, s in world.owner_strength.items() if o not in (-1, world.player)), reverse=True)
    top2_pressure = sum(enemy_strengths[:2])
    g_ratio = garrison_ratio_now(world)
    throttle = (
        world.is_four_player
        and (
            (n_fronts >= ACTIVE_FRONT_THRESHOLD and g_ratio < THROTTLE_GARRISON_RATIO)
            or (n_fronts >= ACTIVE_FRONT_THRESHOLD and top2_pressure > world.my_total * THROTTLE_TOP2_PRESSURE_RATIO)
        )
    )
    if ctx.block_offense:
        throttle = True

    planned_commitments = defaultdict(list)
    source_options_by_target = defaultdict(list)
    missions: List[Mission] = []
    moves: List[List] = []
    spent_total: Dict[int, int] = defaultdict(int)

    # V8.5 reserves: rebuild with adjusted keep_needed (Fix #1+#3).
    adjusted_reserve = {}
    for p in world.my_planets:
        base_keep = world.reserve.get(p.id, 0)
        adjusted_reserve[p.id] = min(int(p.ships), _adjust_keep_needed(world, p, ctx, base_keep))

    def inv_left(sid: int) -> int:
        src = world.planet_by_id[sid]
        return max(0, int(src.ships) - spent_total[sid])

    def atk_left(sid: int) -> int:
        return max(0, inv_left(sid) - adjusted_reserve.get(sid, 0))

    def append_move(sid: int, angle: float, ships: int) -> int:
        send = min(int(ships), inv_left(sid))
        if send < 1:
            return 0
        moves.append([sid, float(angle), int(send)])
        spent_total[sid] += send
        return send

    # Reinforcements (always run — defense baseline)
    missions.extend(_build_reinforcement_missions(world, planned_commitments, modes, inv_left))

    # Probe harassment (new first-class primitive)
    if ctx.probe_boost != 1.0 or ctx.variant == "probe_focus":
        missions.extend(_build_probe_missions(world, planned_commitments, modes, atk_left, probe_boost=ctx.probe_boost))

    # Offensive candidates (skip if throttled)
    if not throttle:
        for src in world.my_planets:
            src_available = atk_left(src.id)
            if src_available <= 0:
                continue
            for target in world.planets:
                if target.id == src.id or target.owner == world.player:
                    continue

                # Variant filters
                if ctx.only_targets_owner is not None:
                    # eliminate_weakest mode: only attack the weakest enemy (and neutrals when nothing close)
                    if target.owner not in (-1, ctx.only_targets_owner):
                        continue
                if ctx.only_depleted_targets:
                    # opportunistic: only truly depleted planets that are contested
                    # or already changed hands. This avoids chasing soft neutrals.
                    is_depleted = target.ships < target.production * OPPORTUNISTIC_DEPLETED_PROD_MULT
                    has_enemy_inbound = any(o not in (-1, world.player)
                                            for _, o, _ in world.arrivals_by_planet.get(target.id, []))
                    recently_fought = target.owner not in (-1, world.player) or has_enemy_inbound
                    if not (is_depleted and recently_fought):
                        continue

                rough_ships = max(1, min(src_available, max(PARTIAL_SOURCE_MIN_SHIPS, int(target.ships) + 1)))
                rough_aim = _plan_shot_v8_5(world, src.id, target.id, rough_ships)
                if rough_aim is None:
                    continue
                rough_turns = rough_aim[1]
                if world.is_very_late and rough_turns > world.remaining_steps - VERY_LATE_CAPTURE_BUFFER:
                    continue
                if target.id in world.comet_ids:
                    life = world.comet_life(target.id)
                    if rough_turns >= life or rough_turns > COMET_MAX_CHASE_TURNS:
                        continue

                # Fix #0 — skip if already enough in flight from previous turns
                if already_committed_enough(target.id, rough_turns, world, planned_commitments):
                    continue

                rough_needed = world.ships_needed_to_capture(target.id, rough_turns, planned_commitments)
                if rough_needed <= 0:
                    continue
                if _opening_filter(target, rough_turns, rough_needed, src_available, world):
                    continue

                # Fix #4 — tighter neutral margin in 4p early game
                send_guess = _preferred_send(target, rough_needed, rough_turns, src_available, world, modes)
                if is_4p_early and target.owner == -1:
                    send_guess = max(rough_needed + FOUR_P_EARLY_NEUTRAL_MARGIN, rough_needed)
                    send_guess = min(send_guess, src_available)

                aim = _plan_shot_v8_5(world, src.id, target.id, max(1, send_guess))
                if aim is None:
                    continue
                angle, turns, _, _ = aim
                if world.is_very_late and turns > world.remaining_steps - VERY_LATE_CAPTURE_BUFFER:
                    continue
                if target.id in world.comet_ids:
                    life = world.comet_life(target.id)
                    if turns >= life or turns > COMET_MAX_CHASE_TURNS:
                        continue

                if already_committed_enough(target.id, turns, world, planned_commitments):
                    continue

                needed = world.ships_needed_to_capture(target.id, turns, planned_commitments)
                if needed <= 0:
                    continue
                if _opening_filter(target, turns, needed, src_available, world):
                    continue

                send_cap = min(src_available, _preferred_send(target, needed, turns, src_available, world, modes))
                if is_4p_early and target.owner == -1:
                    send_cap = min(src_available, needed + FOUR_P_EARLY_NEUTRAL_MARGIN)
                if send_cap < 1:
                    continue
                if send_cap < needed and send_cap < PARTIAL_SOURCE_MIN_SHIPS:
                    continue
                value = _target_value(target, turns, "capture", world, modes)
                if value <= 0:
                    continue
                # Variant biases
                if target.owner not in (-1, world.player):
                    value *= ctx.attack_boost
                elif target.owner == -1:
                    value *= ctx.expand_boost
                expected_send = max(needed, min(send_cap, _preferred_send(target, needed, turns, send_cap, world, modes)))
                score = _apply_score_mods(value / (expected_send + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                                          target, "capture", world)
                option = ShotOption(score=score, src_id=src.id, target_id=target.id,
                                    angle=angle, turns=turns, needed=needed, send_cap=send_cap, mission="capture")
                source_options_by_target[target.id].append(option)
                if send_cap >= needed:
                    missions.append(Mission(kind="single", score=score, target_id=target.id,
                                            turns=turns, options=[option]))
                snipe = _build_snipe_mission(src, target, src_available, world, planned_commitments, modes)
                if snipe is not None:
                    missions.append(snipe)

        # Multi-source swarms (re-uses V7 logic)
        for target_id, options in source_options_by_target.items():
            if len(options) < 2:
                continue
            target = world.planet_by_id[target_id]
            top = sorted(options, key=lambda o: -o.score)[:MULTI_SOURCE_TOP_K]
            hostile_target = target.owner not in (-1, world.player)
            eta_tol = HOSTILE_SWARM_ETA_TOLERANCE if hostile_target else MULTI_SOURCE_ETA_TOLERANCE
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    a, b = top[i], top[j]
                    if a.src_id == b.src_id or abs(a.turns - b.turns) > eta_tol:
                        continue
                    jt = max(a.turns, b.turns)
                    need = world.ships_needed_to_capture(target_id, jt, planned_commitments)
                    if need <= 0 or a.send_cap >= need or b.send_cap >= need:
                        continue
                    if a.send_cap + b.send_cap < need:
                        continue
                    value = _target_value(target, jt, "swarm", world, modes)
                    if value <= 0:
                        continue
                    if hostile_target:
                        value *= ctx.attack_boost
                    elif target.owner == -1:
                        value *= ctx.expand_boost
                    ps = _apply_score_mods(value / (need + jt * ATTACK_COST_TURN_WEIGHT + 1.0),
                                           target, "swarm", world) * MULTI_SOURCE_PLAN_PENALTY
                    missions.append(Mission(kind="swarm", score=ps, target_id=target_id, turns=jt, options=[a, b]))
            if THREE_SOURCE_SWARM_ENABLED and hostile_target and int(target.ships) >= THREE_SOURCE_MIN_TARGET_SHIPS and len(top) >= 3:
                limit = min(len(top), 8)
                for i in range(limit):
                    for j in range(i + 1, limit):
                        for k in range(j + 1, limit):
                            trio = [top[i], top[j], top[k]]
                            if len({o.src_id for o in trio}) < 3:
                                continue
                            if max(o.turns for o in trio) - min(o.turns for o in trio) > THREE_SOURCE_ETA_TOLERANCE:
                                continue
                            jt = max(o.turns for o in trio)
                            need = world.ships_needed_to_capture(target_id, jt, planned_commitments)
                            if need <= 0 or sum(o.send_cap for o in trio) < need:
                                continue
                            if any(trio[a].send_cap + trio[b].send_cap >= need for a in range(3) for b in range(a + 1, 3)):
                                continue
                            value = _target_value(target, jt, "swarm", world, modes)
                            if value <= 0:
                                continue
                            value *= ctx.attack_boost
                            ts = _apply_score_mods(value / (need + jt * ATTACK_COST_TURN_WEIGHT + 1.0),
                                                   target, "swarm", world) * THREE_SOURCE_PLAN_PENALTY
                            missions.append(Mission(kind="swarm", score=ts, target_id=target_id, turns=jt, options=trio))

    missions.sort(key=lambda m: -m.score)

    # Apply boost to defense missions (reinforce kind)
    if ctx.defense_boost != 1.0:
        for m in missions:
            if m.kind == "reinforce":
                m.score *= ctx.defense_boost
        missions.sort(key=lambda m: -m.score)

    # Mission emission (mirrors V7)
    for mission in missions:
        target = world.planet_by_id[mission.target_id]
        if mission.kind in ("single", "snipe", "reinforce", "probe"):
            option = mission.options[0]
            left = inv_left(option.src_id) if mission.kind == "reinforce" else atk_left(option.src_id)
            if left <= 0:
                continue
            if mission.kind == "reinforce":
                missing = world.reinforcement_needed_for(option.target_id, option.turns, planned_commitments)
            elif mission.kind == "probe":
                send = min(left, option.send_cap)
                if send < PROBE_MIN_SHIPS:
                    continue
                sent = append_move(option.src_id, option.angle, send)
                if sent < PROBE_MIN_SHIPS:
                    continue
                continue
            else:
                missing = world.ships_needed_to_capture(target.id, option.turns, planned_commitments)
            if missing <= 0:
                continue
            send_limit = min(left, option.send_cap)
            if send_limit < missing:
                continue
            if mission.kind == "snipe":
                send = missing
            elif mission.kind == "reinforce":
                send = min(send_limit, missing + REINFORCE_SAFETY_MARGIN)
            else:
                send = min(send_limit, max(missing, _preferred_send(target, missing, option.turns, send_limit, world, modes)))
            if send < missing:
                continue
            sent = append_move(option.src_id, option.angle, send)
            if sent < missing:
                continue
            planned_commitments[target.id].append((option.turns, world.player, int(sent)))
            continue

        # Swarm
        limits = [min(atk_left(o.src_id), o.send_cap) for o in mission.options]
        if min(limits) <= 0:
            continue
        missing = world.ships_needed_to_capture(target.id, mission.turns, planned_commitments)
        if missing <= 0 or sum(limits) < missing:
            continue
        ordered = sorted(zip(mission.options, limits), key=lambda x: (x[0].turns, -x[1], x[0].src_id))
        remaining = missing
        sends = {}
        for idx, (option, limit) in enumerate(ordered):
            remaining_other = sum(lim for _, lim in ordered[idx + 1:])
            send = min(limit, max(0, remaining - remaining_other))
            sends[option.src_id] = send
            remaining -= send
        if remaining > 0:
            continue
        committed = []
        for option, _ in ordered:
            send = sends.get(option.src_id, 0)
            if send <= 0:
                continue
            actual = append_move(option.src_id, option.angle, send)
            if actual > 0:
                committed.append((option.turns, world.player, int(actual)))
        if sum(item[2] for item in committed) < missing:
            continue
        planned_commitments[target.id].extend(committed)

    # Follow-up pass
    if not throttle and not world.is_very_late:
        for src in world.my_planets:
            src_left = atk_left(src.id)
            if src_left < FOLLOWUP_MIN_SHIPS:
                continue
            best = None
            for target in world.planets:
                if target.id == src.id or target.owner == world.player:
                    continue
                if target.id in world.comet_ids and target.production <= LOW_VALUE_COMET_PRODUCTION:
                    continue
                if ctx.only_targets_owner is not None and target.owner not in (-1, ctx.only_targets_owner):
                    continue
                rough_ships = max(1, min(src_left, max(PARTIAL_SOURCE_MIN_SHIPS, int(target.ships) + 1)))
                rough_aim = _plan_shot_v8_5(world, src.id, target.id, rough_ships)
                if rough_aim is None:
                    continue
                est_turns = rough_aim[1]
                if world.is_late and est_turns > world.remaining_steps - LATE_CAPTURE_BUFFER:
                    continue
                if already_committed_enough(target.id, est_turns, world, planned_commitments):
                    continue
                rough_needed = world.ships_needed_to_capture(target.id, est_turns, planned_commitments)
                if rough_needed <= 0:
                    continue
                if _opening_filter(target, est_turns, rough_needed, src_left, world):
                    continue
                send = _preferred_send(target, rough_needed, est_turns, src_left, world, modes)
                if send < rough_needed:
                    continue
                value = _target_value(target, est_turns, "capture", world, modes)
                if value <= 0:
                    continue
                if target.owner not in (-1, world.player):
                    value *= ctx.attack_boost
                else:
                    value *= ctx.expand_boost
                score = _apply_score_mods(value / (send + est_turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                                          target, "capture", world)
                if best is None or score > best[0]:
                    best = (score, target, send)
            if best is None:
                continue
            _, target, send = best
            aim = _plan_shot_v8_5(world, src.id, target.id, send)
            if aim is None:
                continue
            angle, turns, _, _ = aim
            missing = world.ships_needed_to_capture(target.id, turns, planned_commitments)
            if missing <= 0:
                continue
            if already_committed_enough(target.id, turns, world, planned_commitments):
                continue
            src_left = atk_left(src.id)
            send = min(src_left, max(missing, _preferred_send(target, missing, turns, src_left, world, modes)))
            if send < missing:
                continue
            actual = append_move(src.id, angle, send)
            if actual >= missing:
                planned_commitments[target.id].append((turns, world.player, int(actual)))

    # Doomed evac (always — never throttle survival)
    if world.doomed_candidates:
        frontier_targets = world.enemy_planets or world.static_neutral_planets or world.neutral_planets
        if frontier_targets:
            frontier_distance = {p.id: min(_dist(p.x, p.y, t.x, t.y) for t in frontier_targets) for p in world.my_planets}
        else:
            frontier_distance = {p.id: 1e9 for p in world.my_planets}
        for planet in world.my_planets:
            if planet.id not in world.doomed_candidates:
                continue
            available_now = inv_left(planet.id)
            if available_now < adjusted_reserve.get(planet.id, 0):
                continue
            best_capture = None
            for target in world.planets:
                if target.id == planet.id or target.owner == world.player:
                    continue
                probe_aim = _plan_shot_v8_5(world, planet.id, target.id, available_now)
                if probe_aim is None:
                    continue
                probe_turns = probe_aim[1]
                if probe_turns > world.remaining_steps - 2:
                    continue
                need = world.ships_needed_to_capture(target.id, probe_turns, planned_commitments)
                if need <= 0 or need > available_now:
                    continue
                final_aim = _plan_shot_v8_5(world, planet.id, target.id, need)
                if final_aim is None:
                    continue
                angle, turns, _, _ = final_aim
                score = _target_value(target, turns, "capture", world, modes) / (need + turns + 1.0)
                if target.owner not in (-1, world.player):
                    score *= 1.05
                if best_capture is None or score > best_capture[0]:
                    best_capture = (score, target.id, angle, turns, need)
            if best_capture is not None:
                _, tid, angle, turns, need = best_capture
                actual = append_move(planet.id, angle, need)
                if actual >= 1:
                    planned_commitments[tid].append((turns, world.player, int(actual)))
                continue
            safe_allies = [a for a in world.my_planets if a.id != planet.id and a.id not in world.doomed_candidates]
            if not safe_allies:
                continue
            retreat = min(safe_allies, key=lambda a: (frontier_distance.get(a.id, 1e9), _dist(planet.x, planet.y, a.x, a.y)))
            aim = _plan_shot_v8_5(world, planet.id, retreat.id, available_now)
            if aim is None:
                continue
            append_move(planet.id, aim[0], available_now)

    _emit_transfer_staging(world, ctx, atk_left, append_move)

    # Rear-to-front staging — Fix #4 disables it during 4p early
    rear_staging_active = (
        not throttle
        and not ctx.suppress_rear_staging
        and (world.enemy_planets or world.neutral_planets)
        and len(world.my_planets) > 1
        and not world.is_late
        and not (world.is_four_player and world.step < FOUR_P_REAR_STAGING_MIN_TURN)
    )
    if rear_staging_active:
        frontier_targets = world.enemy_planets or world.static_neutral_planets or world.neutral_planets
        frontier_distance = {p.id: min(_dist(p.x, p.y, t.x, t.y) for t in frontier_targets) for p in world.my_planets}
        safe_fronts = [p for p in world.my_planets if p.id not in world.doomed_candidates]
        if safe_fronts:
            front_anchor = min(safe_fronts, key=lambda p: frontier_distance[p.id])
            send_ratio = REAR_SEND_RATIO_FOUR_PLAYER if world.is_four_player else REAR_SEND_RATIO_TWO_PLAYER
            if modes["is_finishing"]:
                send_ratio = max(send_ratio, REAR_SEND_RATIO_FOUR_PLAYER)
            for rear in sorted(world.my_planets, key=lambda p: -frontier_distance[p.id]):
                if rear.id == front_anchor.id or rear.id in world.doomed_candidates:
                    continue
                if atk_left(rear.id) < REAR_SOURCE_MIN_SHIPS:
                    continue
                if frontier_distance[rear.id] < frontier_distance[front_anchor.id] * REAR_DISTANCE_RATIO:
                    continue
                stage_candidates = [p for p in safe_fronts if p.id != rear.id
                                    and frontier_distance[p.id] < frontier_distance[rear.id] * REAR_STAGE_PROGRESS]
                if stage_candidates:
                    front = min(stage_candidates, key=lambda p: _dist(rear.x, rear.y, p.x, p.y))
                else:
                    obj = min(frontier_targets, key=lambda t: _dist(rear.x, rear.y, t.x, t.y))
                    rem = [p for p in safe_fronts if p.id != rear.id]
                    if not rem:
                        continue
                    front = min(rem, key=lambda p: _dist(p.x, p.y, obj.x, obj.y))
                if front.id == rear.id:
                    continue
                send = int(atk_left(rear.id) * send_ratio)
                if send < REAR_SEND_MIN_SHIPS:
                    continue
                aim = _plan_shot_v8_5(world, rear.id, front.id, send)
                if aim is None:
                    continue
                if aim[1] > REAR_MAX_TRAVEL_TURNS:
                    continue
                append_move(rear.id, aim[0], send)

    # Anti-hoarding (skip when throttled)
    if (not throttle and not ctx.suppress_rear_staging
            and (world.enemy_planets or world.neutral_planets)
            and len(world.my_planets) > 1 and not world.is_late):
        frontier_targets_ah = world.enemy_planets or world.static_neutral_planets or world.neutral_planets
        if frontier_targets_ah:
            front_ah = min(world.my_planets,
                           key=lambda p: min(_dist(p.x, p.y, t.x, t.y) for t in frontier_targets_ah))
            for planet in world.my_planets:
                if planet.id == front_ah.id or planet.id in world.doomed_candidates:
                    continue
                if planet.id in world.threatened_candidates:
                    continue
                hoard_limit = int(planet.production * AOW_HOARD_PROD_RATIO)
                avail = atk_left(planet.id)
                if avail < hoard_limit:
                    continue
                send = int(avail * AOW_HOARD_SEND_RATIO)
                if send < REAR_SEND_MIN_SHIPS:
                    continue
                aim = _plan_shot_v8_5(world, planet.id, front_ah.id, send)
                if aim is None:
                    continue
                if aim[1] > REAR_MAX_TRAVEL_TURNS:
                    continue
                append_move(planet.id, aim[0], send)

    # Fix #2 — garrison ratio cap. Trim moves that would push g_ratio too low.
    if not ctx.suppress_garrison_ratio_cap and moves and world.my_total > 0:
        total_send = sum(m[2] for m in moves)
        on_planets_after = sum(int(p.ships) for p in world.my_planets) - total_send
        ratio_after = on_planets_after / max(1, world.my_total)
        if ratio_after < GARRISON_FLOOR_RATIO:
            target_on_planets = world.my_total * GARRISON_FLOOR_RATIO
            current_on_planets = sum(int(p.ships) for p in world.my_planets)
            max_total_send = max(0, current_on_planets - int(target_on_planets))
            if total_send > max_total_send:
                # Trim least valuable moves (last-emitted = followup/staging) first
                trim = total_send - max_total_send
                trimmed = []
                for sid, ang, ships in reversed(moves):
                    if trim <= 0:
                        trimmed.append([sid, ang, ships])
                        continue
                    if ships <= trim:
                        trim -= ships
                        # drop entirely
                    else:
                        trimmed.append([sid, ang, ships - trim])
                        trim = 0
                trimmed.reverse()
                moves = trimmed

    # Final dedup
    final_moves = []
    used_final: Dict[int, int] = defaultdict(int)
    for sid, angle, ships in moves:
        source = world.planet_by_id[sid]
        max_allowed = int(source.ships) - used_final[sid]
        send = min(int(ships), max_allowed)
        if send >= 1:
            final_moves.append([sid, float(angle), int(send)])
            used_final[sid] += send

    return final_moves, [used_final[sid] for sid in used_final]


# ---------------------------------------------------------------------------
# Candidate plans — produce up to N_PLANS_MAX (some only in 4p)
# ---------------------------------------------------------------------------

def _generate_candidates(world: WorldModel) -> List[Tuple[str, List[List]]]:
    plans: List[Tuple[str, List[List]]] = []
    is_4p = world.is_four_player

    # 0. v7_baseline (V7 grammar + full V8.5 safety fixes)
    plans.append(("v7_baseline", _v8_5_plan(world, PlanContext("v7_baseline"))[0]))
    # 1. expand_focus
    plans.append(("expand_focus",
                  _v8_5_plan(world, PlanContext("expand_focus", expand_boost=1.45,
                                                attack_boost=0.85))[0]))
    # 2. attack_focus
    plans.append(("attack_focus",
                  _v8_5_plan(world, PlanContext("attack_focus", attack_boost=1.55,
                                                expand_boost=0.85))[0]))
    # 3. defense_focus
    plans.append(("defense_focus",
                  _v8_5_plan(world, PlanContext("defense_focus", defense_boost=1.50,
                                                attack_boost=0.85, expand_boost=0.85))[0]))
    # 4. reserve_hold — emit no offense, only reinforcement & evac
    plans.append(("reserve_hold",
                  _v8_5_plan(world, PlanContext("reserve_hold", block_offense=True,
                                                defense_boost=1.30,
                                                suppress_rear_staging=True))[0]))
    # 5. transfer_push — explicitly consolidates rear production into forward launch platforms.
    plans.append(("transfer_push",
                  _v8_5_plan(world, PlanContext("transfer_push",
                                                attack_boost=0.92,
                                                expand_boost=0.95,
                                                transfer_boost=1.30,
                                                force_transfer_staging=True))[0]))
    # 6. probe_focus — small harassment flings on borderline enemy targets.
    plans.append(("probe_focus",
                  _v8_5_plan(world, PlanContext("probe_focus",
                                                attack_boost=0.90,
                                                expand_boost=0.90,
                                                defense_boost=1.00,
                                                probe_boost=1.75))[0]))

    if world.is_opening:
        active_fronts = len(compute_active_fronts(world))
        g_ratio = garrison_ratio_now(world)
        if (world.step < OPENING_FORTIFY_TURN_LIMIT
                and (world.threatened_candidates or world.doomed_candidates
                     or (0 < active_fronts <= OPENING_FORTIFY_MAX_ACTIVE_FRONTS)
                     or g_ratio < OPENING_FORTIFY_MIN_GARRISON_RATIO)):
            plans.append(("opening_fortify",
                          _v8_5_plan(world, PlanContext("opening_fortify",
                                                        block_offense=True,
                                                        defense_boost=1.85,
                                                        suppress_rear_staging=True))[0]))
        if (len(world.my_planets) >= OPENING_RELAY_MIN_PLANETS
                and active_fronts <= OPENING_RELAY_MAX_ACTIVE_FRONTS
                and g_ratio >= OPENING_RELAY_MIN_GARRISON_RATIO):
            plans.append(("opening_relay",
                          _v8_5_plan(world, PlanContext("opening_relay",
                                                        attack_boost=0.90,
                                                        expand_boost=1.05,
                                                        defense_boost=1.10,
                                                        transfer_boost=1.55,
                                                        force_transfer_staging=True))[0]))

    if is_4p:
        # 7. 4p_opportunistic. When no scavenging target exists, keep this as
        # a guarded tempo plan so old rankers do not collapse into passivity.
        if _has_4p_opportunistic_target(world):
            plans.append(("4p_opportunistic",
                          _v8_5_plan(world, PlanContext("4p_opportunistic",
                                                        only_depleted_targets=True,
                                                        attack_boost=1.10,
                                                        expand_boost=1.10))[0]))
        else:
            plans.append(("4p_opportunistic",
                          _v8_5_plan(world, PlanContext("4p_opportunistic",
                                                        attack_boost=0.95,
                                                        expand_boost=1.25,
                                                        defense_boost=1.05))[0]))
        # 8. 4p_eliminate_weakest — only if a player is weak enough to focus down.
        weakest = world.weakest_enemy_id
        weakest_strength = world.owner_strength.get(weakest, 0) if weakest is not None else 0
        if weakest is not None and weakest_strength <= world.my_total * ELIMINATE_WEAKEST_FRACTION:
            plans.append(("4p_eliminate_weakest",
                          _v8_5_plan(world, PlanContext("4p_eliminate_weakest",
                                                         only_targets_owner=weakest,
                                                         attack_boost=1.80,
                                                         expand_boost=0.70))[0]))
        # 9. 4p_conservation — explicit no-offense plan, but still allows rear-to-front redistribution
        conservation_boost = 1.85 if len(compute_active_fronts(world)) < 3 else 2.05
        plans.append(("4p_conservation",
                      _v8_5_plan(world, PlanContext("4p_conservation", block_offense=True,
                                                    defense_boost=conservation_boost,
                                                    suppress_rear_staging=False))[0]))
        # 10. 4p_late_blitz
        if _allow_4p_late_blitz(world, len(compute_active_fronts(world)), garrison_ratio_now(world)):
            plans.append(("4p_late_blitz",
                          _v8_5_plan(world, PlanContext("4p_late_blitz",
                                                        attack_boost=1.80,
                                                        suppress_garrison_floor=True,
                                                        suppress_garrison_ratio_cap=True))[0]))
        # 11. 4p_conversion_push — convert a stable lead before a weak enemy/blitz exists.
        conversion_owner = _conversion_push_owner(world, len(compute_active_fronts(world)), garrison_ratio_now(world))
        if conversion_owner is not None:
            plans.append(("4p_conversion_push",
                          _v8_5_plan(world, PlanContext("4p_conversion_push",
                                                        only_targets_owner=conversion_owner,
                                                        attack_boost=2.05,
                                                        expand_boost=1.05,
                                                        defense_boost=0.95))[0]))

    return plans


# ---------------------------------------------------------------------------
# Candidate features (14-D per plan)
# ---------------------------------------------------------------------------

def build_candidate_features(plan_name: str, moves: List[List], world: WorldModel) -> np.ndarray:
    n_moves = len(moves)
    total_send = sum(m[2] for m in moves) if moves else 0
    my_total = max(1, world.my_total)
    ship_frac = total_send / my_total

    # Targets summary — match each move to the closest planet along its launch
    # angle. The previous tolerance of 0.25 rad was too tight: orbit-prediction
    # angles often deviate by 0.4-0.6 rad from a straight-line shot, so most
    # moves were unmatched and the categorical features collapsed to zero.
    targets_attack = 0
    targets_expand = 0
    targets_defense = 0
    transfer_ships = 0
    avg_eta = 0.0
    n_eta = 0
    distinct_targets = set()
    for sid, angle, ships in moves:
        src = world.planet_by_id.get(sid)
        if src is None:
            continue
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        best = None
        best_score = 1e18
        for p in world.planets:
            if p.id == sid:
                continue
            dx = p.x - src.x
            dy = p.y - src.y
            d = math.hypot(dx, dy)
            if d <= 1e-6:
                continue
            # Project onto launch direction; require positive forward distance
            proj = dx * cos_a + dy * sin_a
            if proj <= 0:
                continue
            # Perpendicular gap shrinks as planets line up with the angle
            perp = abs(dx * sin_a - dy * cos_a)
            if perp > p.radius + 6.0:
                continue
            score = perp - 0.05 * proj  # prefer closer alignment, ties broken toward nearer
            if score < best_score:
                best = p
                best_score = score
        if best is None:
            continue
        distinct_targets.add(best.id)
        if best.owner == world.player:
            targets_defense += 1
            transfer_ships += int(ships)
        elif best.owner == -1:
            targets_expand += 1
        else:
            targets_attack += 1
        avg_eta += math.hypot(best.x - src.x, best.y - src.y)
        n_eta += 1
    avg_eta = (avg_eta / max(1, n_eta)) / 100.0

    transfer_frac = transfer_ships / my_total

    # Plan-id one-hot collapsed: keep plan_idx normalized + dedicated flags
    plan_idx = PLAN_NAMES.index(plan_name) if plan_name in PLAN_NAMES else 0

    feat = np.array([
        min(n_moves / 20.0, 2.0),                                   # CAND_N_MOVES
        min(ship_frac, 2.0),                                        # CAND_SHIP_FRAC
        targets_attack / max(1, n_moves),                           # CAND_ATTACK_FRAC
        targets_expand / max(1, n_moves),                           # CAND_EXPAND_FRAC
        targets_defense / max(1, n_moves),                          # CAND_DEFENSE_FRAC
        avg_eta,                                                    # CAND_AVG_ETA
        plan_idx / float(N_PLANS_MAX),                              # CAND_PLAN_IDX
        1.0 if plan_name == "v7_baseline" else 0.0,                 # CAND_IS_BASELINE
        1.0 if plan_name.startswith("4p_") else 0.0,                # CAND_IS_4P
        1.0 if plan_name in ("defense_focus", "reserve_hold", "4p_conservation", "opening_fortify") else 0.0,  # CAND_IS_DEFENSE
        1.0 if plan_name in ("attack_focus", "4p_eliminate_weakest", "4p_late_blitz", "4p_conversion_push") else 0.0,  # CAND_IS_ATTACK
        min(transfer_frac, 2.0),                                     # CAND_TRANSFER_FRAC
        1.0 if plan_name in ("transfer_push", "opening_relay") else 0.0,  # CAND_IS_TRANSFER
        1.0,                                                        # CAND_BIAS
    ], dtype=np.float32)
    return feat


def _policy_bonus(plan_name: str, cand_feat: np.ndarray, *, is_4p: bool, is_opening: bool,
                  is_late: bool, active_fronts: int, g_ratio: float, ship_lead: float,
                  prod_lead: float, has_conversion: bool, has_opportunity: bool) -> float:
    """Hand-authored macro prior on top of the linear ranker.

    The linear head still decides between nearby candidates, but this nudges the
    policy toward the macro family that matches the current game phase.
    """
    attack_frac = float(cand_feat[CAND_ATTACK_FRAC])
    transfer_frac = float(cand_feat[CAND_TRANSFER_FRAC])
    is_transfer = bool(cand_feat[CAND_IS_TRANSFER] > 0.5)
    is_attack = bool(cand_feat[CAND_IS_ATTACK] > 0.5)
    is_defense = bool(cand_feat[CAND_IS_DEFENSE] > 0.5)

    bonus = 0.0
    if plan_name == "v7_baseline":
        if is_4p and (active_fronts > 0 or g_ratio < 0.34):
            bonus -= 0.10
    elif plan_name == "expand_focus":
        if (not is_4p and active_fronts == 0 and g_ratio >= 0.28) or (
                is_4p and is_opening and active_fronts == 0 and g_ratio >= 0.30):
            bonus += 0.20
    elif plan_name == "attack_focus":
        if is_attack and attack_frac >= 0.18 and active_fronts <= 2:
            bonus += 0.10
    elif plan_name == "defense_focus":
        if is_defense and (active_fronts > 0 or g_ratio < 0.34):
            bonus += 0.16
    elif plan_name == "reserve_hold":
        if active_fronts >= 2 or g_ratio < 0.30:
            bonus += 0.18
        else:
            bonus -= 0.08
    elif plan_name == "transfer_push":
        if is_transfer and transfer_frac >= 0.06 and g_ratio >= 0.38 and active_fronts <= 1:
            bonus += 0.24
        else:
            bonus -= 0.04
    elif plan_name == "probe_focus":
        if is_4p and (active_fronts > 0 or has_opportunity or g_ratio >= 0.26):
            bonus += 0.18
        else:
            bonus -= 0.04
    elif plan_name == "4p_opportunistic":
        if has_opportunity:
            bonus += 0.24
        else:
            bonus -= 0.10
    elif plan_name == "4p_eliminate_weakest":
        if has_conversion:
            bonus += 0.26
        else:
            bonus -= 0.12
    elif plan_name == "4p_conservation":
        if is_4p and (active_fronts >= 2 or g_ratio < 0.30):
            bonus += 0.34
        else:
            bonus -= 0.14
    elif plan_name == "4p_late_blitz":
        if is_4p and is_late and (ship_lead >= 1.05 or prod_lead >= 1.05):
            bonus += 0.28
        else:
            bonus -= 0.10
    elif plan_name == "4p_conversion_push":
        if has_conversion and active_fronts <= 2 and g_ratio >= 0.32:
            bonus += 0.30
        else:
            bonus -= 0.12
    elif plan_name == "opening_fortify":
        if is_opening and (g_ratio < OPENING_FORTIFY_MIN_GARRISON_RATIO or active_fronts > 0):
            bonus += 0.38
        else:
            bonus -= 0.10
    elif plan_name == "opening_relay":
        if is_opening and g_ratio >= OPENING_RELAY_MIN_GARRISON_RATIO and active_fronts <= 1 and is_transfer:
            bonus += 0.30
        else:
            bonus -= 0.06
    return bonus


# ---------------------------------------------------------------------------
# Ranker scoring
# ---------------------------------------------------------------------------

def score_candidates(state_feat: np.ndarray, candidates: List[Tuple[str, List[List]]],
                     world: WorldModel) -> Tuple[List[float], List[np.ndarray]]:
    """Return (scores per candidate, candidate features per candidate)."""
    cand_feats: List[np.ndarray] = []
    scores: List[float] = []
    active_fronts = len(compute_active_fronts(world))
    g_ratio = garrison_ratio_now(world)
    strongest_enemy = max(
        (s for owner, s in world.owner_strength.items() if owner not in (-1, world.player)),
        default=0,
    )
    strongest_enemy_prod = max(
        (p for owner, p in world.owner_production.items() if owner not in (-1, world.player)),
        default=0,
    )
    ship_lead = world.my_total / max(1, strongest_enemy)
    prod_lead = world.my_prod / max(1, strongest_enemy_prod)
    has_conversion = _conversion_push_owner(world, active_fronts, g_ratio) is not None
    has_opportunity = _has_4p_opportunistic_target(world)
    for plan_name, moves in candidates:
        cf = build_candidate_features(plan_name, moves, world)
        plan_idx = PLAN_NAMES.index(plan_name)
        # Score = state_w[plan_idx] · state_feat + cand_w · cand_feat
        s = float(_STATE_W[plan_idx] @ state_feat) + float(_CAND_W @ cf)
        s += _policy_bonus(
            plan_name,
            cf,
            is_4p=bool(state_feat[STATE_IS_4P] > 0.5),
            is_opening=bool(state_feat[STATE_IS_OPENING] > 0.5),
            is_late=bool(state_feat[STATE_IS_LATE] > 0.5),
            active_fronts=active_fronts,
            g_ratio=g_ratio,
            ship_lead=ship_lead,
            prod_lead=prod_lead,
            has_conversion=has_conversion,
            has_opportunity=has_opportunity,
        )
        scores.append(s)
        cand_feats.append(cf)
    return scores, cand_feats


def value_estimate(state_feat: np.ndarray) -> float:
    return float(_VALUE_W @ state_feat) + float(_VALUE_B)


def _select_candidate_index(candidates: List[Tuple[str, List[List]]], scores: List[float],
                            cand_feats: List[np.ndarray], world: WorldModel) -> int:
    active_fronts = len(compute_active_fronts(world))
    g_ratio = garrison_ratio_now(world)
    if (world.is_four_player
            and world.step < EARLY_4P_EXPAND_OVERRIDE_TURN
            and len(world.my_planets) < EARLY_4P_EXPAND_PLANET_TARGET
            and not active_fronts
            and g_ratio >= EARLY_4P_EXPAND_MIN_GARRISON_RATIO):
        for i, (plan_name, moves) in enumerate(candidates):
            if plan_name == "expand_focus" and moves:
                return i

    if (world.is_opening and world.step < OPENING_FORTIFY_TURN_LIMIT
            and (world.threatened_candidates or world.doomed_candidates
                 or (0 < active_fronts <= OPENING_FORTIFY_MAX_ACTIVE_FRONTS)
                 or g_ratio < OPENING_FORTIFY_MIN_GARRISON_RATIO)):
        for i, (plan_name, moves) in enumerate(candidates):
            if plan_name == "opening_fortify" and moves:
                return i

    best_idx = 0
    best_score = scores[0]
    for i in range(1, len(scores)):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
    if not candidates[best_idx][1]:
        best_nonempty = None
        for i, (_, moves) in enumerate(candidates):
            if not moves:
                continue
            if best_nonempty is None or scores[i] > scores[best_nonempty]:
                best_nonempty = i
        if best_nonempty is not None:
            best_idx = best_nonempty
    if ((world.step >= TRANSFER_OVERRIDE_MIN_STEP or world.step < OPENING_RELAY_TURN_LIMIT)
            and not world.is_late
            and active_fronts <= max(OPENING_RELAY_MAX_ACTIVE_FRONTS, TRANSFER_OVERRIDE_MAX_ACTIVE_FRONTS)
            and g_ratio >= OPENING_RELAY_MIN_GARRISON_RATIO):
        chosen_transfer_frac = float(cand_feats[best_idx][CAND_TRANSFER_FRAC])
        for i, (plan_name, moves) in enumerate(candidates):
            if plan_name not in ("transfer_push", "opening_relay") or not moves:
                continue
            transfer_frac = float(cand_feats[i][CAND_TRANSFER_FRAC])
            if transfer_frac >= max(OPENING_RELAY_MIN_TRANSFER_FRAC, TRANSFER_OVERRIDE_MIN_FRAC) and transfer_frac > chosen_transfer_frac + 0.02:
                return i
    if _conversion_push_owner(world) is not None:
        chosen_name, chosen_moves = candidates[best_idx]
        chosen_attack_ratio = float(cand_feats[best_idx][CAND_ATTACK_FRAC])
        chosen_is_generic = chosen_name in ("v7_baseline", "attack_focus", "4p_opportunistic")
        chosen_is_passive = chosen_name in ("defense_focus", "reserve_hold", "4p_conservation")
        chosen_lacks_attack = bool(chosen_moves) and chosen_attack_ratio <= 0.05
        for i, (plan_name, moves) in enumerate(candidates):
            if plan_name == "4p_conversion_push" and moves and float(cand_feats[i][CAND_ATTACK_FRAC]) > 0.05:
                if chosen_is_generic or chosen_is_passive or chosen_lacks_attack or not chosen_moves:
                    return i
    return best_idx


# Public deterministic planner useful for tests and ablations. It returns the
# current ranker-selected V8.5 candidate plan for an already-built WorldModel.
def plan_moves_v8_5(world: WorldModel) -> List[List]:
    candidates = _generate_candidates(world)
    if not candidates:
        return []
    state_feat = build_state_features(world)
    scores, cand_feats = score_candidates(state_feat, candidates, world)
    best_idx = _select_candidate_index(candidates, scores, cand_feats, world)
    return candidates[best_idx][1]


# ---------------------------------------------------------------------------
# Agent entry
# ---------------------------------------------------------------------------

def agent(obs, config=None):
    try:
        world = _build_world(obs)
        if not world.my_planets:
            return []
        candidates = _generate_candidates(world)
        if not candidates:
            return []
        # Single candidate? Return immediately.
        if len(candidates) == 1:
            return candidates[0][1]
        state_feat = build_state_features(world)
        scores, cand_feats = score_candidates(state_feat, candidates, world)
        best_idx = _select_candidate_index(candidates, scores, cand_feats, world)
        if _CANDIDATE_LOG_CB is not None:
            try:
                _CANDIDATE_LOG_CB({
                    "step": world.step,
                    "player": world.player,
                    "state_feat": state_feat,
                    "candidate_names": [c[0] for c in candidates],
                    "candidate_feats": cand_feats,
                    "candidate_moves": [c[1] for c in candidates],
                    "scores": scores,
                    "chosen": best_idx,
                    "value": value_estimate(state_feat),
                })
            except Exception:
                pass
        return candidates[best_idx][1]
    except Exception:
        # Hard fallback to V7 grammar — never forfeit a turn silently.
        try:
            return v7.agent(obs, config)
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "evaluations" / "v8_5_ranker.npz"


def save_checkpoint(path: Optional[str] = None, meta: Optional[dict] = None) -> str:
    p = Path(path) if path else DEFAULT_CHECKPOINT
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_w": _STATE_W,
        "candidate_w": _CAND_W,
        "value_w": _VALUE_W,
        "value_b": np.array([_VALUE_B], dtype=np.float32),
    }
    if meta:
        for k, v in meta.items():
            payload[f"meta_{k}"] = np.asarray(v)
    np.savez_compressed(str(p), **payload)
    return str(p)


def load_checkpoint(path: Optional[str] = None) -> bool:
    p = Path(path) if path else DEFAULT_CHECKPOINT
    if not p.exists():
        return False
    try:
        data = np.load(str(p))
        state_w = np.asarray(data["state_w"], dtype=np.float32)
        if state_w.shape != (N_PLANS_MAX, N_STATE_FEATURES):
            migrated = np.zeros((N_PLANS_MAX, N_STATE_FEATURES), dtype=np.float32)
            rows = min(state_w.shape[0], N_PLANS_MAX)
            cols = min(state_w.shape[1], N_STATE_FEATURES)
            migrated[:rows, :cols] = state_w[:rows, :cols]
            state_w = migrated
        candidate_w = np.asarray(data["candidate_w"], dtype=np.float32).ravel()
        if candidate_w.size != N_CANDIDATE_FEATURES:
            migrated_candidate = np.zeros(N_CANDIDATE_FEATURES, dtype=np.float32)
            cols = min(candidate_w.size, N_CANDIDATE_FEATURES)
            migrated_candidate[:cols] = candidate_w[:cols]
            candidate_w = migrated_candidate
        set_ranker_weights(
            state_w=state_w,
            candidate_w=candidate_w,
            value_w=data["value_w"],
            value_b=float(data["value_b"][0]),
        )
        return True
    except Exception:
        return False


# Auto-load on import if checkpoint exists
if os.environ.get("BOT_V8_5_NO_AUTOLOAD") != "1":
    load_checkpoint()


__all__ = [
    "agent",
    "set_ranker_weights",
    "get_ranker_weights",
    "set_candidate_log_callback",
    "save_checkpoint",
    "load_checkpoint",
    "build_state_features",
    "build_candidate_features",
    "score_candidates",
    "value_estimate",
    "plan_moves_v8_5",
    "PLAN_NAMES",
    "N_STATE_FEATURES",
    "N_CANDIDATE_FEATURES",
    "N_PLANS_MAX",
    "DEFAULT_CHECKPOINT",
]
