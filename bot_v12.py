"""
bot_v12.py — Simulation-based deterministic bot (Path C)

Strategy:
  1. Simulate exact future state of every planet (production + in-transit fleets)
  2. Find windows where enemy/neutral planets are minimally defended
  3. Launch coordinated swarms: all sources aimed at target's future position,
     timed to arrive at the same turn
  4. Full-send when simulation confirms capture
  5. Garrison computed from incoming threats, not fixed ratio
"""

from __future__ import annotations

import math
from typing import Any

# ── Game constants (from orbit_wars.py) ───────────────────────────────────────
_MAX_SPEED = 6.0
_SUN_X = _SUN_Y = 50.0
_SUN_R = 10.0
_BOARD = 100.0
_CENTER = 50.0
_ROT_LIMIT = 50.0          # orbital_radius + planet_radius < this → orbits

# ── Tuning ────────────────────────────────────────────────────────────────────
HORIZON = 55               # turns to look ahead
MIN_GARRISON = 6           # absolute minimum ships to keep
GARRISON_PROD_MULT = 2.0   # keep prod * mult extra ships on front planets
ATTACK_MARGIN = 1.08       # swarm must be 8% more than defenders
MIN_SEND = 5               # don't bother with tiny fleets
MAX_ACTIONS = 8            # max orders per turn


# ═════════════════════════════ PHYSICS ════════════════════════════════════════

def _fleet_speed(ships: int) -> float:
    s = max(1, int(ships))
    return min(_MAX_SPEED, 1.0 + (_MAX_SPEED - 1.0) * (math.log(s) / math.log(1000)) ** 1.5)


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _seg_dist_to_point(px: float, py: float,
                       ax: float, ay: float,
                       bx: float, by: float) -> float:
    """Min distance from point (px,py) to segment (ax,ay)→(bx,by)."""
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return _dist(px, py, ax, ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return _dist(px, py, ax + t * dx, ay + t * dy)


def _planet_pos(init_x: float, init_y: float, radius: float,
                av: float, step: int) -> tuple[float, float]:
    """Return planet position at absolute game step `step`."""
    dx = init_x - _CENTER
    dy = init_y - _CENTER
    r = math.sqrt(dx * dx + dy * dy)
    if r + radius >= _ROT_LIMIT:           # static planet
        return init_x, init_y
    init_angle = math.atan2(dy, dx)
    angle = init_angle + av * step
    return _CENTER + r * math.cos(angle), _CENTER + r * math.sin(angle)


def _path_hits_sun(fx: float, fy: float, angle: float, total_dist: float) -> bool:
    """Does the straight-line path from (fx,fy) for total_dist in direction `angle` hit the sun?"""
    ex = fx + math.cos(angle) * total_dist
    ey = fy + math.sin(angle) * total_dist
    return _seg_dist_to_point(_SUN_X, _SUN_Y, fx, fy, ex, ey) < _SUN_R


def _compute_eta_and_angle(
    src_x: float, src_y: float, src_radius: float,
    tgt_init_x: float, tgt_init_y: float, tgt_radius: float, tgt_planet_radius_static: float,
    ships: int,
    av: float,
    current_step: int,
) -> tuple[int, float] | None:
    """
    Find the earliest ETA and launch angle to send `ships` from src to tgt.
    Returns (eta_turns, angle) or None if unreachable within HORIZON.

    For orbiting targets: aim at the planet's future position at ETA
    so the fleet arrives exactly when the planet is at that position.
    """
    speed = _fleet_speed(ships)

    for eta in range(1, HORIZON + 1):
        tx, ty = _planet_pos(tgt_init_x, tgt_init_y, tgt_planet_radius_static,
                             av, current_step + eta)
        # Fleet starts just outside source planet radius
        angle = math.atan2(ty - src_y, tx - src_x)
        fleet_sx = src_x + math.cos(angle) * (src_radius + 0.1)
        fleet_sy = src_y + math.sin(angle) * (src_radius + 0.1)

        d = _dist(fleet_sx, fleet_sy, tx, ty) - tgt_radius
        if d <= 0.0:
            return eta, angle  # adjacent planets

        # Fleet covers speed*eta distance — does it reach target?
        if speed * eta >= d:
            # Make sure it doesn't arrive *before* eta (would fight alone early)
            if eta == 1 or speed * (eta - 1) < d:
                # Sun check: does path from fleet_start toward tgt intersect sun?
                if not _path_hits_sun(fleet_sx, fleet_sy, angle, speed * eta):
                    return eta, angle
                else:
                    return None   # this src→tgt combo blocked by sun for all ETAs
    return None


# ═════════════════════════════ SIMULATION ═════════════════════════════════════

def _build_initial_map(obs: Any) -> dict[int, tuple[float, float, float]]:
    """Map planet_id → (init_x, init_y, radius)."""
    ip_list = _get(obs, 'initial_planets', []) or []
    return {int(p[0]): (float(p[2]), float(p[3]), float(p[4])) for p in ip_list}


def _fleet_arrives_at(fleet: Any, planet: Any,
                      ip: dict[int, tuple[float, float, float]],
                      av: float, current_step: int) -> int | None:
    """
    Simulate an in-transit fleet to find which planet it hits and when.
    Returns turns-until-arrival or None if it won't hit this planet within HORIZON.
    """
    fx, fy = float(fleet[2]), float(fleet[3])
    angle = float(fleet[4])
    ships = int(fleet[6])
    speed = _fleet_speed(ships)
    pid = int(planet[0])
    init_x, init_y, pradius = ip.get(pid, (float(planet[2]), float(planet[3]), float(planet[4])))

    for t in range(1, HORIZON + 1):
        ofx, ofy = fx, fy
        fx += math.cos(angle) * speed
        fy += math.sin(angle) * speed

        # Sun destroys fleet
        if _seg_dist_to_point(_SUN_X, _SUN_Y, ofx, ofy, fx, fy) < _SUN_R:
            return None

        # Out of bounds
        if not (0.0 <= fx <= _BOARD and 0.0 <= fy <= _BOARD):
            return None

        # Check this planet (continuous: old→new segment vs planet circle)
        px, py = _planet_pos(init_x, init_y, pradius, av, current_step + t)
        if _seg_dist_to_point(px, py, ofx, ofy, fx, fy) < pradius:
            return t

    return None


def _build_fleet_arrival_table(
    obs: Any,
    ip: dict[int, tuple[float, float, float]],
    av: float,
    current_step: int,
) -> dict[int, list[tuple[int, int, int]]]:
    """
    For each planet_id: list of (eta_turns, owner, ships) for fleets heading there.
    Only includes fleets that WILL arrive (not sun-blocked, not OOB).
    """
    obs_planets = list(_get(obs, 'planets', []) or [])
    obs_fleets = list(_get(obs, 'fleets', []) or [])
    table: dict[int, list[tuple[int, int, int]]] = {int(p[0]): [] for p in obs_planets}
    planets = obs_planets

    for fleet in obs_fleets:
        fx, fy = float(fleet[2]), float(fleet[3])
        angle = float(fleet[4])
        ships = int(fleet[6])
        fowner = int(fleet[1])
        speed = _fleet_speed(ships)
        cx, cy = fx, fy

        for t in range(1, HORIZON + 1):
            ocx, ocy = cx, cy
            cx += math.cos(angle) * speed
            cy += math.sin(angle) * speed

            if _seg_dist_to_point(_SUN_X, _SUN_Y, ocx, ocy, cx, cy) < _SUN_R:
                break
            if not (0.0 <= cx <= _BOARD and 0.0 <= cy <= _BOARD):
                break

            hit = False
            for planet in planets:
                pid = int(planet[0])
                init_x, init_y, pradius = ip.get(pid, (float(planet[2]), float(planet[3]), float(planet[4])))
                px, py = _planet_pos(init_x, init_y, pradius, av, current_step + t)
                if _seg_dist_to_point(px, py, ocx, ocy, cx, cy) < pradius:
                    table[pid].append((t, fowner, ships))
                    hit = True
                    break
            if hit:
                break

    return table


def _planet_defenders_at_eta(
    planet: Any,
    eta: int,
    my_id: int,
    arrival_table: dict[int, list[tuple[int, int, int]]],
) -> float:
    """
    Estimate how many ships will defend `planet` (not owned by my_id) at turn +eta.
    Conservative estimate (favors caution): assumes planet keeps owner.
    """
    pid = int(planet[0])
    owner = int(planet[1])
    ships = float(planet[5])
    prod = float(planet[6])

    # Production each turn until eta
    if owner != -1:
        ships += prod * eta

    # Add reinforcements from fleets arriving before eta
    for (t, fowner, fships) in arrival_table.get(pid, []):
        if t <= eta and fowner == owner:
            ships += fships
        # Enemy fleet arriving could reduce defenders (simplification: ignore for now)

    return max(0.0, ships)


def _my_planet_garrison_needed(
    planet: Any,
    my_id: int,
    arrival_table: dict[int, list[tuple[int, int, int]]],
) -> int:
    """
    Minimum ships to keep on this planet considering incoming enemy threats.
    """
    pid = int(planet[0])
    prod = float(planet[6])
    base = MIN_GARRISON

    # Find largest incoming enemy fleet
    max_threat = 0
    earliest_threat = HORIZON + 1
    for (t, fowner, fships) in arrival_table.get(pid, []):
        if fowner != my_id:
            if fships > max_threat:
                max_threat = fships
                earliest_threat = t

    if max_threat > 0:
        # Need enough to survive: max_threat + production grown by enemy
        # + buffer for uncertainty
        needed = int(max_threat * 1.15 - prod * earliest_threat)
        base = max(base, needed)

    return base


# ═════════════════════════════ DECISION ENGINE ════════════════════════════════

def _score_target(planet: Any, eta: int, defenders: float, swarm_cost: float) -> float:
    """
    Score an attack opportunity. Higher = better.
    Favors: high production, low defenders, low cost, short ETA.
    """
    prod = float(planet[6])
    efficiency = prod / max(1.0, swarm_cost)   # production gained per ship spent
    urgency = 1.0 / max(1, eta)                # prefer sooner
    # Penalize expensive captures
    cost_penalty = 1.0 / max(1.0, swarm_cost / max(1.0, prod * 5))
    return efficiency * urgency + prod * 0.1


def _find_best_attack(
    planets: list,
    my_id: int,
    ip: dict[int, tuple[float, float, float]],
    av: float,
    current_step: int,
    arrival_table: dict[int, list[tuple[int, int, int]]],
    committed: dict[int, int],   # planet_id → ships already committed
) -> list[tuple[int, float, int]] | None:
    """
    Find the best attack opportunity and return list of (from_planet_id, angle, ships).
    Returns None if no good opportunity found.
    """
    my_planets = [p for p in planets if int(p[1]) == my_id]
    target_planets = [p for p in planets if int(p[1]) != my_id]

    if not my_planets or not target_planets:
        return None

    best_score = -1.0
    best_moves: list[tuple[int, float, int]] | None = None

    for tgt in target_planets:
        tid = int(tgt[0])
        tgt_init_x, tgt_init_y, tgt_radius = ip.get(
            tid, (float(tgt[2]), float(tgt[3]), float(tgt[4])))

        # Find which source planets can reach this target and at what ETA
        src_options: list[tuple[int, int, float, int]] = []  # (src_id, eta, angle, avail_ships)
        for src in my_planets:
            sid = int(src[0])
            sx, sy = float(src[2]), float(src[3])
            sr = float(src[4])
            avail = int(src[5]) - committed.get(sid, 0) - _my_planet_garrison_needed(src, my_id, arrival_table)
            if avail < MIN_SEND:
                continue

            result = _compute_eta_and_angle(
                sx, sy, sr,
                tgt_init_x, tgt_init_y, tgt_radius, tgt_radius,
                avail, av, current_step,
            )
            if result is None:
                continue
            eta, angle = result
            src_options.append((sid, eta, angle, avail))

        if not src_options:
            continue

        # Group by ETA: for each possible ETA, find all sources that arrive at exactly that ETA
        # Then check if combined swarm beats defenders
        etas_seen = sorted(set(eta for (_, eta, _, _) in src_options))

        for target_eta in etas_seen:
            # Sources arriving at this exact ETA
            exact_sources = [(sid, angle, avail) for (sid, eta, angle, avail) in src_options
                             if eta == target_eta]

            defenders = _planet_defenders_at_eta(tgt, target_eta, my_id, arrival_table)
            total_swarm = sum(avail for (_, _, avail) in exact_sources)

            if total_swarm < defenders * ATTACK_MARGIN:
                # Try adding sources with earlier ETA (they arrive first, weaken defenders)
                # This handles sequential waves: not perfectly coordinated but still effective
                earlier = [(sid, angle, avail) for (sid, eta, angle, avail) in src_options
                           if eta < target_eta]
                total_with_earlier = total_swarm + sum(avail for (_, _, avail) in earlier)
                if total_with_earlier < defenders * ATTACK_MARGIN:
                    continue
                # Use all sources (earlier + exact)
                combined = [(sid, angle, avail) for (sid, angle, avail) in exact_sources]
                combined += [(sid, angle, avail) for (sid, angle, avail) in earlier]
            else:
                combined = [(sid, angle, avail) for (sid, angle, avail) in exact_sources]

            total_swarm = sum(avail for (_, _, avail) in combined)
            score = _score_target(tgt, target_eta, defenders, total_swarm)

            if score > best_score:
                best_score = score
                # Build move list: send all available ships from each source
                moves = []
                for (sid, angle, avail) in combined:
                    moves.append((sid, angle, avail))
                best_moves = moves

    return best_moves


def _find_expand_target(
    planets: list,
    my_id: int,
    ip: dict[int, tuple[float, float, float]],
    av: float,
    current_step: int,
    arrival_table: dict[int, list[tuple[int, int, int]]],
    committed: dict[int, int],
) -> list[tuple[int, float, int]] | None:
    """
    Expand toward best neutral planet (highest prod/dist ratio).
    """
    my_planets = [p for p in planets if int(p[1]) == my_id]
    neutrals = [p for p in planets if int(p[1]) == -1]

    if not my_planets or not neutrals:
        return None

    best_score = -1.0
    best_move: tuple[int, float, int] | None = None

    for tgt in neutrals:
        tid = int(tgt[0])
        tgt_init_x, tgt_init_y, tgt_radius = ip.get(
            tid, (float(tgt[2]), float(tgt[3]), float(tgt[4])))

        # Find closest source with enough ships
        for src in my_planets:
            sid = int(src[0])
            sx, sy = float(src[2]), float(src[3])
            sr = float(src[4])
            avail = int(src[5]) - committed.get(sid, 0) - _my_planet_garrison_needed(src, my_id, arrival_table)
            if avail < MIN_SEND:
                continue

            result = _compute_eta_and_angle(
                sx, sy, sr,
                tgt_init_x, tgt_init_y, tgt_radius, tgt_radius,
                avail, av, current_step,
            )
            if result is None:
                continue
            eta, angle = result

            defenders = _planet_defenders_at_eta(tgt, eta, my_id, arrival_table)
            if avail <= defenders:
                continue

            prod = float(tgt[6])
            dist = _dist(sx, sy, tgt_init_x, tgt_init_y)
            score = prod / max(1.0, dist) / max(1.0, defenders + 1)

            if score > best_score:
                best_score = score
                # Send just enough + small buffer
                send = min(avail, int(defenders * 1.15) + int(prod * eta) + MIN_SEND)
                send = max(send, MIN_SEND)
                best_move = (sid, angle, send)

    return [best_move] if best_move else None


def _rear_staging(
    planets: list,
    my_id: int,
    ip: dict[int, tuple[float, float, float]],
    av: float,
    current_step: int,
    arrival_table: dict[int, list[tuple[int, int, int]]],
    committed: dict[int, int],
    front_ids: set[int],
) -> list[tuple[int, float, int]]:
    """
    Move excess ships from safe rear planets toward front-line planets.
    """
    moves = []
    my_planets = [p for p in planets if int(p[1]) == my_id]
    rear_planets = [p for p in my_planets if int(p[0]) not in front_ids]
    front_planets = [p for p in my_planets if int(p[0]) in front_ids]

    if not rear_planets or not front_planets:
        return moves

    for rear in rear_planets:
        rid = int(rear[0])
        garrison = _my_planet_garrison_needed(rear, my_id, arrival_table)
        excess = int(rear[5]) - committed.get(rid, 0) - garrison - int(rear[6]) * 4
        if excess < MIN_SEND * 2:
            continue

        # Find closest front planet
        best_front = min(front_planets, key=lambda fp: _dist(
            float(rear[2]), float(rear[3]), float(fp[2]), float(fp[3])))
        fid = int(best_front[0])
        fx_init, fy_init, fr = ip.get(fid, (float(best_front[2]), float(best_front[3]), float(best_front[4])))

        result = _compute_eta_and_angle(
            float(rear[2]), float(rear[3]), float(rear[4]),
            fx_init, fy_init, fr, fr,
            excess, av, current_step,
        )
        if result is None:
            continue
        _, angle = result
        moves.append((rid, angle, excess))

    return moves


# ═════════════════════════════ MAIN AGENT ═════════════════════════════════════

def agent(obs: Any, config: Any = None) -> list[list]:
    try:
        return _agent_inner(obs, config)
    except Exception:
        return []


def _get(obs: Any, key: str, default: Any = None) -> Any:
    """Access obs whether it's a dict or an attribute-based Struct."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _agent_inner(obs: Any, config: Any) -> list[list]:
    my_id = int(_get(obs, 'player', 0))
    current_step = int(_get(obs, 'step', 0))
    av = float(_get(obs, 'angular_velocity', 0.03))

    planets = list(_get(obs, 'planets', []) or [])
    fleets = list(_get(obs, 'fleets', []) or [])

    if not planets:
        return []

    # ── Build lookup structures ────────────────────────────────────────────────
    ip = _build_initial_map(obs)

    # Update current positions (planets have moved since initial_planets)
    # obs.planets already has current positions — use those for distance calcs
    # but ip stores initial positions for orbital calculations

    arrival_table = _build_fleet_arrival_table(obs, ip, av, current_step)

    committed: dict[int, int] = {}   # planet_id → ships reserved

    my_planets = [p for p in planets if int(p[1]) == my_id]
    enemy_planets = [p for p in planets if int(p[1]) not in (-1, my_id)]

    if not my_planets:
        return []

    actions: list[list] = []

    # ── Identify front-line planets (near enemies) ─────────────────────────────
    front_ids: set[int] = set()
    if enemy_planets:
        for mp in my_planets:
            min_enemy_dist = min(
                _dist(float(mp[2]), float(mp[3]), float(ep[2]), float(ep[3]))
                for ep in enemy_planets
            )
            if min_enemy_dist < 40.0:
                front_ids.add(int(mp[0]))
    if not front_ids:
        front_ids = {int(mp[0]) for mp in my_planets}

    # ── Emergency defense ──────────────────────────────────────────────────────
    for mp in my_planets:
        mid = int(mp[0])
        immediate_threats = [
            (t, fships) for (t, fowner, fships) in arrival_table.get(mid, [])
            if fowner != my_id and t <= 6
        ]
        if not immediate_threats:
            continue

        max_threat = sum(fships for (_, fships) in immediate_threats)
        current_defenders = int(mp[5])

        if current_defenders >= max_threat * 1.1:
            continue  # already safe

        # Find nearest friendly planet to reinforce
        neighbors = [p for p in my_planets if int(p[0]) != mid]
        if not neighbors:
            continue

        for neighbor in sorted(neighbors,
                               key=lambda p: _dist(float(p[2]), float(p[3]),
                                                   float(mp[2]), float(mp[3]))):
            nid = int(neighbor[0])
            ngarrison = _my_planet_garrison_needed(neighbor, my_id, arrival_table)
            navail = int(neighbor[5]) - committed.get(nid, 0) - ngarrison
            if navail < MIN_SEND:
                continue

            init_x, init_y, pradius = ip.get(mid, (float(mp[2]), float(mp[3]), float(mp[4])))
            result = _compute_eta_and_angle(
                float(neighbor[2]), float(neighbor[3]), float(neighbor[4]),
                init_x, init_y, pradius, pradius,
                navail, av, current_step,
            )
            if result is None:
                continue
            _, angle = result
            send = min(navail, max_threat + MIN_GARRISON * 2)
            actions.append([nid, angle, send])
            committed[nid] = committed.get(nid, 0) + send
            if len(actions) >= MAX_ACTIONS:
                return actions
            break

    # ── Main attack ────────────────────────────────────────────────────────────
    best_attack = _find_best_attack(
        planets, my_id, ip, av, current_step, arrival_table, committed)

    if best_attack:
        for (sid, angle, ships) in best_attack:
            if len(actions) >= MAX_ACTIONS:
                break
            send = max(MIN_SEND, ships)
            actual_avail = int(next(p[5] for p in my_planets if int(p[0]) == sid))
            send = min(send, actual_avail - committed.get(sid, 0) - MIN_GARRISON)
            if send < MIN_SEND:
                continue
            actions.append([sid, angle, send])
            committed[sid] = committed.get(sid, 0) + send

    # ── Expand neutral if no attack ────────────────────────────────────────────
    if not best_attack or len(actions) < 2:
        expand = _find_expand_target(
            planets, my_id, ip, av, current_step, arrival_table, committed)
        if expand:
            for (sid, angle, ships) in expand:
                if len(actions) >= MAX_ACTIONS:
                    break
                actual_avail = int(next(p[5] for p in my_planets if int(p[0]) == sid))
                send = min(ships, actual_avail - committed.get(sid, 0) - MIN_GARRISON)
                if send < MIN_SEND:
                    continue
                actions.append([sid, angle, send])
                committed[sid] = committed.get(sid, 0) + send

    # ── Rear staging ──────────────────────────────────────────────────────────
    if len(actions) < MAX_ACTIONS:
        staging = _rear_staging(
            planets, my_id, ip, av, current_step, arrival_table, committed, front_ids)
        for (sid, angle, ships) in staging:
            if len(actions) >= MAX_ACTIONS:
                break
            actual_avail = int(next((p[5] for p in my_planets if int(p[0]) == sid), 0))
            send = min(ships, actual_avail - committed.get(sid, 0) - MIN_GARRISON)
            if send < MIN_SEND:
                continue
            actions.append([sid, angle, send])
            committed[sid] = committed.get(sid, 0) + send

    return actions
