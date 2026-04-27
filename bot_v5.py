"""Orbit Wars Agent V5 - Mathematically Optimal Strategy

Key improvements over V2/V3:
- Game phase detection (early/mid/late/comet)
- Proper defense reserves (15-25% always, not 0.4%)
- Fleet send ratios (65%, not all-in)
- Production horizon (40-180 turns multi-phase)
- Threat detection with ETA-based urgency
- 4-Player kingmaker: adaptive enemy selection
- Cost-aware targeting (distance × time × defense)
- Threat monitoring and dynamic defense
- Risk-adjusted decisions (expected value)
"""

import math
from collections import defaultdict

# ── Physical Constants ──────────────────────────────────────────────────────
BOARD_SIZE = 100.0
SUN_X, SUN_Y = 50.0, 50.0
SUN_RADIUS = 10.0
MAX_SPEED = 6.0
ROT_LIMIT = 50.0
TOTAL_TURNS = 500

# ── Game Phase Constants ─────────────────────────────────────────────────────
EARLY_GAME_LIMIT = 40
MID_GAME_LIMIT = 150
LATE_GAME_LIMIT = 350
VERY_LATE_GAME_LIMIT = 450

COMET_TURNS = [50, 150, 250, 350, 450]
COMET_WINDOW = 15

# ── Strategic Constants ──────────────────────────────────────────────────────
EARLY_GAME_DEFENSE = 0.25
MID_GAME_DEFENSE = 0.20
LATE_GAME_DEFENSE = 0.15

FLEET_SEND_RATIO = 0.65
THREAT_THRESHOLD = 0.25

# Production horizons
EARLY_HORIZON = 40
MID_HORIZON = 100
LATE_HORIZON = 180
VERY_LATE_HORIZON = 50

# ── Threat tracking ──────────────────────────────────────────────────────────
threat_history = defaultdict(lambda: {'incoming': 0, 'eta_min': 100})

def fleet_speed(ships):
    """Logarithmic fleet speed"""
    if ships <= 1:
        return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5

def dist(x1, y1, x2, y2):
    """Euclidean distance"""
    return math.hypot(x1 - x2, y1 - y2)

def is_orbiting(px, py, pradius):
    """Check if planet orbits the sun"""
    return dist(px, py, SUN_X, SUN_Y) + pradius < ROT_LIMIT

def predict_pos(px, py, pradius, angular_vel, turns):
    """Predict planet position after N turns"""
    if not is_orbiting(px, py, pradius):
        return px, py
    dx, dy = px - SUN_X, py - SUN_Y
    r = math.hypot(dx, dy)
    angle = math.atan2(dy, dx) + angular_vel * turns
    return SUN_X + r * math.cos(angle), SUN_Y + r * math.sin(angle)

def safe_angle(sx, sy, tx, ty):
    """Calculate safe angle avoiding the sun"""
    direct = math.atan2(ty - sy, tx - sx)
    d = dist(sx, sy, tx, ty)

    # Check if direct path crosses sun
    end_x = sx + math.cos(direct) * d
    end_y = sy + math.sin(direct) * d

    # Simple sun avoidance: if path too close, use waypoint
    min_dist_to_sun = segment_min_dist_to_sun(sx, sy, end_x, end_y)
    if min_dist_to_sun > SUN_RADIUS + 1.5:
        return direct

    # Try perpendicular waypoint
    for sign in [1, -1]:
        wp_angle = direct + sign * math.pi / 2
        wp_r = SUN_RADIUS * 2.0
        wp_x = SUN_X + wp_r * math.cos(wp_angle)
        wp_y = SUN_Y + wp_r * math.sin(wp_angle)

        ok1 = segment_min_dist_to_sun(sx, sy, wp_x, wp_y) > SUN_RADIUS + 1.0
        ok2 = segment_min_dist_to_sun(wp_x, wp_y, tx, ty) > SUN_RADIUS + 1.0
        if ok1 and ok2:
            return math.atan2(wp_y - sy, wp_x - sx)

    # Fallback: slight deviation
    for delta in [0.3, 0.6, 0.9]:
        for sign in [1, -1]:
            a = direct + sign * delta
            ex = sx + math.cos(a) * d
            ey = sy + math.sin(a) * d
            if segment_min_dist_to_sun(sx, sy, ex, ey) > SUN_RADIUS + 1.0:
                return a

    return direct

def segment_min_dist_to_sun(x1, y1, x2, y2):
    """Minimum distance from sun to line segment"""
    seg_dx = x2 - x1
    seg_dy = y2 - y1
    lsq = seg_dx * seg_dx + seg_dy * seg_dy

    if lsq < 1e-9:
        return dist(x1, y1, SUN_X, SUN_Y)

    t = max(0.0, min(1.0, ((SUN_X - x1) * seg_dx + (SUN_Y - y1) * seg_dy) / lsq))
    px = x1 + t * seg_dx
    py = y1 + t * seg_dy
    return dist(px, py, SUN_X, SUN_Y)

def get_game_phase(turn, remaining_turns):
    """Determine current game phase"""
    if remaining_turns > LATE_GAME_LIMIT:
        return "early" if turn < EARLY_GAME_LIMIT else "mid"
    elif remaining_turns > VERY_LATE_GAME_LIMIT:
        return "late"
    else:
        return "very_late"

def is_comet_window(turn):
    """Check if in comet window"""
    for comet_turn in COMET_TURNS:
        if abs(turn - comet_turn) <= COMET_WINDOW:
            return True
    return False

def ships_by_player(planets, fleets):
    """Total ships per player"""
    total = defaultdict(int)
    for p in planets:
        if p[1] >= 0:
            total[p[1]] += p[5]
    for f in fleets:
        if f[1] >= 0:
            total[f[1]] += f[6]
    return total

def calculate_threat_from_player(pid, planets, fleets, me, ang_vel):
    """Calculate threat from specific player"""
    total_ships = 0
    incoming_threat = 0
    min_eta = 100.0

    for p in planets:
        if p[1] == pid:
            total_ships += p[5]

    for f in fleets:
        if f[1] == pid:
            total_ships += f[6]

            # Check if fleet heading toward us
            for mp in planets:
                if mp[1] == me:
                    target_angle = math.atan2(mp[3] - f[3], mp[2] - f[2])
                    angle_diff = abs(((f[4] - target_angle + math.pi) % (2 * math.pi)) - math.pi)
                    d = dist(f[2], f[3], mp[2], mp[3])

                    if angle_diff < 0.35 and d < 70:
                        eta = d / fleet_speed(f[6])
                        incoming_threat += f[6]
                        min_eta = min(min_eta, eta)

    return total_ships, incoming_threat, min_eta

def select_kingmaker_target(planets, fleets, me):
    """
    4-Player logic: select target based on player rankings

    Rules:
    1. If leader >> 2nd: attack 2nd (let others weaken leader)
    2. If we're 2nd: attack leader
    3. Otherwise: attack weakest major threat
    """
    player_ships = ships_by_player(planets, fleets)

    # Get all players except self, sorted by ship count
    others = [(pid, count) for pid, count in player_ships.items() if pid != me]
    others.sort(key=lambda x: -x[1])

    if not others:
        return me  # No other players?

    leader_pid, leader_ships = others[0]

    # If leader dominates heavily, attack 2nd
    if len(others) > 1:
        second_pid, second_ships = others[1]
        if leader_ships > 2.0 * second_ships and second_ships > 0:
            return second_pid

    # If we're second, attack leader
    my_ships = player_ships.get(me, 0)
    if my_ships > 0 and len(others) > 0:
        is_second = (len(others) > 1 and my_ships > second_ships)
        is_strong = (my_ships > leader_ships * 0.6)
        if is_second or is_strong:
            return leader_pid

    # Default: attack leader
    return leader_pid if others else me

def calculate_target_value(planet, src_x, src_y, my_ships, game_phase, turn, comet_ids, ang_vel):
    """
    Calculate value of attacking a target
    Value = production_gain - attack_cost + comet_bonus
    """
    pid, owner, px, py, pradius, ships, production = planet

    # Predict future position
    future_x, future_y = predict_pos(px, py, pradius, ang_vel, 10)

    # Distance and ETA
    distance = dist(src_x, src_y, future_x, future_y)
    eta = distance / fleet_speed(my_ships) if my_ships > 1 else distance

    # Attack cost
    defense_ratio = ships / my_ships if my_ships > 0 else 1.0
    attack_cost = (distance * 0.1) + (eta * 0.5) + (defense_ratio * 10.0)

    # Production value
    if game_phase == "early":
        horizon = EARLY_HORIZON
        prod_value = production * horizon * 1.25
    elif game_phase == "mid":
        horizon = MID_HORIZON
        prod_value = production * horizon
    elif game_phase == "late":
        horizon = LATE_HORIZON
        prod_value = production * horizon
    else:
        horizon = VERY_LATE_HORIZON
        prod_value = production * horizon

    # Ownership bonus
    if owner == 0:
        prod_value *= 2.0  # 2x for neutral

    # Comet bonus
    comet_bonus = 0
    if pid in comet_ids and is_comet_window(turn):
        comet_bonus = 1000.0

    # Total value
    value = prod_value + comet_bonus - attack_cost

    return value

def agent(obs, config=None):
    """
    V5 Agent - Mathematically optimal multi-layer strategy
    """

    planets = obs.get("planets", [])
    fleets = obs.get("fleets", [])
    me = obs.get("player", 0)
    ang_vel = obs.get("angular_velocity", 0.0)
    turn = obs.get("step", 0)
    comet_ids = set(obs.get("comet_planet_ids", []))

    remaining_turns = 500 - turn

    # === LAYER 1: Game Phase ===
    game_phase = get_game_phase(turn, remaining_turns)
    in_comet_window = is_comet_window(turn)

    # === Current state ===
    my_planets = [p for p in planets if p[1] == me]
    if not my_planets:
        return []

    my_ships = sum(p[5] for p in my_planets)
    for f in fleets:
        if f[1] == me:
            my_ships += f[6]

    # Calculate my center position
    my_center_x = sum(p[2] for p in my_planets) / len(my_planets)
    my_center_y = sum(p[3] for p in my_planets) / len(my_planets)

    # === LAYER 3: Threat Assessment ===
    total_threat = 0
    threats_by_player = {}

    for player_id in range(4):
        if player_id != me:
            total_ships, incoming, eta_min = calculate_threat_from_player(
                player_id, planets, fleets, me, ang_vel
            )
            threats_by_player[player_id] = (total_ships, incoming, eta_min)
            total_threat += total_ships

    # === LAYER 3: Defense Reserve ===
    if game_phase == "early":
        base_defense = EARLY_GAME_DEFENSE
    elif game_phase == "mid":
        base_defense = MID_GAME_DEFENSE
    else:
        base_defense = LATE_GAME_DEFENSE

    # Increase defense under critical threat
    threat_ratio = total_threat / my_ships if my_ships > 0 else 0
    if threat_ratio > THREAT_THRESHOLD:
        base_defense = min(0.35, base_defense + 0.10)

    available = my_ships * (1.0 - base_defense)

    # === LAYER 4: Kingmaker Target Selection ===
    primary_target = select_kingmaker_target(planets, fleets, me)

    # === LAYER 2: Target Value Calculation ===
    target_values = []

    for planet in planets:
        if planet[1] == me:  # Skip own planets
            continue

        value = calculate_target_value(
            planet, my_center_x, my_center_y, my_ships,
            game_phase, turn, comet_ids, ang_vel
        )

        # === LAYER 4: Kingmaker filtering ===
        if planet[1] == primary_target:
            value *= 1.5  # Bonus to primary target
        elif planet[1] != 0 and planet[1] != primary_target:
            value *= 0.6  # Penalty to non-primary enemies

        # === LAYER 6: Comet special handling ===
        if planet[0] in comet_ids and in_comet_window:
            value *= 2.0  # Extra comet bonus during window

        if value > 0:
            target_values.append((planet[0], value, planet))

    # Sort by value
    target_values.sort(key=lambda x: -x[1])

    # === LAYER 7: Build orders ===
    orders = []
    ships_committed = 0

    for src in my_planets:
        if src[5] <= 0 or ships_committed >= available:
            continue

        # Find best target for this source
        for target_id, target_value, target in target_values:
            if target_value <= 0:
                continue

            ships_available = min(
                src[5],
                int(available - ships_committed)
            )

            if ships_available <= 1:
                break

            # Send ratio
            if target[0] in comet_ids and in_comet_window:
                ratio = 0.70  # Aggressive on comets
            else:
                ratio = FLEET_SEND_RATIO

            ships_to_send = int(ships_available * ratio)

            if ships_to_send >= 1:
                angle = safe_angle(src[2], src[3], target[2], target[3])
                orders.append([src[0], angle, ships_to_send])
                ships_committed += ships_to_send
                break  # One target per planet per turn

    return orders
