from __future__ import annotations

import math
from typing import Any, Dict, Tuple


CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
SUN_SHOT_MARGIN = 4.0
FLEET_SPEED = 6.0
SHOT_HORIZON = 90
INTERCEPT_TOLERANCE = 1.35
SUN_GUARD_RAY_DISTANCE = 150.0


def _planet_id(planet: Dict[str, Any]) -> int:
    return int(planet.get("id", -1))


def _point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def path_sun_clearance(src_x: float, src_y: float, angle: float, distance: float | None = None) -> float:
    ray_distance = max(float(distance or 0.0), SUN_GUARD_RAY_DISTANCE)
    end_x = src_x + math.cos(angle) * ray_distance
    end_y = src_y + math.sin(angle) * ray_distance
    return _point_segment_distance(CENTER_X, CENTER_Y, src_x, src_y, end_x, end_y)


def path_too_close_to_sun(src_x: float, src_y: float, angle: float, distance: float | None = None) -> bool:
    return path_sun_clearance(src_x, src_y, angle, distance) < SUN_RADIUS + SUN_SHOT_MARGIN


def _initial_by_id(game: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {_planet_id(p): p for p in game.get("initial_planets", []) if isinstance(p, dict)}


def predict_planet_position(planet: Dict[str, Any], game: Dict[str, Any], turns: float) -> Tuple[float, float]:
    angular_velocity = float(game.get("angular_velocity", 0.0) or 0.0)
    initial = _initial_by_id(game).get(_planet_id(planet))
    if initial is None or abs(angular_velocity) < 1e-12:
        return float(planet.get("x", 0.0)), float(planet.get("y", 0.0))

    init_x = float(initial.get("x", planet.get("x", 0.0)))
    init_y = float(initial.get("y", planet.get("y", 0.0)))
    radius = math.hypot(init_x - CENTER_X, init_y - CENTER_Y)
    if radius <= 1e-9:
        return init_x, init_y
    theta0 = math.atan2(init_y - CENTER_Y, init_x - CENTER_X)
    theta = theta0 + angular_velocity * (float(game.get("turn", 0.0) or 0.0) + max(0.0, float(turns)))
    return CENTER_X + math.cos(theta) * radius, CENTER_Y + math.sin(theta) * radius


def _target_too_close_to_sun(x: float, y: float, target_radius: float) -> bool:
    return math.hypot(x - CENTER_X, y - CENTER_Y) - max(0.0, target_radius) < SUN_RADIUS + SUN_SHOT_MARGIN


def _angle_if_safe(src_x: float, src_y: float, tgt_x: float, tgt_y: float, target_radius: float) -> float | None:
    distance = math.hypot(tgt_x - src_x, tgt_y - src_y)
    if distance <= 1e-9:
        return None
    if _target_too_close_to_sun(tgt_x, tgt_y, target_radius):
        return None
    angle = math.atan2(tgt_y - src_y, tgt_x - src_x)
    if path_too_close_to_sun(src_x, src_y, angle, distance):
        return None
    return float(angle)


def safe_plan_shot(src: Dict[str, Any], tgt: Dict[str, Any], game: Dict[str, Any]) -> float | None:
    """Return a safe firing angle, or None if the target line is sun-blocked."""
    src_x = float(src.get("x", 0.0))
    src_y = float(src.get("y", 0.0))
    target_radius = float(tgt.get("radius", 0.0) or 0.0)
    if math.hypot(src_x - CENTER_X, src_y - CENTER_Y) < SUN_RADIUS + SUN_SHOT_MARGIN:
        return None

    current_tx, current_ty = predict_planet_position(tgt, game, 0.0)
    eta = math.hypot(current_tx - src_x, current_ty - src_y) / FLEET_SPEED
    for _ in range(10):
        eta = min(float(SHOT_HORIZON), max(0.0, eta))
        target_x, target_y = predict_planet_position(tgt, game, eta)
        eta = math.hypot(target_x - src_x, target_y - src_y) / FLEET_SPEED

    if eta <= SHOT_HORIZON:
        target_x, target_y = predict_planet_position(tgt, game, eta)
        angle = _angle_if_safe(src_x, src_y, target_x, target_y, target_radius)
        if angle is not None:
            return angle

    start = max(0, int(math.hypot(current_tx - src_x, current_ty - src_y) / FLEET_SPEED) - 8)
    best: tuple[float, float] | None = None
    for turn in range(start, SHOT_HORIZON + 1):
        target_x, target_y = predict_planet_position(tgt, game, float(turn))
        distance = math.hypot(target_x - src_x, target_y - src_y)
        arrival = distance / FLEET_SPEED
        error = abs(arrival - float(turn))
        if error > INTERCEPT_TOLERANCE:
            continue
        angle = _angle_if_safe(src_x, src_y, target_x, target_y, target_radius)
        if angle is None:
            continue
        if best is None or error < best[0]:
            best = (error, angle)
    return None if best is None else best[1]
