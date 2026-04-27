"""Heuristiques publiques reproduites localement (≠ scraping).

distance_priority_agent  ≈ idée du notebook 'Distance-Prioritized' (LB 1100)
sun_dodging_agent        ≈ idée du notebook 'Sun-Dodging Baseline'

Ces implémentations sont volontairement minimalistes et lisibles; elles
servent de cibles d'entraînement variées, pas de copies fidèles. Pour les
versions réelles voir analysis/scrape.py.
"""

import math


CENTER = 50.0
SUN_RADIUS = 10.0


def _segment_min_dist_to_sun(x1, y1, x2, y2):
    seg_dx, seg_dy = x2 - x1, y2 - y1
    lsq = seg_dx * seg_dx + seg_dy * seg_dy
    if lsq < 1e-9:
        return math.hypot(x1 - CENTER, y1 - CENTER)
    t = max(0.0, min(1.0, ((CENTER - x1) * seg_dx + (CENTER - y1) * seg_dy) / lsq))
    return math.hypot(x1 + t * seg_dx - CENTER, y1 + t * seg_dy - CENTER)


def distance_priority_agent(obs, config=None):
    """Greedy plus malin: cible neutres faibles d'abord, puis ennemis proches."""
    planets = obs.get("planets", [])
    me = obs.get("player", 0)
    my = [p for p in planets if p[1] == me]
    moves = []
    for src in my:
        if src[5] < 12:
            continue
        candidates = []
        for t in planets:
            if t[1] == me:
                continue
            d = math.hypot(t[2] - src[2], t[3] - src[3])
            score = (-d) - 0.5 * t[5] + (8 if t[1] == -1 else 0)
            candidates.append((score, t))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0], reverse=True)
        tgt = candidates[0][1]
        if src[5] > tgt[5] + 5:
            angle = math.atan2(tgt[3] - src[3], tgt[2] - src[2])
            moves.append([src[0], angle, int(src[5] * 0.6)])
    return moves


def sun_dodging_agent(obs, config=None):
    """Greedy + dévie l'angle si trajectoire passe par le soleil."""
    planets = obs.get("planets", [])
    me = obs.get("player", 0)
    my = [p for p in planets if p[1] == me]
    others = [p for p in planets if p[1] != me]
    moves = []
    for src in my:
        if src[5] < 10 or not others:
            continue
        tgt = min(others, key=lambda t: math.hypot(t[2] - src[2], t[3] - src[3]))
        d = math.hypot(tgt[2] - src[2], tgt[3] - src[3])
        ex = src[2] + (tgt[2] - src[2])
        ey = src[3] + (tgt[3] - src[3])
        angle = math.atan2(tgt[3] - src[3], tgt[2] - src[2])
        if _segment_min_dist_to_sun(src[2], src[3], ex, ey) < SUN_RADIUS + 1.5:
            for delta in [0.3, -0.3, 0.6, -0.6, 0.9, -0.9]:
                a = angle + delta
                ex2 = src[2] + math.cos(a) * d
                ey2 = src[3] + math.sin(a) * d
                if _segment_min_dist_to_sun(src[2], src[3], ex2, ey2) > SUN_RADIUS + 1.0:
                    angle = a
                    break
        moves.append([src[0], angle, src[5] // 2])
    return moves
