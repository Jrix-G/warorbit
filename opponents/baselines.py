"""Adversaires triviaux : passive / random / greedy / starter.

starter = port direct du starter_agent embarqué dans kaggle_environments.envs.orbit_wars.
"""

import math
import random as _random


CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0


def passive_agent(obs, config=None):
    return []


def random_agent(obs, config=None):
    planets = obs.get("planets", [])
    me = obs.get("player", 0)
    moves = []
    for p in planets:
        if p[1] == me and p[5] > 10:
            angle = _random.uniform(0, 2 * math.pi)
            moves.append([p[0], angle, p[5] // 2])
    return moves


def greedy_agent(obs, config=None):
    planets = obs.get("planets", [])
    me = obs.get("player", 0)
    my = [p for p in planets if p[1] == me]
    others = [p for p in planets if p[1] != me]
    moves = []
    for src in my:
        if src[5] < 10 or not others:
            continue
        tgt = min(others, key=lambda t: math.hypot(t[2] - src[2], t[3] - src[3]))
        angle = math.atan2(tgt[3] - src[3], tgt[2] - src[2])
        moves.append([src[0], angle, src[5] // 2])
    return moves


def starter_agent(obs, config=None):
    """Starter officiel de l'env: cible la planète statique non-alliée la plus proche."""
    me = obs.get("player", 0)
    planets = obs.get("planets", [])

    static_targets = []
    for p in planets:
        orbital_r = math.hypot(p[2] - CENTER, p[3] - CENTER)
        if orbital_r + p[4] >= ROTATION_RADIUS_LIMIT and p[1] != me:
            static_targets.append(p)

    moves = []
    for mp in planets:
        if mp[1] != me or mp[5] <= 0:
            continue
        if not static_targets:
            continue
        closest = min(static_targets, key=lambda t: math.hypot(t[2] - mp[2], t[3] - mp[3]))
        angle = math.atan2(closest[3] - mp[3], closest[2] - mp[2])
        ships = mp[5] // 2
        if ships >= 20:
            moves.append([mp[0], angle, ships])
    return moves
