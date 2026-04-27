"""Orbit Wars V6 beam search.

Numpy-only tactical search over a small set of generated action plans. The
simulator is copied once per candidate, then advanced in place.
"""

import math
import time

import numpy as np

import eval
import sim


COMET_TURNS = (50, 150, 250, 350, 450)


def _planet_rows(state, owner):
    if len(state.planets) == 0:
        return np.zeros((0, 7), dtype=np.float32)
    mask = state.planets[:, sim.P_OWNER] == owner
    return state.planets[mask] if np.any(mask) else np.zeros((0, 7), dtype=np.float32)


def _targets_not_owned(state, player):
    if len(state.planets) == 0:
        return np.zeros((0, 7), dtype=np.float32)
    mask = state.planets[:, sim.P_OWNER] != player
    return state.planets[mask] if np.any(mask) else np.zeros((0, 7), dtype=np.float32)


def _angle_to(src, target):
    return math.atan2(
        float(target[sim.P_Y]) - float(src[sim.P_Y]),
        float(target[sim.P_X]) - float(src[sim.P_X]),
    )


def _active_comet_window(step):
    for comet_turn in COMET_TURNS:
        if abs(int(step) - comet_turn) < 15:
            return True
    return False


def _heuristic_actions(state, player):
    """Fast V5-like opponent model used inside rollouts."""
    my_planets = _planet_rows(state, player)
    if len(my_planets) == 0:
        return []

    targets = _targets_not_owned(state, player)
    if len(targets) == 0:
        return []

    actions = []
    tx = targets[:, sim.P_X]
    ty = targets[:, sim.P_Y]
    prod = targets[:, sim.P_PROD]

    for src in my_planets:
        ships = int(src[sim.P_SHIPS])
        if ships <= 5:
            continue

        dx = tx - src[sim.P_X]
        dy = ty - src[sim.P_Y]
        dist = np.hypot(dx, dy)
        scores = prod * 30.0 - dist * 0.5
        target = targets[int(np.argmax(scores))]
        n_ships = int(ships * 0.60)
        if n_ships > 0:
            actions.append([int(src[sim.P_ID]), _angle_to(src, target), n_ships])

    return actions


def _best_aggressive_target(state, src, me):
    targets = state.planets[state.planets[:, sim.P_OWNER] != me]
    if len(targets) == 0:
        return None

    enemy = targets[targets[:, sim.P_OWNER] >= 0]
    target_pool = enemy if len(enemy) > 0 else targets
    dist = np.hypot(target_pool[:, sim.P_X] - src[sim.P_X],
                    target_pool[:, sim.P_Y] - src[sim.P_Y])
    score = target_pool[:, sim.P_PROD] * 18.0 - target_pool[:, sim.P_SHIPS] * 0.45 - dist * 0.8
    return target_pool[int(np.argmax(score))]


def _best_production_target(state, src):
    neutral = state.planets[state.planets[:, sim.P_OWNER] == -1]
    if len(neutral) == 0:
        return None
    dist = np.hypot(neutral[:, sim.P_X] - src[sim.P_X],
                    neutral[:, sim.P_Y] - src[sim.P_Y])
    score = neutral[:, sim.P_PROD] * 35.0 - neutral[:, sim.P_SHIPS] * 0.6 - dist * 0.7
    return neutral[int(np.argmax(score))]


def _threatened_planets(state, me):
    my_planets = _planet_rows(state, me)
    if len(my_planets) == 0 or len(state.fleets) == 0:
        return []

    enemies = state.fleets[state.fleets[:, sim.F_OWNER] != me]
    threatened = []
    for p in my_planets:
        incoming = 0.0
        nearest = 999.0
        for f in enemies:
            dx = float(p[sim.P_X]) - float(f[sim.F_X])
            dy = float(p[sim.P_Y]) - float(f[sim.F_Y])
            d = math.hypot(dx, dy)
            if d > 45.0:
                continue
            target_angle = math.atan2(dy, dx)
            diff = abs(((float(f[sim.F_ANGLE]) - target_angle + math.pi) % (2.0 * math.pi)) - math.pi)
            if diff < 0.40:
                incoming += float(f[sim.F_SHIPS])
                nearest = min(nearest, d)
        if incoming > float(p[sim.P_SHIPS]) * 0.35:
            threatened.append((p, incoming, nearest))
    threatened.sort(key=lambda x: (-(x[1] - float(x[0][sim.P_SHIPS])), x[2]))
    return threatened


def _best_defensive_target(state, src, me):
    threatened = _threatened_planets(state, me)
    for target, _, _ in threatened:
        if int(target[sim.P_ID]) != int(src[sim.P_ID]):
            return target

    my_planets = _planet_rows(state, me)
    weaker = []
    for p in my_planets:
        if int(p[sim.P_ID]) != int(src[sim.P_ID]):
            weaker.append(p)
    if not weaker:
        return None

    arr = np.array(weaker, dtype=np.float32)
    dist = np.hypot(arr[:, sim.P_X] - src[sim.P_X], arr[:, sim.P_Y] - src[sim.P_Y])
    score = -arr[:, sim.P_SHIPS] - dist * 0.2 + arr[:, sim.P_PROD] * 2.0
    return arr[int(np.argmax(score))]


def _best_comet_target(state, src):
    if not _active_comet_window(state.step) or len(state.comet_ids) == 0:
        return None
    comet_rows = []
    for p in state.planets:
        if int(p[sim.P_ID]) in state.comet_ids:
            comet_rows.append(p)
    if not comet_rows:
        return None
    arr = np.array(comet_rows, dtype=np.float32)
    dist = np.hypot(arr[:, sim.P_X] - src[sim.P_X], arr[:, sim.P_Y] - src[sim.P_Y])
    score = arr[:, sim.P_PROD] * 100.0 - arr[:, sim.P_SHIPS] * 0.25 - dist
    return arr[int(np.argmax(score))]


def _fallback_target(state, src, me):
    target = _best_production_target(state, src)
    if target is not None:
        return target
    return _best_aggressive_target(state, src, me)


def _generate_action_set(state, me, strategy, best_actions=None):
    """Generate one plan according to a fixed strategic bias."""
    my_planets = _planet_rows(state, me)
    if len(my_planets) == 0:
        return []

    if strategy == 4 and best_actions:
        actions = []
        pid_to_ships = {
            int(p[sim.P_ID]): int(p[sim.P_SHIPS])
            for p in my_planets
        }
        for from_id, angle, ships in best_actions:
            avail = pid_to_ships.get(int(from_id), 0)
            if avail <= 5:
                continue
            angle_delta = float(np.random.uniform(-0.20, 0.20))
            ship_mult = float(np.random.uniform(0.80, 1.20))
            n_ships = min(avail, max(1, int(float(ships) * ship_mult)))
            actions.append([int(from_id), float(angle) + angle_delta, n_ships])
        if actions:
            return actions
        strategy = 0

    actions = []
    for src in my_planets:
        ships = int(src[sim.P_SHIPS])
        if ships <= 5:
            continue

        if strategy == 0:
            ratio = float(np.random.uniform(0.65, 0.80))
            target = _best_aggressive_target(state, src, me)
        elif strategy == 1:
            ratio = float(np.random.uniform(0.30, 0.50))
            target = _best_defensive_target(state, src, me)
        elif strategy == 2:
            ratio = float(np.random.uniform(0.50, 0.65))
            target = _best_production_target(state, src)
        elif strategy == 3:
            ratio = 0.78
            target = _best_comet_target(state, src)
            if target is None:
                target = _fallback_target(state, src, me)
                ratio = 0.60
        else:
            ratio = float(np.random.uniform(0.62, 0.74))
            target = _best_aggressive_target(state, src, me)

        if target is None:
            continue
        n_ships = int(ships * ratio)
        if n_ships > 0:
            actions.append([int(src[sim.P_ID]), _angle_to(src, target), n_ships])

    return actions


def generate_candidates(state, me, n_candidates):
    """Return `n_candidates` diverse action plans."""
    candidates = []
    best_actions = None
    for i in range(int(n_candidates)):
        strategy = i % 5
        actions = _generate_action_set(state, me, strategy, best_actions)
        candidates.append(actions)
        if best_actions is None and actions:
            best_actions = actions
    return candidates


def _active_players(state, me):
    players = set()
    if len(state.planets) > 0:
        for owner in state.planets[:, sim.P_OWNER]:
            owner_int = int(owner)
            if owner_int >= 0 and owner_int != me:
                players.add(owner_int)
    if len(state.fleets) > 0:
        for owner in state.fleets[:, sim.F_OWNER]:
            owner_int = int(owner)
            if owner_int >= 0 and owner_int != me:
                players.add(owner_int)
    return sorted(players)


def beam_search(state, time_budget=0.85, evaluator=None):
    """Search the best immediate action plan under a strict time budget."""
    t_start = time.time()
    me = int(state.player)
    n_candidates = min(100, max(20, int(50 * float(time_budget) / 0.85)))
    horizon = min(25, max(10, int(20 * float(time_budget) / 0.85)))
    other_players = _active_players(state, me)
    candidates = generate_candidates(state, me, n_candidates)

    best_score = -1.0e18
    best_actions = []
    gamma = 0.97

    for my_actions in candidates:
        if time.time() - t_start > float(time_budget) * 0.90:
            break

        sim_state = state.copy()
        score = 0.0
        discount = 1.0

        for _ in range(horizon):
            all_actions = {me: my_actions}
            for opp in other_players:
                all_actions[opp] = _heuristic_actions(sim_state, opp)

            sim.step_inplace(sim_state, all_actions)
            score += eval.evaluate_state(sim_state, me, evaluator) * discount
            discount *= gamma

            if sim.is_terminal(sim_state):
                break
            if time.time() - t_start > float(time_budget) * 0.90:
                break

        score += eval.evaluate_state(sim_state, me, evaluator) * discount * 5.0
        if score > best_score:
            best_score = score
            best_actions = my_actions

    return best_actions if best_actions is not None else []
