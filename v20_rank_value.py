"""Rank-aware 4p leaf correction for V20 search.

The scorer is intended as a weak residual on top of standard ESC, not as a
replacement evaluator.  It rewards stable first-place pressure, survival,
production and planet curves, and penalizes last-place leaves plus positions
where a non-us leader is being fed.
"""

from __future__ import annotations

import math

import numpy as np

import v15_bc
import v15_eval
import v15_fast_sim as fsim

_EPS = 1e-9


def eval_combo(
    fs: fsim.FastState,
    player: int,
    combo: list,
    horizon: int,
    bc_cont: bool,
    weights: v15_eval.EvalWeights | None,
    correction_weight: float,
) -> float:
    """Mirror V15 combo evaluation and add a weak 4p rank-aware correction."""
    st = combo_leaf(fs, player, combo, horizon, bc_cont)
    esc = v15_eval.evaluate(st, player, weights if weights is not None else v15_eval.ESC)
    if correction_weight <= 0.0 or int(getattr(st, "n_players", 2) or 2) < 4:
        return esc
    residual = rank_value(st, player) - esc
    return esc + float(correction_weight) * residual


def combo_leaf(
    fs: fsim.FastState,
    player: int,
    combo: list,
    horizon: int,
    bc_cont: bool,
) -> fsim.FastState:
    """Apply combo then deterministic continuation, returning the leaf state."""
    n = int(getattr(fs, "n_players", 2) or 2)
    actions = v15_bc.bc_policy(fs) if bc_cont else [[] for _ in range(n)]
    actions[player] = list(combo)
    st = fsim.step(fs, actions)
    for _ in range(max(0, int(horizon) - 1)):
        if st.done:
            break
        cont = v15_bc.bc_policy(st) if bc_cont else [[] for _ in range(n)]
        st = fsim.step(st, cont)
    return st


def rank_value(st: fsim.FastState, player: int) -> float:
    """Return a 4p leaf value in [0, 1]."""
    scores = np.asarray(fsim.scores(st), dtype=np.float64)
    if scores.size <= int(player) or scores.size < 2:
        return 0.0

    garrison, fleet, prod, planets = v15_eval.player_totals(st)
    ships = garrison + fleet
    alive = (ships > _EPS) | (planets > _EPS)

    rank_score = _rank_score(ships, player)
    first_pressure = _first_pressure(ships, prod, player)
    survival = _survival_curve(float(ships[player]), float(planets[player]), bool(alive[player]))
    prod_curve = _share_curve(float(prod[player]), float(prod.sum()), floor=0.08)
    planet_curve = _share_curve(float(planets[player]), float(planets.sum()), floor=0.08)
    last_penalty = _last_penalty(ships, planets, player)
    feed_penalty = _anti_feed_leader(ships, prod, player)
    middle = _middle_stability(ships, prod, planets, player)

    value = (
        0.28 * rank_score
        + 0.16 * first_pressure
        + 0.20 * survival
        + 0.16 * prod_curve
        + 0.11 * planet_curve
        + 0.09 * middle
        - 0.16 * last_penalty
        - 0.12 * feed_penalty
    )
    return _clip01(value)


def _rank_score(ships: np.ndarray, player: int) -> float:
    order = sorted(range(len(ships)), key=lambda p: (-float(ships[p]), p))
    rank = order.index(int(player))
    return 1.0 - rank / max(1, len(ships) - 1)


def _first_pressure(ships: np.ndarray, prod: np.ndarray, player: int) -> float:
    power = ships + 8.0 * prod
    my = float(power[player])
    opp_best = max((float(power[p]) for p in range(len(power)) if p != player), default=0.0)
    return _clip01(0.5 + 0.5 * (my - opp_best) / (my + opp_best + _EPS))


def _survival_curve(my_ships: float, my_planets: float, alive: bool) -> float:
    if not alive:
        return 0.0
    base = 0.50 if my_planets > 0.0 else 0.25
    return _clip01(base + 0.50 * (1.0 - math.exp(-max(0.0, my_ships) / 36.0)))


def _share_curve(value: float, total: float, *, floor: float) -> float:
    if total <= _EPS:
        return floor
    share = max(0.0, value / total)
    return _clip01(floor + (1.0 - floor) * math.sqrt(min(1.0, share * 4.0)))


def _last_penalty(ships: np.ndarray, planets: np.ndarray, player: int) -> float:
    alive = (ships > _EPS) | (planets > _EPS)
    if not alive[player]:
        return 1.0
    alive_players = [p for p in range(len(ships)) if alive[p]]
    if len(alive_players) <= 1:
        return 0.0
    my = float(ships[player] + 5.0 * planets[player])
    weakest_other = min(float(ships[p] + 5.0 * planets[p]) for p in alive_players if p != player)
    if my >= weakest_other:
        return 0.0
    return _clip01((weakest_other - my) / (weakest_other + my + _EPS))


def _anti_feed_leader(ships: np.ndarray, prod: np.ndarray, player: int) -> float:
    power = ships + 8.0 * prod
    order = sorted(range(len(power)), key=lambda p: -float(power[p]))
    if not order or order[0] == player:
        return 0.0
    leader = order[0]
    runner = order[1] if len(order) > 1 else leader
    total = float(power.sum()) + _EPS
    leader_share = float(power[leader]) / total
    gap = float(power[leader] - power[runner]) / total
    return _clip01(0.55 * max(0.0, leader_share - 0.34) / 0.41 + 0.45 * gap)


def _middle_stability(
    ships: np.ndarray,
    prod: np.ndarray,
    planets: np.ndarray,
    player: int,
) -> float:
    power = ships + 8.0 * prod + 4.0 * planets
    my = float(power[player])
    stronger = sum(1 for p in range(len(power)) if p != player and power[p] > my)
    weaker = sum(1 for p in range(len(power)) if p != player and power[p] < my)
    if stronger == 0:
        return 0.75
    if weaker == 0:
        return 0.15
    return 0.55


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
