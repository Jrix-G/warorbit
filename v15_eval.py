"""v15_eval — Composite Static Evaluator (ESC) for the V15.3 RCC search.

Scores a FastState from one player's perspective WITHOUT any rollout. The
score is a deterministic function of five position signals, each mapped to
[0,1], combined with mode-specific weights:

  ship_share    my ships / total ships            (immediate strength)
  prod_share    my production / total production  (economic / snowball power)
  planet_share  my planets / total planets        (territorial control)
  domination    (my_ships - max_opp_ships) norm.  (relative dominance)
  prod_margin   (my_prod - max_opp_prod) norm.    (economic edge vs leader)

2p weights favour immediate strength (direct confrontation); 4p weights
favour production, because the snowball is the dominant 4p win factor — this
is confirmed empirically by the logistic value-function coefficients fitted
on 2631 top-10 replays (prod_share = +1.67 was the strongest winning signal).

The evaluator is deterministic: same state always yields the same score. It
introduces zero variance, unlike a Monte-Carlo rollout estimate.
"""

from __future__ import annotations

import numpy as np

import v15_fast_sim as fsim

OWNER, SHIPS, PROD = fsim.OWNER, fsim.SHIPS, fsim.PROD
F_OWNER, F_SHIPS = fsim.F_OWNER, fsim.F_SHIPS

_EPS = 1e-9

# weight vector order: ship_share, prod_share, planet_share, domination, prod_margin
_W_2P = np.array([0.40, 0.30, 0.05, 0.15, 0.10])
_W_4P = np.array([0.25, 0.35, 0.05, 0.20, 0.15])


def player_totals(fs: fsim.FastState):
    """Return (ships, prod, planets) — float arrays of length n_players.

    ships  = planet garrisons + in-flight fleet ships, per player
    prod   = sum of planet production, per player
    planets= planet count, per player
    """
    n = fs.n_players
    ships = np.zeros(n)
    prod = np.zeros(n)
    planets = np.zeros(n)
    if len(fs.planets):
        po = fs.planets[:, OWNER].astype(np.int64)
        for p in range(n):
            m = po == p
            if m.any():
                ships[p] += fs.planets[m, SHIPS].sum()
                prod[p] += fs.planets[m, PROD].sum()
                planets[p] += int(m.sum())
    if len(fs.fleets):
        fo = fs.fleets[:, F_OWNER].astype(np.int64)
        for p in range(n):
            m = fo == p
            if m.any():
                ships[p] += fs.fleets[m, F_SHIPS].sum()
    return ships, prod, planets


def features(fs: fsim.FastState, player: int) -> np.ndarray:
    """Five position signals for `player`, each clipped to [0,1]."""
    ships, prod, planets = player_totals(fs)
    n = fs.n_players
    tot_s = float(ships.sum())
    tot_p = float(prod.sum())
    tot_pl = float(planets.sum())

    ship_share = ships[player] / tot_s if tot_s > _EPS else 1.0 / n
    prod_share = prod[player] / tot_p if tot_p > _EPS else 1.0 / n
    planet_share = planets[player] / tot_pl if tot_pl > _EPS else 1.0 / n

    opp = [q for q in range(n) if q != player]
    max_opp_s = max((ships[q] for q in opp), default=0.0)
    max_opp_p = max((prod[q] for q in opp), default=0.0)

    denom_s = ships[player] + max_opp_s
    domination = ((ships[player] - max_opp_s) / denom_s) if denom_s > _EPS else 0.0
    prod_margin = ((prod[player] - max_opp_p) / tot_p) if tot_p > _EPS else 0.0

    # map the two signed signals from [-1,1] into [0,1]
    domination = 0.5 * (domination + 1.0)
    prod_margin = 0.5 * (prod_margin + 1.0)

    return np.array([
        min(max(ship_share, 0.0), 1.0),
        min(max(prod_share, 0.0), 1.0),
        min(max(planet_share, 0.0), 1.0),
        min(max(domination, 0.0), 1.0),
        min(max(prod_margin, 0.0), 1.0),
    ])


def evaluate(fs: fsim.FastState, player: int) -> float:
    """ESC score in [0,1]. Higher means a better position for `player`."""
    f = features(fs, player)
    w = _W_4P if fs.n_players >= 4 else _W_2P
    return float(f @ w)
