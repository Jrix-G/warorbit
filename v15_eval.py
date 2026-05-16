"""v15_eval — position evaluator for the V15.3 RCC search.

Scores a FastState from one player's perspective WITHOUT any rollout, from
five position signals (each mapped to [0,1]):

  ship_share    my ships / total ships            (immediate strength)
  prod_share    my production / total production  (economic / snowball power)
  planet_share  my planets / total planets        (territorial control)
  domination    (my_ships - max_opp_ships) norm.  (relative dominance)
  prod_margin   (my_prod - max_opp_prod) norm.    (economic edge vs leader)

Two weight sets share the same five features:

  ESC  — hand-tuned weights (the V15.3 baseline / generation 0). 2p favours
         immediate strength, 4p favours production (snowball).
  learned — weights fitted by the self-play value-iteration loop. Same five
         features, but the weights (and per-feature standardisation) are
         learned from self-play game outcomes instead of guessed.

`evaluate` is given the weight set explicitly (default ESC) so generation N
and generation N-1 can be compared in the same process with zero global
state. The score is a monotone linear function — RCC only needs the ranking,
so no sigmoid is applied (it would not change any argmax).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import v15_fast_sim as fsim

OWNER, SHIPS, PROD = fsim.OWNER, fsim.SHIPS, fsim.PROD
F_OWNER, F_SHIPS = fsim.F_OWNER, fsim.F_SHIPS

_EPS = 1e-9
N_FEATURES = 5

# hand-tuned ESC weights — feature order:
# ship_share, prod_share, planet_share, domination, prod_margin
_W_2P = np.array([0.40, 0.30, 0.05, 0.15, 0.10])
_W_4P = np.array([0.25, 0.35, 0.05, 0.20, 0.15])


@dataclass
class EvalWeights:
    """A weight set for the evaluator. score = ((features - mean)/std) @ w.

    For the hand-tuned ESC, mean=0 and std=1 (no standardisation). For a
    learned value function, all three are fitted on self-play data."""
    w2p: np.ndarray
    w4p: np.ndarray
    mean2p: np.ndarray
    std2p: np.ndarray
    mean4p: np.ndarray
    std4p: np.ndarray
    tag: str = "esc"

    def save(self, path: str) -> None:
        np.savez(path, w2p=self.w2p, w4p=self.w4p,
                 mean2p=self.mean2p, std2p=self.std2p,
                 mean4p=self.mean4p, std4p=self.std4p,
                 tag=np.array(self.tag))

    @staticmethod
    def load(path: str) -> "EvalWeights":
        d = np.load(path, allow_pickle=True)
        return EvalWeights(
            w2p=d["w2p"], w4p=d["w4p"],
            mean2p=d["mean2p"], std2p=d["std2p"],
            mean4p=d["mean4p"], std4p=d["std4p"],
            tag=str(d["tag"]) if "tag" in d.files else "learned",
        )


# generation-0 weights: the hand-tuned ESC, no standardisation
ESC = EvalWeights(
    w2p=_W_2P.copy(), w4p=_W_4P.copy(),
    mean2p=np.zeros(N_FEATURES), std2p=np.ones(N_FEATURES),
    mean4p=np.zeros(N_FEATURES), std4p=np.ones(N_FEATURES),
    tag="esc",
)


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


def evaluate(fs: fsim.FastState, player: int,
             weights: EvalWeights = ESC) -> float:
    """Position score for `player`. Higher = better. Monotone linear; with
    `weights=ESC` it lies in [0,1], with learned weights it is a logit-scale
    score (only the ranking is used by RCC)."""
    f = features(fs, player)
    if fs.n_players >= 4:
        w, m, s = weights.w4p, weights.mean4p, weights.std4p
    else:
        w, m, s = weights.w2p, weights.mean2p, weights.std2p
    return float(((f - m) / s) @ w)
