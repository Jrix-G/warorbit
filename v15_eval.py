"""v15_eval — position evaluator for the V15.3/4 RCC search.

Scores a FastState from one player's perspective WITHOUT any rollout, from
eleven position signals, each in [0,1]:

   0 ship_share         my ships / total ships
   1 prod_share         my production / total production
   2 planet_share       my planets / total planets
   3 domination         (my_ships - max_opp_ships) normalised
   4 prod_margin        (my_prod - max_opp_prod) normalised
   5 fleet_share        my in-flight ships / my total ships  (commitment)
   6 elim_share         opponents eliminated / (n-1)
   7 top_planet_prod    my richest planet's prod / total prod
   8 ship_concentration my biggest garrison / my total ships
   9 step_frac          game progress
  10 enemy_fleet_press  opponents' in-flight ships / total ships  (threat)

Two weight sets share these features:

  ESC  — hand-tuned weights (generation 0). Only the first five features are
         weighted; signals 5-10 carry weight 0, so the ESC behaves exactly as
         the original 5-signal evaluator.
  learned — weights fitted by the self-play value-iteration loop, free to use
         all eleven signals. This is where the loop gains ground on the ESC.

`evaluate` takes the weight set explicitly (default ESC) so two generations
can be compared in one process with zero global state. The score is a
monotone linear function — RCC needs only the ranking, so no sigmoid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import v15_fast_sim as fsim

OWNER, SHIPS, PROD = fsim.OWNER, fsim.SHIPS, fsim.PROD
X, Y, R = fsim.X, fsim.Y, fsim.R
F_OWNER, F_SHIPS = fsim.F_OWNER, fsim.F_SHIPS
F_X, F_Y, F_ANGLE = fsim.F_X, fsim.F_Y, fsim.F_ANGLE
BOARD_SIZE = fsim.BOARD_SIZE

_EPS = 1e-9
# Features 0-10  : V15 global standing (original ESC basis).
# Features 11-23 : V18 expansion basis — geometry / threat / tempo / opportunity
#                  / coalition. ESC weights for 5-23 are 0.0, so the ESC scores
#                  EXACTLY as V15; the ES loop is what gives them weight. This is
#                  the lever that lifts the V15 ceiling without a neural net.
N_FEATURES = 24
_N_EXTRA = 13
_DIAG = float(np.hypot(BOARD_SIZE, BOARD_SIZE))
_TAU = 10.0  # tempo decay (turns) for ETA-weighted enemy fleet pressure

# hand-tuned ESC weights for the first five features; signals 5-23 -> 0.0
_W_2P = np.array([0.40, 0.30, 0.05, 0.15, 0.10] + [0.0] * 19)
_W_4P = np.array([0.25, 0.35, 0.05, 0.20, 0.15] + [0.0] * 19)


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
    """Return (garrison, fleet, prod, planets) — float arrays of length
    n_players. garrison = planet ship counts; fleet = in-flight ship counts;
    prod = planet production; planets = planet count."""
    n = fs.n_players
    garrison = np.zeros(n)
    fleet = np.zeros(n)
    prod = np.zeros(n)
    planets = np.zeros(n)
    if len(fs.planets):
        po = fs.planets[:, OWNER].astype(np.int64)
        for p in range(n):
            m = po == p
            if m.any():
                garrison[p] += fs.planets[m, SHIPS].sum()
                prod[p] += fs.planets[m, PROD].sum()
                planets[p] += int(m.sum())
    if len(fs.fleets):
        fo = fs.fleets[:, F_OWNER].astype(np.int64)
        for p in range(n):
            m = fo == p
            if m.any():
                fleet[p] += fs.fleets[m, F_SHIPS].sum()
    return garrison, fleet, prod, planets


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def _extra_features(fs: fsim.FastState, player: int,
                    ships_by_player: np.ndarray) -> np.ndarray:
    """13 V18 signals (geometry/threat/tempo/opportunity/coalition), clipped.

    `ships_by_player` is garrison+fleet power per player (from player_totals)."""
    n = fs.n_players
    P = fs.planets
    frontier = dispersion = contested = 0.5
    incoming = at_risk = 0.0
    worst = 0.5
    eta_enemy = free_cap = expansion = 0.0
    if len(P):
        po = P[:, OWNER].astype(np.int64)
        px, py, pr = P[:, X], P[:, Y], P[:, R]
        psh, ppr = P[:, SHIPS], P[:, PROD]
        mine = po == player
        enemy = (po >= 0) & (po != player)
        neutral = po < 0
        tot_p = float(ppr.sum())
        if mine.any():
            mx, my_ = px[mine], py[mine]
            if enemy.any():
                ex, ey = px[enemy], py[enemy]
                d = np.hypot(mx[:, None] - ex[None, :], my_[:, None] - ey[None, :])
                frontier = 1.0 - min(float(d.min(axis=1).mean()) / _DIAG, 1.0)
            cx, cy = float(mx.mean()), float(my_.mean())
            disp = float(np.hypot(mx - cx, my_ - cy).mean())
            dispersion = 1.0 - min(disp / (_DIAG * 0.5), 1.0)
            if neutral.any():
                nx, ny = px[neutral], py[neutral]
                dmy = np.hypot(nx[:, None] - mx[None, :], ny[:, None] - my_[None, :]).min(axis=1)
                if enemy.any():
                    den = np.hypot(nx[:, None] - px[enemy][None, :],
                                   ny[:, None] - py[enemy][None, :]).min(axis=1)
                else:
                    den = np.full_like(dmy, np.inf)
                contested = float((dmy < den).mean())
            if len(fs.fleets):
                fo = fs.fleets[:, F_OWNER].astype(np.int64)
                efl = (fo >= 0) & (fo != player)
                if efl.any():
                    fx, fy = fs.fleets[efl, F_X], fs.fleets[efl, F_Y]
                    fang, fsh = fs.fleets[efl, F_ANGLE], fs.fleets[efl, F_SHIPS]
                    dcos, dsin = np.cos(fang), np.sin(fang)
                    mrp, msh = pr[mine], psh[mine]
                    wx = mx[None, :] - fx[:, None]
                    wy = my_[None, :] - fy[:, None]
                    t = wx * dcos[:, None] + wy * dsin[:, None]
                    perp = np.abs(-wx * dsin[:, None] + wy * dcos[:, None])
                    hit = (t > 0) & (perp <= (mrp[None, :] + 2.0))
                    threat = (fsh[:, None] * hit).sum(axis=0)
                    incoming = min(float(threat.sum()) / (float(msh.sum()) + _EPS), 1.0)
                    at_risk = float((threat > msh).mean())
                    diff = (threat - msh) / (threat + msh + _EPS)
                    worst = 0.5 * (float(diff.max()) + 1.0)
                    speed = max(float(fs.ship_speed), 1.0)
                    eta = np.hypot(wx, wy).min(axis=1) / speed
                    tot_s = float(ships_by_player.sum())
                    if tot_s > _EPS:
                        eta_enemy = min(float((fsh * np.exp(-eta / _TAU)).sum()) / tot_s, 1.0)
            targets = ~mine
            if targets.any() and tot_p > _EPS:
                tx, ty = px[targets], py[targets]
                tsh, tpr, town = psh[targets], ppr[targets], po[targets]
                d2 = np.hypot(tx[:, None] - mx[None, :], ty[:, None] - my_[None, :])
                force = psh[mine][d2.argmin(axis=1)] * 0.5
                cap = force > tsh
                free_cap = min(float(tpr[cap].sum()) / tot_p, 1.0)
                expansion = float(((town < 0) & cap).sum()) / max(1, len(P))
    me_pow = float(ships_by_player[player])
    opp = [q for q in range(n) if q != player]
    max_opp = max((float(ships_by_player[q]) for q in opp), default=0.0)
    tot = float(ships_by_player.sum())
    am_leader = 1.0 if me_pow >= max_opp else 0.0
    gap_to_leader = min(max(max_opp - me_pow, 0.0) / (tot + _EPS), 1.0)
    lead_margin = min(max(me_pow - max_opp, 0.0) / (tot + _EPS), 1.0) * am_leader
    gangup = min((tot - me_pow) / (me_pow + _EPS), 1.0)
    out = np.array([frontier, dispersion, contested, incoming, at_risk, worst,
                    eta_enemy, free_cap, expansion, am_leader, gap_to_leader,
                    lead_margin, gangup])
    return np.clip(out, 0.0, 1.0)


def features(fs: fsim.FastState, player: int) -> np.ndarray:
    """Eleven position signals for `player`, each clipped to [0,1]."""
    garrison, fleet, prod, planets = player_totals(fs)
    n = fs.n_players
    ships = garrison + fleet
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
    domination = 0.5 * (domination + 1.0)
    prod_margin = 0.5 * (prod_margin + 1.0)

    my_ships = float(ships[player])
    fleet_share = fleet[player] / my_ships if my_ships > _EPS else 0.0
    elim = sum(1 for q in opp
               if (garrison[q] + fleet[q]) <= _EPS and planets[q] <= _EPS)
    elim_share = elim / max(1, n - 1)

    # my single richest planet's production share
    top_prod = 0.0
    biggest_garrison = 0.0
    if len(fs.planets):
        po = fs.planets[:, OWNER].astype(np.int64)
        mine = fs.planets[po == player]
        if len(mine):
            top_prod = float(mine[:, PROD].max())
            biggest_garrison = float(mine[:, SHIPS].max())
    top_planet_prod = top_prod / tot_p if tot_p > _EPS else 0.0
    ship_concentration = biggest_garrison / my_ships if my_ships > _EPS else 0.0

    enemy_fleet = sum(fleet[q] for q in opp)
    enemy_fleet_press = enemy_fleet / tot_s if tot_s > _EPS else 0.0

    step_frac = min(fs.step / 500.0, 1.0)

    base = np.array([
        _clip01(ship_share), _clip01(prod_share), _clip01(planet_share),
        _clip01(domination), _clip01(prod_margin), _clip01(fleet_share),
        _clip01(elim_share), _clip01(top_planet_prod),
        _clip01(ship_concentration), _clip01(step_frac),
        _clip01(enemy_fleet_press),
    ])
    return np.concatenate([base, _extra_features(fs, player, ships)])


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
