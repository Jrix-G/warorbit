"""v17_encode — state <-> tensors for the V17 AlphaZero network.

The network is entity-wise: it consumes a per-planet feature matrix plus a
small global vector, and emits a per-planet policy (which target to attack)
and a value. This module is the bridge between a FastState and those tensors.

All per-planet features are PLAYER-RELATIVE (computed from the perspective of
the player to move) so a single network serves any seat.

Layout:
  planet features [N, 14]  — see PLANET_FEATS
  global features [8]      — see GLOBAL_FEATS

Decoding: the policy head gives, per owned planet, a target choice; this
module turns (source planet, target planet) into a real launch
[src_id, intercept_angle, ships].
"""

from __future__ import annotations

import math

import numpy as np

import v15_fast_sim as fsim
import v15_eval

ID, OWNER, X, Y, R, SHIPS, PROD = range(7)
F_OWNER, F_X, F_Y, F_ANGLE, F_SHIPS = 1, 2, 3, 4, 6
CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0
_LOG1000 = math.log(1000.0)
_ANGLE_TOL = 0.45            # rad — fleet-to-target bearing match

PLANET_FEATS = [
    "is_mine", "is_enemy", "is_neutral", "ships_lin", "ships_log",
    "prod", "x", "y", "radius", "rotating", "dist_sun",
    "incoming_friendly", "incoming_enemy", "ships_share_mine",
]
GLOBAL_FEATS = [
    "step_frac", "is_4p", "my_ship_share", "my_prod_share",
    "my_planet_share", "max_opp_ship_share", "my_fleet_share", "alive_frac",
]
P_DIM = len(PLANET_FEATS)
G_DIM = len(GLOBAL_FEATS)


def _fleet_incoming(fs: fsim.FastState, n_players: int) -> np.ndarray:
    """[N, n_players] — ships in flight heading toward each planet, by owner.
    A fleet is assigned to the planet whose bearing best matches its angle."""
    N = len(fs.planets)
    inc = np.zeros((N, n_players))
    if N == 0 or len(fs.fleets) == 0:
        return inc
    px = fs.planets[:, X]
    py = fs.planets[:, Y]
    for f in fs.fleets:
        o = int(f[F_OWNER])
        if not (0 <= o < n_players):
            continue
        bearings = np.arctan2(py - f[F_Y], px - f[F_X])
        diff = np.abs((bearings - f[F_ANGLE] + math.pi) % (2 * math.pi)
                      - math.pi)
        j = int(np.argmin(diff))
        if diff[j] < _ANGLE_TOL:
            inc[j, o] += f[F_SHIPS]
    return inc


def encode(fs: fsim.FastState, player: int):
    """Return (planet_feats [N,14], global_feats [8]) for `player`."""
    n = fs.n_players
    N = len(fs.planets)
    pf = np.zeros((N, P_DIM), dtype=np.float32)
    if N == 0:
        return pf, np.zeros(G_DIM, dtype=np.float32)

    owner = fs.planets[:, OWNER].astype(np.int64)
    ships = fs.planets[:, SHIPS]
    prod = fs.planets[:, PROD]
    px = fs.planets[:, X]
    py = fs.planets[:, Y]
    rad = fs.planets[:, R]

    dx = fs.p_init[:, 0] - CENTER
    dy = fs.p_init[:, 1] - CENTER
    orb_r = np.hypot(dx, dy)
    rotating = (orb_r + rad < ROTATION_RADIUS_LIMIT).astype(np.float32)
    dist_sun = np.hypot(px - CENTER, py - CENTER)

    inc = _fleet_incoming(fs, n)
    inc_friendly = inc[:, player]
    inc_enemy = inc.sum(axis=1) - inc_friendly

    garrison, fleet, prodt, planets = v15_eval.player_totals(fs)
    my_total_ships = max(garrison[player], 1e-6)

    pf[:, 0] = (owner == player).astype(np.float32)
    pf[:, 1] = ((owner >= 0) & (owner != player)).astype(np.float32)
    pf[:, 2] = (owner == -1).astype(np.float32)
    pf[:, 3] = np.clip(ships / 100.0, 0.0, 4.0)
    pf[:, 4] = np.clip(np.log1p(np.maximum(ships, 0.0)) / _LOG1000, 0.0, 1.5)
    pf[:, 5] = np.clip(prod / 5.0, 0.0, 3.0)
    pf[:, 6] = px / 100.0
    pf[:, 7] = py / 100.0
    pf[:, 8] = np.clip(rad / 5.0, 0.0, 3.0)
    pf[:, 9] = rotating
    pf[:, 10] = np.clip(dist_sun / 70.0, 0.0, 1.5)
    pf[:, 11] = np.clip(inc_friendly / 100.0, 0.0, 4.0)
    pf[:, 12] = np.clip(inc_enemy / 100.0, 0.0, 4.0)
    pf[:, 13] = np.clip(
        np.where(owner == player, ships / my_total_ships, 0.0), 0.0, 1.0)

    tot_s = max(float((garrison + fleet).sum()), 1e-6)
    tot_p = max(float(prodt.sum()), 1e-6)
    tot_pl = max(float(planets.sum()), 1e-6)
    opp = [q for q in range(n) if q != player]
    max_opp_s = max(((garrison[q] + fleet[q]) / tot_s for q in opp),
                    default=0.0)
    alive = sum(1 for q in range(n)
                if (garrison[q] + fleet[q]) > 0 or planets[q] > 0)

    gf = np.array([
        min(fs.step / 500.0, 1.0),
        1.0 if n >= 4 else 0.0,
        (garrison[player] + fleet[player]) / tot_s,
        prodt[player] / tot_p,
        planets[player] / tot_pl,
        max_opp_s,
        fleet[player] / max(garrison[player] + fleet[player], 1e-6),
        alive / n,
    ], dtype=np.float32)
    return pf, gf


def needed_ships(src_row, tgt_row) -> int:
    """Capture-sized ships for src -> tgt: defenders + production over the
    travel time + a margin."""
    dist = math.hypot(tgt_row[X] - src_row[X], tgt_row[Y] - src_row[Y])
    eta = max(1.0, dist / 4.0)
    defenders = tgt_row[SHIPS] + tgt_row[PROD] * eta
    margin = 5.0 if tgt_row[OWNER] >= 0 else 3.0
    return int(defenders + margin) + 1


def decode_move(fs: fsim.FastState, player: int,
                targets: np.ndarray) -> list:
    """targets[i] = chosen target planet INDEX for owned planet i, or -1=pass.
    Returns a launch list [[src_id, angle, ships], ...]."""
    moves = []
    planets = fs.planets
    for i in range(len(planets)):
        if int(planets[i, OWNER]) != player:
            continue
        j = int(targets[i])
        if j < 0 or j >= len(planets) or j == i:
            continue
        ang = math.atan2(planets[j, Y] - planets[i, Y],
                         planets[j, X] - planets[i, X])
        need = needed_ships(planets[i], planets[j])
        send = min(int(planets[i, SHIPS]), need)
        if send > 0:
            moves.append([int(planets[i, ID]), float(ang), send])
    return moves


def action_to_targets(fs: fsim.FastState, player: int,
                      launches) -> np.ndarray:
    """Inverse of decode_move: map a launch list back to per-owned-planet
    target indices — the policy-head label for imitation learning.

    launches: [[src_id, angle, ships], ...] (the engine action format).
    Returns targets[N]: targets[i] = target planet index for owned planet i,
    or -1 (pass / not owned / unmatched). A launch's target is the planet
    whose current bearing from the source best matches the launch angle.

    Note: launch angles aim at an interception of the target's future orbital
    position, so the bearing match is approximate; it is exact for sources
    that aim at the current position (e.g. V15) and a close proxy otherwise.
    """
    planets = fs.planets
    n = len(planets)
    targets = np.full(n, -1, dtype=np.int64)
    if n == 0 or not launches:
        return targets
    row_of = {int(planets[i, ID]): i for i in range(n)}
    px = planets[:, X]
    py = planets[:, Y]
    for launch in launches:
        if launch is None or len(launch) < 2:
            continue
        i = row_of.get(int(launch[0]))
        if i is None or int(planets[i, OWNER]) != player:
            continue
        ang = float(launch[1])
        best_j, best_d = -1, 1e18
        for j in range(n):
            if j == i:
                continue
            bearing = math.atan2(py[j] - py[i], px[j] - px[i])
            d = abs((bearing - ang + math.pi) % (2.0 * math.pi) - math.pi)
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            targets[i] = best_j
    return targets


if __name__ == "__main__":
    import random
    import v14_core
    from local_simulator.official_fast import OfficialFastGame
    for n in (2, 4):
        random.seed(5)
        np.random.seed(5)
        g = OfficialFastGame(n, seed=5, episode_steps=300, use_c_accel=False)
        for _ in range(60):
            g.step([[] for _ in range(n)])
        obs = v14_core.obs_as_dict(g.observation(0))
        fs = fsim.from_obs(obs, n_players=n)
        fs.n_players = n
        pf, gf = encode(fs, 0)
        assert pf.shape == (len(fs.planets), P_DIM), pf.shape
        assert gf.shape == (G_DIM,), gf.shape
        assert np.isfinite(pf).all() and np.isfinite(gf).all()
        # is_mine/is_enemy/is_neutral are a partition
        assert np.allclose(pf[:, 0] + pf[:, 1] + pf[:, 2], 1.0)
        mine = int(pf[:, 0].sum())
        print(f"{n}p: planets={len(fs.planets)} mine={mine} "
              f"feats ok, global={np.round(gf, 3).tolist()}")
    print("v17_encode self-check passed")
