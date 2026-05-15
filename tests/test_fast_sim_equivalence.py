"""P1 gate — v15_fast_sim must match OfficialFastGame.

For each transition of a real game, build a FastState from the observation,
apply the same actions through v15_fast_sim.step, and compare to the engine's
next observation.

Integer fields (owner, ships) must match EXACTLY. Float fields (x, y) must
match within a tight tolerance (math vs numpy trig differ by ~1 ULP).

Run:
    KMP_DUPLICATE_LIB_OK=TRUE python -u tests/test_fast_sim_equivalence.py
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import v15_fast_sim as fsim
from local_simulator.official_fast import OfficialFastGame

POS_TOL = 1e-6


def _rand_agent(obs, config=None):
    """Light random agent: launches a few fleets to exercise fleet logic."""
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    for p in planets or []:
        if int(p[1]) == player and int(p[5]) > 10:
            if random.random() < 0.4:
                angle = random.uniform(0, 2 * math.pi)
                ships = int(p[5]) // 2
                moves.append([int(p[0]), angle, ships])
    return moves


def _obs_planets(obs):
    return [list(p) for p in (obs.get("planets", []) or [])]


def _obs_fleets(obs):
    return [list(f) for f in (obs.get("fleets", []) or [])]


def _compare(eng_obs, st: fsim.FastState):
    """Return (ok, detail). ok=False on integer mismatch or large float drift."""
    eng_p = sorted(_obs_planets(eng_obs), key=lambda p: int(p[0]))
    sim_p = sorted([list(st.planets[i]) for i in range(len(st.planets))],
                   key=lambda p: int(p[0]))
    if len(eng_p) != len(sim_p):
        return False, f"planet count {len(eng_p)} vs {len(sim_p)}"
    max_pos = 0.0
    for ep, sp in zip(eng_p, sim_p):
        if int(ep[0]) != int(sp[0]):
            return False, f"planet id {ep[0]} vs {sp[0]}"
        if int(ep[1]) != int(sp[1]):
            return False, f"planet {ep[0]} owner {ep[1]} vs {int(sp[1])}"
        if int(ep[5]) != int(sp[5]):
            return False, f"planet {ep[0]} ships {ep[5]} vs {int(sp[5])}"
        max_pos = max(max_pos, abs(ep[2] - sp[2]), abs(ep[3] - sp[3]))

    eng_f = sorted(_obs_fleets(eng_obs), key=lambda f: int(f[0]))
    sim_f = sorted([list(st.fleets[i]) for i in range(len(st.fleets))],
                   key=lambda f: int(f[0]))
    if len(eng_f) != len(sim_f):
        return False, f"fleet count {len(eng_f)} vs {len(sim_f)}"
    for ef, sf in zip(eng_f, sim_f):
        if int(ef[0]) != int(sf[0]):
            return False, f"fleet id {ef[0]} vs {sf[0]}"
        if int(ef[1]) != int(sf[1]) or int(ef[6]) != int(sf[6]):
            return False, f"fleet {ef[0]} owner/ships mismatch"
        max_pos = max(max_pos, abs(ef[2] - sf[2]), abs(ef[3] - sf[3]))

    if max_pos > POS_TOL:
        return False, f"position drift {max_pos:.2e} > {POS_TOL:.0e}"
    return True, max_pos


def run(n_games: int, n_players: int, max_steps: int) -> tuple[int, int, float]:
    """Returns (transitions_checked, failures, worst_pos_drift)."""
    checked = 0
    failures = 0
    worst = 0.0
    for game_i in range(n_games):
        seed = 4000 + game_i
        random.seed(seed)
        np.random.seed(seed)
        game = OfficialFastGame(n_players, seed=seed,
                                episode_steps=max_steps, use_c_accel=False)
        while not game.done:
            cur_step = int(game.observation(0).get("step", 0) or 0)
            # Stay away from comet spawn boundaries (fast_sim does not spawn).
            if any((cur_step + 1) == sp for sp in fsim.COMET_SPAWN_STEPS):
                break
            obs_before = game.observation(0)
            st = fsim.from_obs(obs_before, n_players=n_players,
                               episode_steps=max_steps, ship_speed=6.0)
            actions = [_rand_agent(game.observation(p)) for p in range(n_players)]
            game.step(actions)
            nxt = fsim.step(st, actions)
            ok, detail = _compare(game.observation(0), nxt)
            checked += 1
            if not ok:
                failures += 1
                if failures <= 5:
                    print(f"  FAIL game={game_i} step={cur_step}: {detail}")
            else:
                worst = max(worst, detail)
    return checked, failures, worst


def main() -> int:
    total_checked = 0
    total_fail = 0
    worst = 0.0
    for mode, n_players, n_games, max_steps in [
        ("2p", 2, 60, 160),
        ("4p", 4, 40, 160),
    ]:
        c, f, w = run(n_games, n_players, max_steps)
        worst = max(worst, w)
        total_checked += c
        total_fail += f
        print(f"[{mode}] transitions={c} failures={f} worst_pos_drift={w:.2e}")

    print(f"\nTotal: {total_checked - total_fail}/{total_checked} transitions match "
          f"(worst float drift {worst:.2e}).")
    if total_fail:
        print("P1 GATE FAILED — v15_fast_sim diverges from the official engine.")
        return 1
    print("P1 GATE PASSED — v15_fast_sim is equivalent to the official engine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
