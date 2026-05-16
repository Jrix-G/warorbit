"""Validate v15_gpu_sim (batched torch) against v15_fast_sim (numpy).

Both engines step the SAME comet-free states with the SAME actions; the
resulting planets / fleets / step / done must match. Fleets are compared as
sets keyed by fleet id (the GPU engine uses fixed slots, so fleet ordering
differs from the numpy engine's append order).

Run:
    python tests/test_gpu_sim_equivalence.py
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import v14_core
import v15_fast_sim as fsim
import v15_gpu_sim as gsim
from local_simulator.official_fast import OfficialFastGame

ID, OWNER, X, Y, R, SHIPS, PROD = range(7)
F_ID = 0
A_MAX = 4
TOL = 1e-6


def _rand_action(fs: fsim.FastState, player: int, rng: random.Random):
    """Up to A_MAX launches from the player's planets (random angle/ships)."""
    owned = [int(p[ID]) for p in fs.planets if int(p[OWNER]) == player]
    rng.shuffle(owned)
    acts = []
    for pid in owned[:A_MAX]:
        row = next(p for p in fs.planets if int(p[ID]) == pid)
        garrison = int(row[SHIPS])
        if garrison < 2 or rng.random() < 0.4:
            continue
        ships = rng.randint(1, garrison)
        angle = rng.uniform(-3.14159, 3.14159)
        acts.append([pid, angle, ships])
    return acts


def _make_states(n_players, n_states, rng):
    """Comet-free mid-game states: step random self-play maps on v15_fast_sim."""
    states, actions = [], []
    for k in range(n_states):
        seed = 9000 + k
        g = OfficialFastGame(n_players, seed=seed, episode_steps=500,
                             use_c_accel=False)
        obs = v14_core.obs_as_dict(g.observation(0))
        fs = fsim.from_obs(obs, n_players=n_players)
        fs.n_players = n_players
        # advance a random number of turns with random actions
        for _ in range(rng.randint(3, 60)):
            acts = [_rand_action(fs, p, rng) for p in range(n_players)]
            fs = fsim.step(fs, acts)
            if fs.done:
                break
        states.append(fs)
        actions.append([_rand_action(fs, p, rng) for p in range(n_players)])
    return states, actions


def _actions_tensor(actions, n_players):
    B = len(actions)
    t = torch.zeros((B, n_players, A_MAX, 3), dtype=torch.float64)
    for b, per_player in enumerate(actions):
        for p, acts in enumerate(per_player):
            for a, mv in enumerate(acts[:A_MAX]):
                t[b, p, a, :] = torch.tensor(mv, dtype=torch.float64)
    return t


def _fleet_dict(fs):
    return {int(f[F_ID]): np.asarray(f, dtype=np.float64) for f in fs.fleets}


def _compare(fs_cpu, fs_gpu, b):
    # planets — sorted by id
    pc = sorted(fs_cpu.planets.tolist(), key=lambda r: r[ID])
    pg = sorted(fs_gpu.planets.tolist(), key=lambda r: r[ID])
    if len(pc) != len(pg):
        return f"game {b}: planet count {len(pc)} vs {len(pg)}"
    for rc, rg in zip(pc, pg):
        if max(abs(a - bb) for a, bb in zip(rc, rg)) > TOL:
            return f"game {b}: planet mismatch\n  cpu={rc}\n  gpu={rg}"
    # fleets — keyed by id
    fc, fg = _fleet_dict(fs_cpu), _fleet_dict(fs_gpu)
    if set(fc) != set(fg):
        return (f"game {b}: fleet ids differ "
                f"cpu={sorted(fc)} gpu={sorted(fg)}")
    for fid in fc:
        if np.max(np.abs(fc[fid] - fg[fid])) > TOL:
            return (f"game {b}: fleet {fid} mismatch\n"
                    f"  cpu={fc[fid]}\n  gpu={fg[fid]}")
    if fs_cpu.step != fs_gpu.step:
        return f"game {b}: step {fs_cpu.step} vs {fs_gpu.step}"
    if bool(fs_cpu.done) != bool(fs_gpu.done):
        return f"game {b}: done {fs_cpu.done} vs {fs_gpu.done}"
    return None


def run(n_players, n_states, rng):
    states, actions = _make_states(n_players, n_states, rng)
    # CPU reference
    cpu_next = [fsim.step(s, a) for s, a in zip(states, actions)]
    # GPU batched
    batch = gsim.from_faststates(states, device="cpu", dtype=torch.float64)
    at = _actions_tensor(actions, n_players)
    gpu_batch = gsim.step(batch, at)
    gpu_next = [gsim.to_faststate(gpu_batch, b) for b in range(len(states))]

    fails = 0
    for b in range(len(states)):
        err = _compare(cpu_next[b], gpu_next[b], b)
        if err:
            fails += 1
            if fails <= 3:
                print("  FAIL " + err)
    ok = len(states) - fails
    print(f"  {n_players}p: {ok}/{len(states)} transitions match")
    return fails


def main():
    rng = random.Random(0)
    print("v15_gpu_sim vs v15_fast_sim — single-step equivalence")
    total = 0
    for n_players in (2, 4):
        total += run(n_players, 60, rng)
    if total == 0:
        print("ALL TRANSITIONS MATCH — GPU engine is bit-exact")
    else:
        print(f"{total} MISMATCHES — GPU engine NOT equivalent")
        sys.exit(1)


if __name__ == "__main__":
    main()
