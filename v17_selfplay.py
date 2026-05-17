"""v17_selfplay — generate AlphaZero training data by MCTS self-play.

Every player in a game is the current net + MCTS. For each (state, player)
we record: the encoded state, the MCTS visit-marginal policy, and (at game
end) the player's result. The net is then trained toward the MCTS policy and
the outcomes — this is the policy-improvement ratchet.

CPU multiprocessing: the net is tiny, runs fast on CPU; each worker plays
games independently with its own net copy.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

import v14_core
import v15_fast_sim as fsim
import v15_search
from local_simulator.official_fast import OfficialFastGame
import v17_encode as enc
from v17_mcts import mcts_move
from v17_net import V17Net

N_MAX = 48
EPISODE = 260
TEMP_MOVES = 30          # temperature 1 for the opening, then greedy


def _pad(pf, pol):
    """Pad a sample to N_MAX planets."""
    n = pf.shape[0]
    k = min(n, N_MAX)
    pfp = np.zeros((N_MAX, enc.P_DIM), dtype=np.float32)
    polp = np.zeros((N_MAX, N_MAX + 1), dtype=np.float32)
    mask = np.zeros(N_MAX, dtype=bool)
    pfp[:k] = pf[:k]
    mask[:k] = True
    # policy: index 0 = pass, 1.. = target planet; clip targets to N_MAX
    polp[:k, 0] = pol[:k, 0]
    tk = min(pol.shape[1] - 1, N_MAX)
    polp[:k, 1:1 + tk] = pol[:k, 1:1 + tk]
    return pfp, polp, mask


def play_game(args):
    """Play one self-play game; return a list of training samples.

    If vs_v15_frac > 0, some opponents are replaced by V15 (RCC).
    Only V17's (state, policy, outcome) samples are collected.
    """
    state_dict, d, n_players, seed, n_sims, vs_v15_frac = args
    net = V17Net(d=d)
    net.load_state_dict(state_dict)
    net.eval()
    rng = np.random.default_rng(seed)

    # decide which players use V15 as opponent
    v17_players = list(range(n_players))
    v15_players = []
    if vs_v15_frac > 0 and n_players >= 2:
        # always keep at least one V17 player; randomise opponents
        for p in range(1, n_players):
            if rng.random() < vs_v15_frac:
                v15_players.append(p)
                v17_players.remove(p)

    g = OfficialFastGame(n_players, seed=seed, episode_steps=EPISODE,
                         use_c_accel=False)
    obs0 = v14_core.obs_as_dict(g.observation(0))
    fs = fsim.from_obs(obs0, n_players=n_players, episode_steps=EPISODE)
    fs.n_players = n_players

    raw = []
    t = 0
    while not fs.done:
        temp = 1.0 if t < TEMP_MOVES else 0.0
        moves = []
        step_samples = []
        for p in range(n_players):
            if p in v15_players:
                o = v15_search.state_to_obs(fs, p)
                action = v15_search.search(o, None)
                moves.append(action if isinstance(action, list) else [])
            else:
                pf, gf = enc.encode(fs, p)
                action, pol = mcts_move(net, fs, p, n_sims=n_sims, rng=rng,
                                        temperature=temp)
                moves.append(action)
                pfp, polp, mask = _pad(pf, pol)
                step_samples.append([pfp, gf, polp, mask, p])
        fs = fsim.step(fs, moves)
        raw.extend(step_samples)
        t += 1

    sc = fsim.scores(fs)
    best = max(sc) if sc else 0
    winners = [p for p in range(n_players) if sc[p] == best and best > 0]
    val = {}
    for p in range(n_players):
        val[p] = (1.0 if (len(winners) == 1 and winners[0] == p)
                  else (0.0 if p in winners else -1.0))
    return [(pf, gf, pol, mask, val[p]) for (pf, gf, pol, mask, p) in raw]
