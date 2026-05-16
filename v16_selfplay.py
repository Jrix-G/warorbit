"""v16_selfplay — batched GPU games driven by the v16 (MLP + SCR) search.

Plays B Orbit Wars games at once on v15_gpu_sim, every player driven by
v16_search.gpu_search_v16 with its own theta. Used by the ES loop to score a
candidate theta: candidate in seat 0, league opponents in the other seats,
final scores -> win-rate.

The torch.compile'd engine step and the initial-map drawing are reused from
v15_gpu_selfplay.
"""

from __future__ import annotations

import torch

import v15_gpu_selfplay as sp15
import v15_gpu_sim as gsim
import v16_search

initial_states = sp15.initial_states          # re-export (map drawing)


def play_batch_v16(states, theta_by_player, hidden, *, horizon=16,
                   device="cuda", max_steps=500, explore=0.0):
    """Play one batch of games to the end with the v16 search.

    theta_by_player[p] — flat v16 parameter vector for player p.
    Returns scores [B, n_players] (numpy)."""
    sp15._ensure_compiled()
    batch = gsim.from_faststates(states, device=device, m_max=sp15.M_MAX,
                                 dtype=torch.float32, n_fixed=sp15.N_FIXED)
    P = batch.n_players
    for _ in range(max_steps):
        if bool(batch.done.all()):
            break
        moves = [v16_search.gpu_search_v16(batch, p, theta_by_player[p],
                                           hidden, horizon=horizon,
                                           explore=explore)
                 for p in range(P)]
        actions = torch.stack(moves, dim=1)            # [B,P,A,3]
        batch = gsim.step(batch, actions)
    return gsim.scores(batch).cpu().numpy()            # [B,P]
