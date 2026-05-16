"""v16_search — GPU-native RCC search using the v16 evaluator (MLP + SCR).

Same combination search as v15_gpu_search (enumerate atomic shots, stage-1
prune, stage-2 subset search, pick), but the leaf scoring is the
Standing-Conditioned-Risk score of v16_eval instead of the linear ESC:

  * one MLP value head + one spread head per leaf,
  * the combo is chosen by  value + z_gain*(0.5 - root_value)*spread,
  * root_value is our win-prob estimate at the current position.

Stateless geometry/enumeration helpers are reused from v15_gpu_search; only
the evaluation differs, so the search behaviour is otherwise identical and
benefits from the same torch.compile'd engine.
"""

from __future__ import annotations

import torch

import v15_eval
import v15_gpu_search as g15
import v15_gpu_sim as gsim
import v16_eval

A_SLOTS = g15.A_SLOTS
T_TOP = g15.T_TOP


def _rollout_leaves(states, first_actions, horizon, player):
    """Apply first_actions, run horizon-1 passive steps, return leaf features
    [G, 11] for `player` (NOT a score — scoring is done by the v16 evaluator)."""
    G = states.B
    P = states.n_players
    st = gsim.step(states, first_actions, has_launch=True)
    empty = torch.zeros((G, P, A_SLOTS, 3), dtype=states.planets.dtype,
                        device=states.device)
    for _ in range(horizon - 1):
        st = gsim.step_passive(st, empty)
    return g15.batch_features(st, player)


def gpu_search_v16(batch: gsim.GpuBatch, player: int, theta, hidden: int, *,
                   horizon: int = 16, explore: float = 0.0) -> torch.Tensor:
    """Best move for `player` across all B games, using the v16 evaluator.

    theta  — flat v16 parameter vector (MLP weights + z_gain),
    hidden — MLP hidden width.
    Returns [B, A_SLOTS, 3]."""
    B = batch.B
    P = batch.n_players
    dev = batch.device
    dt = batch.planets.dtype
    esc_w = torch.as_tensor(
        v15_eval.ESC.w4p if P >= 4 else v15_eval.ESC.w2p,
        dtype=dt, device=dev)
    p = v16_eval.unpack(theta, hidden, dev, dt)

    # our win-prob estimate at the current position (drives the SCR tilt)
    root_feats = g15.batch_features(batch, player)             # [B,11]
    root_value, _ = v16_eval.value_and_spread(root_feats, esc_w, p)  # [B]

    def _score(leaf_feats, factor):
        rv = root_value.repeat_interleave(factor)              # [B*factor]
        return v16_eval.scr_score(leaf_feats, rv, esc_w, p)

    shots, valid = g15.enumerate_shots(batch, player)          # [B,K,3],[B,K]
    K = shots.shape[1]

    # --- baseline: do nothing ---
    empty1 = torch.zeros((B, P, A_SLOTS, 3), dtype=dt, device=dev)
    base_leaf = _rollout_leaves(batch, empty1, horizon, player)
    baseline = _score(base_leaf, 1)                            # [B]

    # --- stage 1: each atomic shot alone ---
    big = g15._expand(batch, K)
    single = torch.zeros((B, K, A_SLOTS, 3), dtype=dt, device=dev)
    single[:, :, 0, :] = shots
    single = single.reshape(B * K, A_SLOTS, 3)
    fa = g15._combo_actions(single, player, P)
    leaves1 = _rollout_leaves(big, fa, horizon, player)
    s1 = _score(leaves1, K).reshape(B, K)
    s1 = torch.where(valid, s1, torch.full_like(s1, -1e9))

    # --- top-T survivors ---
    t = min(T_TOP, K)
    top_val, top_idx = torch.topk(s1, k=t, dim=1)
    top_shots = torch.gather(shots, 1, top_idx.unsqueeze(-1).expand(-1, -1, 3))
    top_valid = top_val > -1e8

    # --- stage 2: every subset of the survivors ---
    pats = g15._PATTERNS.to(dev)
    S = pats.shape[0]
    gidx = pats.clamp(min=0)
    combos = top_shots[:, gidx, :]                             # [B,S,A,3]
    slot_on = (pats >= 0)
    surv_ok = torch.gather(
        top_valid, 1, gidx.reshape(-1).unsqueeze(0).expand(B, -1)
    ).reshape(B, S, A_SLOTS)
    active = slot_on.unsqueeze(0) & surv_ok
    combos = combos * active.unsqueeze(-1)
    combo_valid = active.any(dim=2)

    big2 = g15._expand(batch, S)
    fa2 = g15._combo_actions(combos.reshape(B * S, A_SLOTS, 3), player, P)
    leaves2 = _rollout_leaves(big2, fa2, horizon, player)
    s2 = _score(leaves2, S).reshape(B, S)
    s2 = torch.where(combo_valid, s2, torch.full_like(s2, -1e9))

    # --- pick best combo (vs do-nothing baseline), with optional exploration ---
    best_val, best_pat = torch.max(s2, dim=1)
    bidx = torch.arange(B, device=dev)
    explored = torch.zeros(B, dtype=torch.bool, device=dev)
    if explore > 0.0:
        explored = torch.rand(B, device=dev) < explore
        rand_pat = torch.randint(0, S, (B,), device=dev)
        best_pat = torch.where(explored, rand_pat, best_pat)
    chosen = combos[bidx, best_pat]
    take = (explored | (best_val > baseline)).view(B, 1, 1)
    return torch.where(take, chosen, torch.zeros_like(chosen))
