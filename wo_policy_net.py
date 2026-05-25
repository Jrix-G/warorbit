"""wo_policy_net — policy network for V15++ candidate generation.

Predicts, for each owned planet, a distribution over target planets (including
pass). Trained by imitation on top-player actions. Used NOT to evaluate leaves
(that's ESC's job) but to SUGGEST which (source→target) candidates to add to
V15's combo search pool.

Architecture mirrors WOValueNet: per-planet MLP encoder with deep-sets pooling,
then a per-owned-planet head that scores each possible target.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import v17_encode as enc


class WOPolicyNet(nn.Module):
    """(PF[B,N,P_DIM], GF[B,G_DIM], MASK[B,N]) -> logits[B,N,N+1].

    logits[b, i, 0] = score for planet i to pass
    logits[b, i, j+1] = score for planet i to target planet j
    Non-owned planets / padding positions are masked out by the caller.
    """

    def __init__(self, d: int = 96):
        super().__init__()
        in_dim = enc.P_DIM + enc.G_DIM
        self.planet_enc = nn.Sequential(
            nn.Linear(in_dim, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
        )
        # deep-sets global context
        self.global_head = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(),
        )
        # per-planet policy head: context + own encoding → scores over N+1 actions
        self.policy_head = nn.Sequential(
            nn.Linear(d + d, d), nn.ReLU(),
            nn.Linear(d, d),
        )
        # final per-target scorer: dot between policy embedding and target embedding
        self.d = d

    def forward(self, pf: torch.Tensor, gf: torch.Tensor,
                pmask: torch.Tensor) -> torch.Tensor:
        """
        pf:    [B, N, P_DIM]
        gf:    [B, G_DIM]
        pmask: [B, N]  bool
        returns logits [B, N, N+1]
        """
        B, N, _ = pf.shape
        gf_exp = gf.unsqueeze(1).expand(-1, N, -1)        # [B,N,G_DIM]
        x = torch.cat([pf, gf_exp], dim=-1)               # [B,N,P_DIM+G_DIM]
        enc = self.planet_enc(x)                           # [B,N,d]

        # masked mean+max pooling for global context
        mask_f = pmask.float().unsqueeze(-1)               # [B,N,1]
        enc_masked = enc * mask_f
        n_valid = mask_f.sum(dim=1).clamp(min=1)
        mean_pool = enc_masked.sum(dim=1) / n_valid        # [B,d]
        big = -1e9 * (1 - mask_f)
        max_pool = (enc + big).max(dim=1).values           # [B,d]
        global_ctx = self.global_head(
            torch.cat([mean_pool, max_pool], dim=-1))      # [B,d]

        # per-planet policy embedding
        ctx_exp = global_ctx.unsqueeze(1).expand(-1, N, -1)  # [B,N,d]
        pol_emb = self.policy_head(
            torch.cat([enc, ctx_exp], dim=-1))             # [B,N,d]

        # logits: for each planet i, score each target j using dot product
        # pass (j=-1) gets its own score via pol_emb . enc[i] (self-attention)
        # target j: pol_emb[i] . enc[j]
        target_enc = enc                                    # [B,N,d]  (all as targets)
        logits_targets = torch.bmm(pol_emb, target_enc.transpose(1, 2))  # [B,N,N]
        # pass logit = norm of policy embedding (proxy for "confidence to act")
        logits_pass = (pol_emb * enc).sum(dim=-1, keepdim=True)          # [B,N,1]
        logits = torch.cat([logits_pass, logits_targets], dim=-1)        # [B,N,N+1]
        return logits

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
