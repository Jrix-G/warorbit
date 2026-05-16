"""v17_net — entity-wise policy + value network for the V17 AlphaZero bot.

Architecture (small, ~40k params — fits 6 GB, fast to evaluate millions of
times):

  planet features [N,14] (+) global features [8]
        |  per-planet encoder MLP            -> e  [N,d]
        |  deep-sets context (mean/max pool) -> e2 [N,d]   (each planet now
        |                                                   sees the board)
        +-- POLICY: per planet i, logits over { pass } U { target j != i }.
        |   target logit(i,j) = (Wq e2_i) . (Wk e2_j)   (attention-style)
        +-- VALUE: pooled board -> tanh -> expected result for the player to
            move, in [-1,1]. Works for 2p and 4p (player-relative encoding).

The per-planet policy head is what makes the variable, combinatorial action
space tractable: a full move = one target choice per owned planet.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from v17_encode import G_DIM, P_DIM


class V17Net(nn.Module):
    def __init__(self, d: int = 64):
        super().__init__()
        self.d = d
        self.enc = nn.Sequential(
            nn.Linear(P_DIM + G_DIM, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
        )
        self.ctx = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU())
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.pass_head = nn.Linear(d, 1)
        self.value_head = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1),
        )

    def forward(self, pf: torch.Tensor, gf: torch.Tensor,
                pmask: torch.Tensor):
        """pf [B,N,14], gf [B,8], pmask [B,N] bool.

        Returns:
          policy_logits [B,N,N+1] — per planet, index 0 = pass, index k>=1 =
                                    target planet k-1; self & invalid masked,
          value         [B]       — expected result for the player to move.
        """
        B, N, _ = pf.shape
        d = self.d
        x = torch.cat([pf, gf[:, None, :].expand(B, N, G_DIM)], dim=-1)
        e = self.enc(x)                                   # [B,N,d]

        m = pmask.unsqueeze(-1).to(e.dtype)
        cnt = m.sum(1).clamp(min=1.0)
        mean = (e * m).sum(1) / cnt                       # [B,d]
        neg = (~pmask).unsqueeze(-1).to(e.dtype) * (-1e9)
        mx = (e + neg).max(1).values                      # [B,d]
        ctx = torch.cat([mean, mx], dim=-1)               # [B,2d]
        e2 = self.ctx(torch.cat(
            [e, ctx[:, None, :].expand(B, N, 2 * d)], dim=-1))   # [B,N,d]

        q = self.Wq(e2)
        k = self.Wk(e2)
        tgt = (q @ k.transpose(1, 2)) / (d ** 0.5)        # [B,N,N]
        # mask self-targeting and invalid (padded) target planets
        eye = torch.eye(N, dtype=torch.bool, device=pf.device)
        tgt = tgt.masked_fill(eye[None], -1e9)
        tgt = tgt.masked_fill(~pmask[:, None, :], -1e9)
        pas = self.pass_head(e2)                          # [B,N,1]
        policy_logits = torch.cat([pas, tgt], dim=-1)     # [B,N,N+1]

        value = torch.tanh(self.value_head(ctx)).squeeze(-1)   # [B]
        return policy_logits, value


def policy_probs(policy_logits: torch.Tensor) -> torch.Tensor:
    """Softmax over the per-planet action axis (pass + targets)."""
    return torch.softmax(policy_logits, dim=-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    net = V17Net(d=64)
    n_params = sum(p.numel() for p in net.parameters())
    B, N = 3, 12
    pf = torch.rand(B, N, P_DIM)
    gf = torch.rand(B, G_DIM)
    pmask = torch.ones(B, N, dtype=torch.bool)
    pmask[0, 9:] = False                                  # padded planets
    logits, value = net(pf, gf, pmask)
    assert logits.shape == (B, N, N + 1), logits.shape
    assert value.shape == (B,), value.shape
    probs = policy_probs(logits)
    assert torch.allclose(probs.sum(-1), torch.ones(B, N), atol=1e-5)
    # self-target probability must be ~0
    for i in range(N):
        assert probs[0, i, i + 1] < 1e-4
    # padded target columns must get ~0 probability
    assert probs[0, :, 10:].max() < 1e-4
    assert (value >= -1).all() and (value <= 1).all()
    print(f"v17_net: {n_params} params  forward/shapes/masking OK")
