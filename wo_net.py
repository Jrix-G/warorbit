"""wo_net — value network for the V15++ search.

A single scalar evaluator: given a board (player-relative features), it
predicts that player's expected game result in [-1, 1]. It is NOT a policy
and never picks a move — V15's search picks moves; this net only *scores* the
leaf positions the search reaches, as a residual on V15's hand-coded eval, so
a wrong net can only nudge, never drive (no distribution shift, no collapse).

Entity-wise + deep-sets pooling: a per-planet encoder MLP feeds masked
mean/max pooling, so the net is order-invariant over planets and handles a
variable planet count via a padding mask. Small (~30k params at d=96) — a
forward is ~0.1 ms, cheap enough to score thousands of search leaves per turn.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from v17_encode import G_DIM, P_DIM


class WOValueNet(nn.Module):
    def __init__(self, d: int = 96):
        super().__init__()
        self.d = d
        self.enc = nn.Sequential(
            nn.Linear(P_DIM + G_DIM, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(),
            nn.Linear(d, 1),
        )

    def forward(self, pf: torch.Tensor, gf: torch.Tensor,
                pmask: torch.Tensor) -> torch.Tensor:
        """pf [B,N,P_DIM], gf [B,G_DIM], pmask [B,N] bool -> value [B] in [-1,1]."""
        B, N, _ = pf.shape
        x = torch.cat([pf, gf[:, None, :].expand(B, N, G_DIM)], dim=-1)
        e = self.enc(x)                                   # [B,N,d]
        m = pmask.unsqueeze(-1).to(e.dtype)
        mean = (e * m).sum(1) / m.sum(1).clamp(min=1.0)   # masked mean [B,d]
        neg = (~pmask).unsqueeze(-1).to(e.dtype) * (-1e9)
        mx = (e + neg).max(1).values                      # masked max  [B,d]
        ctx = torch.cat([mean, mx], dim=-1)               # [B,2d]
        return torch.tanh(self.head(ctx)).squeeze(-1)     # [B]


if __name__ == "__main__":
    torch.manual_seed(0)
    net = WOValueNet(d=96)
    n_params = sum(p.numel() for p in net.parameters())
    B, N = 4, 12
    pf = torch.rand(B, N, P_DIM)
    gf = torch.rand(B, G_DIM)
    pmask = torch.ones(B, N, dtype=torch.bool)
    pmask[0, 9:] = False                                  # padded planets
    v = net(pf, gf, pmask)
    assert v.shape == (B,), v.shape
    assert (v >= -1).all() and (v <= 1).all()
    # padding-invariance: padded planets must not change the value
    pf2 = pf.clone()
    pf2[0, 9:] = torch.rand(3, P_DIM)
    v2 = net(pf2, gf, pmask)
    assert torch.allclose(v[0], v2[0], atol=1e-5), (v[0], v2[0])
    print(f"wo_net: {n_params} params  forward/mask/padding-invariance OK")
