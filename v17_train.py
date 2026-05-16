"""v17_train — train the V17 network on MCTS self-play data.

Two losses, the AlphaZero pair:
  * policy — cross-entropy of the net's per-planet policy toward the MCTS
    visit-marginal (the MCTS is a stronger policy; the net distills it),
  * value — MSE of the net's value toward the game outcome.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def train_net(net, PF, GF, POL, MASK, VAL, *, epochs, lr, device, bs=256):
    """Train `net` in place on a dataset; return per-epoch stats."""
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    pf = torch.as_tensor(PF, device=device)
    gf = torch.as_tensor(GF, device=device)
    pol = torch.as_tensor(POL, device=device)
    mask = torch.as_tensor(MASK, device=device)
    val = torch.as_tensor(VAL, device=device)
    owned = pol.sum(dim=-1) > 0.5                 # [B,N] — owned-planet rows
    n = len(PF)
    idx = np.arange(n)
    stats = []
    for ep in range(epochs):
        np.random.shuffle(idx)
        tp = tv = 0.0
        for s in range(0, n, bs):
            b = torch.as_tensor(idx[s:s + bs], device=device)
            logits, value = net(pf[b], gf[b], mask[b])
            logp = F.log_softmax(logits, dim=-1)
            ce = -(pol[b] * logp).sum(dim=-1)     # [B,N] per-planet CE
            ow = owned[b]
            ploss = (ce * ow).sum() / ow.sum().clamp(min=1.0)
            vloss = F.mse_loss(value, val[b])
            loss = ploss + vloss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tp += ploss.item() * len(b)
            tv += vloss.item() * len(b)
        stats.append((tp / n, tv / n))
    net.eval()
    return stats
