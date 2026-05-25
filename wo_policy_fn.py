"""wo_policy_fn — load trained policy net as a callable for v15_search.

Returns policy_fn(fs, player) -> list of (src_planet_idx, tgt_planet_idx)
sorted by confidence (highest first), up to top_k suggestions.
These become additional atomic candidates in V15's combo search.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np

import v17_encode as enc
from wo_policy_net import WOPolicyNet


def load_policy_fn(ckpt: str = "analysis/wo_policy.pt",
                   device: str = "cpu",
                   top_k: int = 8):
    """Return policy_fn(fs, player) -> [(src_idx, tgt_idx), ...] top_k pairs."""
    c = torch.load(ckpt, map_location=device, weights_only=False)
    net = WOPolicyNet(d=int(c["d"]))
    net.load_state_dict(c["state_dict"])
    net.eval().to(device)
    torch.set_num_threads(1)

    def policy_fn(fs, player: int):
        pf, gf = enc.encode(fs, player)
        n = pf.shape[0]
        if n == 0:
            return []

        with torch.no_grad():
            logits = net(
                torch.as_tensor(pf[None], dtype=torch.float32, device=device),
                torch.as_tensor(gf[None], dtype=torch.float32, device=device),
                torch.ones(1, n, dtype=torch.bool, device=device),
            )  # [1, N, N+1]

        logits = logits[0].cpu().numpy()  # [N, N+1]

        # identify owned planets
        from v17_encode import OWNER
        owned = [i for i in range(n) if int(fs.planets[i, OWNER]) == player]

        candidates = []
        for src_enc_idx in owned:
            # logits[src_enc_idx, 1:] are scores for targets 0..N-1
            scores = logits[src_enc_idx, 1:n + 1]  # [N]
            # exclude self-targeting and non-planets
            for tgt_enc_idx in range(n):
                if tgt_enc_idx == src_enc_idx:
                    continue
                candidates.append((scores[tgt_enc_idx], src_enc_idx, tgt_enc_idx))

        candidates.sort(key=lambda x: -x[0])
        return [(s, t) for _, s, t in candidates[:top_k]]

    return policy_fn
