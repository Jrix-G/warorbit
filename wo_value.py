"""wo_value — load the trained V15++ value net as a scoring callable.

Exposes load_value_fn() -> value_fn(fs, player) -> float in [-1,1]. The V15
search calls value_fn to score leaf positions (see v15_search._eval_combo).
The net is loaded once and reused; single-threaded (the search is already
parallelised across games).
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

import v17_encode as enc
from wo_net import WOValueNet


def load_value_fn(ckpt: str = "analysis/wo_value.pt", device: str = "cpu"):
    """Return a callable value_fn(fs, player) -> float.

    If the checkpoint was trained with --residual, value_fn returns the
    predicted residual (outcome01 - ESC) in [-1,1] and carries
    value_fn.is_residual = True.  The search then uses ESC + λ*residual
    instead of (1-λ)*ESC + λ*net01.
    """
    c = torch.load(ckpt, map_location=device, weights_only=False)
    net = WOValueNet(d=int(c["d"]))
    net.load_state_dict(c["state_dict"])
    net.eval().to(device)
    torch.set_num_threads(1)
    is_residual = bool(c.get("residual", False))

    def value_fn(fs, player: int) -> float:
        pf, gf = enc.encode(fs, player)
        n = pf.shape[0]
        if n == 0:
            return 0.0
        with torch.no_grad():
            v = net(torch.as_tensor(pf[None], device=device),
                    torch.as_tensor(gf[None], device=device),
                    torch.ones(1, n, dtype=torch.bool, device=device))
        return float(v[0])

    value_fn.is_residual = is_residual
    return value_fn


if __name__ == "__main__":
    # smoke: load the net and score a mid-game state.
    import v14_core
    import v15_fast_sim as fsim
    from local_simulator.official_fast import OfficialFastGame

    fn = load_value_fn()
    g = OfficialFastGame(2, seed=1, episode_steps=250, use_c_accel=False)
    for _ in range(40):
        g.step([[], []])
    fs = fsim.from_obs(v14_core.obs_as_dict(g.observation(0)), n_players=2,
                       episode_steps=250)
    fs.n_players = 2
    v0 = fn(fs, 0)
    v1 = fn(fs, 1)
    assert -1.0 <= v0 <= 1.0 and -1.0 <= v1 <= 1.0, (v0, v1)
    print(f"wo_value: value_fn OK  p0={v0:+.3f}  p1={v1:+.3f}")
