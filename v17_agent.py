"""v17_agent — play the V17 network (net + MCTS) as an agent.

Loads a trained checkpoint and answers obs -> move via MCTS. Used both for
benchmarking (Phase-0 gate, iteration gates) and, after packaging, as the
Kaggle submission entry point. n_sims is tuned so a move stays well inside
the ~1 s budget.
"""

from __future__ import annotations

import numpy as np
import torch

import v15_fast_sim as fsim
import v15_search as _rcc          # only for _infer_n_players
from v17_mcts import mcts_move
from v17_net import V17Net

_NET = None
_NET_PATH = None


def load_net(path: str):
    """Load (and cache) a V17 checkpoint."""
    global _NET, _NET_PATH
    if _NET is None or _NET_PATH != path:
        c = torch.load(path, map_location="cpu")
        net = V17Net(d=c["d"])
        net.load_state_dict(c["state_dict"])
        net.eval()
        _NET, _NET_PATH = net, path
    return _NET


def make_agent(checkpoint: str, n_sims: int = 120):
    """Return an agent(obs, config) closure backed by `checkpoint`."""
    net = load_net(checkpoint)
    rng = np.random.default_rng(0)

    def agent(obs, config=None):
        try:
            if isinstance(obs, dict):
                player = int(obs.get("player", 0) or 0)
            else:
                player = int(getattr(obs, "player", 0) or 0)
            fs = fsim.from_obs(obs, n_players=2)
            fs.n_players = _rcc._infer_n_players(fs.planets)
            action, _ = mcts_move(net, fs, player, n_sims=n_sims,
                                  device="cpu", rng=rng, temperature=0.0)
            return action if isinstance(action, list) else []
        except Exception:
            return []

    return agent
