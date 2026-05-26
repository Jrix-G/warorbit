"""Parity: v15_eval.features (CPU/numpy) == v15_gpu_search.batch_features (torch).

The ES loop scores leaves on the GPU path; the deploy/v15 path scores on the
CPU path. If they diverge, the agent trained in the ES run is not the agent that
plays. This locks the 24-feature V18 basis across both implementations.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import v15_eval
import v15_gpu_search as g15
import v15_gpu_sim as gsim
from v15_gpu_selfplay import initial_states


def _check(n_players, n_games, seed, warm):
    states = initial_states(n_players, n_games, seed, warm_steps=warm)
    batch = gsim.from_faststates(states, device="cpu", dtype=torch.float64)
    worst = 0.0
    for p in range(n_players):
        gpu = g15.batch_features(batch, p).numpy()          # [B, 24]
        for b in range(n_games):
            cpu = v15_eval.features(states[b], p)            # [24]
            d = float(np.abs(cpu - gpu[b]).max())
            if d > worst:
                worst = d
            assert d < 1e-4, (
                f"mismatch n_players={n_players} game={b} player={p} "
                f"maxdiff={d}\ncpu={np.round(cpu,4)}\ngpu={np.round(gpu[b],4)}"
            )
    return worst


def main():
    assert v15_eval.N_FEATURES == 24, v15_eval.N_FEATURES
    cases = [(2, 16, 1234, 0), (2, 16, 99, 25), (4, 16, 7, 0), (4, 16, 555, 40)]
    for n_players, n_games, seed, warm in cases:
        w = _check(n_players, n_games, seed, warm)
        print(f"OK {n_players}p games={n_games} warm={warm} maxdiff={w:.2e}")
    print("PARITY PASS — 24 features match CPU<->GPU")


if __name__ == "__main__":
    main()
