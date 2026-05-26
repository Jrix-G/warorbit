"""Bench wrapper: play moves with a behavior-cloned candidate-policy net.

Loads a checkpoint trained by neural_network_gpu/scripts/train_imitation_4p.py
and exposes ``agent(obs, config)`` compatible with v20_bench (candidate-style
obs).  Encoder scales match the imitation_4p_top10_v1 dataset so inference
features live in the same numeric space the net was trained on.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

_STAGE = Path(__file__).resolve().parent / "neural_network_gpu" / "kaggle_submission_stage"
import sys

if str(_STAGE) not in sys.path:
    sys.path.insert(0, str(_STAGE))

from neural_network.src.model import ModelConfig, NeuralNetworkModel, load_compatible_state_dict  # noqa: E402
from neural_network.src.notebook_4p_training import _make_policy_agent  # noqa: E402


# Matches imitation_4p_top10_v1/report.json encoder_config.
_CONFIG = {
    "allow_support_actions": True,
    "board_scale": 100.0,
    "game_engine": "official_fast",
    "horizon_scale": 100.0,
    "max_actions_per_turn": 4,
    "max_fleets": 128,
    "max_planets": 64,
    "max_players": 4,
    "max_turns": 250,
    "min_expand_attack_ships": 6,
    "official_fast_c_accel": False,
    "planet_id_scale": 100.0,
    "policy_prior_strength": float(os.environ.get("BC_PRIOR_STRENGTH", "0.55")),
    "production_scale": 10.0,
    "radius_scale": 10.0,
    "seed": 42,
    "send_ratios": [0.25, 0.35, 0.5, 0.65, 0.8, 0.95],
    "ship_scale": 2000.0,
    "simple_2p_only": False,
}


@lru_cache(maxsize=1)
def _agent():
    ckpt = os.environ.get(
        "BC_CKPT",
        str(Path(__file__).resolve().parent / "runs" / "imitation_4p_top10_v1" / "bc_4p_top10_best.npz"),
    )
    z = np.load(ckpt, allow_pickle=True)
    state = {k: z[k] for k in z.files}
    w = state["input_proj.0.weight"]
    hidden_dim, input_dim = int(np.asarray(w).shape[0]), int(np.asarray(w).shape[1])
    model = NeuralNetworkModel(ModelConfig(input_dim=input_dim, hidden_dim=hidden_dim))
    load_compatible_state_dict(model, state)
    model.eval()
    return _make_policy_agent(model, _CONFIG, temperature=0.0, explore=False)


def agent(obs, config=None):
    return _agent()(obs, config)
