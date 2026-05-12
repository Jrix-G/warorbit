from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from neural_network.src.model import ModelConfig, NeuralNetworkModel, load_compatible_state_dict
from neural_network.src.notebook_4p_training import _make_policy_agent
from neural_network.src.storage import load_checkpoint
import neural_network.src.storage as storage_module


ROOT = Path(storage_module.__file__).resolve().parents[2]
CHECKPOINT_PATH = ROOT / "best_validated.npz"
CONFIG_PATH = ROOT / "config.json"
_CHECKPOINT_STATE = None
_CHECKPOINT_META = None


def _load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}
    config.update(
        {
            "allow_support_actions": False,
            "board_scale": 100.0,
            "game_engine": "official_fast",
            "hidden_dim": 320,
            "horizon_scale": 100.0,
            "max_actions_per_turn": 4,
            "max_fleets": 128,
            "max_planets": 64,
            "max_players": 4,
            "max_turns": 100,
            "min_expand_attack_ships": 6,
            "official_fast_c_accel": True,
            "planet_id_scale": 64.0,
            "policy_prior_strength": 0.55,
            "production_scale": 10.0,
            "radius_scale": 10.0,
            "seed": 42,
            "send_ratios": [0.25, 0.35, 0.5, 0.65, 0.8, 0.95],
            "ship_scale": 2000.0,
            "simple_2p_only": True,
        }
    )
    return config


@lru_cache(maxsize=1)
def _build_agent():
    config = _load_config()
    state, _metadata = load_checkpoint(CHECKPOINT_PATH)
    input_weight = state["input_proj.0.weight"]
    hidden_dim, input_dim = int(input_weight.shape[0]), int(input_weight.shape[1])
    model = NeuralNetworkModel(ModelConfig(input_dim=input_dim, hidden_dim=hidden_dim))
    load_compatible_state_dict(model, state)
    model.eval()
    return _make_policy_agent(model, config, temperature=0.0, explore=False)


def agent(obs, config=None):
    return _build_agent()(obs, config)
