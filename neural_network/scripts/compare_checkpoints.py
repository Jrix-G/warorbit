from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_network.src.utils import load_json
from neural_network.src.model import NeuralNetworkModel, ModelConfig, load_compatible_state_dict
from neural_network.src.storage import load_checkpoint
from neural_network.src.benchmark import compare_checkpoints
from neural_network.src.trainer import _infer_input_dim


def _load_model(path: str, cfg: dict) -> NeuralNetworkModel:
    model = NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(cfg)))
    state, _ = load_checkpoint(path)
    load_compatible_state_dict(model, state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--games", type=int, default=None)
    args = parser.parse_args()
    cfg = load_json(args.config)
    games = args.games or int(cfg["benchmark_games"])
    model_a = _load_model(args.checkpoint_a, cfg)
    model_b = _load_model(args.checkpoint_b, cfg)
    print(compare_checkpoints(model_a, model_b, cfg, games=games))


if __name__ == "__main__":
    main()
