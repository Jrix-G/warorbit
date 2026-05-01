from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_network.src.utils import load_json
from neural_network.src.model import NeuralNetworkModel, ModelConfig
from neural_network.src.trainer import _infer_input_dim
from neural_network.src.benchmark import benchmark_model, benchmark_matchups
from neural_network.src.storage import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="neural_network/configs/default_config.json")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--checkpoint", default="neural_network/checkpoints/latest.npz")
    args = parser.parse_args()
    cfg = load_json(args.config)
    model = NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(cfg), hidden_dim=int(cfg.get("hidden_dim", 128))))
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        state, _ = load_checkpoint(checkpoint)
        model.load_state_dict(state)
    result = benchmark_model(model, cfg, games=args.episodes)
    result.update(benchmark_matchups(model, cfg, episodes=args.episodes))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
