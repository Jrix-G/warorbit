from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_network.src.utils import load_json
from neural_network.src.model import NeuralNetworkModel, ModelConfig
from neural_network.src.trainer import _infer_input_dim
from neural_network.src.benchmark import benchmark_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="neural_network/configs/default_config.json")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    cfg = load_json(args.config)
    model = NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(cfg), hidden_dim=int(cfg.get("hidden_dim", 128))))
    print(json.dumps(benchmark_model(model, cfg, games=args.episodes), indent=2))


if __name__ == "__main__":
    main()
