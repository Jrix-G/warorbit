from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_network.src.utils import load_json
from neural_network.src.model import NeuralNetworkModel, ModelConfig
from neural_network.src.self_play import play_episode
from neural_network.src.trainer import _infer_input_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    model = NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(cfg)))
    episode = play_episode(model, cfg, seed=0)
    print({"steps": len(episode), "reward": sum(s["reward"] for s in episode)})


if __name__ == "__main__":
    main()

