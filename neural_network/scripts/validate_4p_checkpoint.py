from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PACKAGE_DIR = Path(__file__).resolve().parents[1]

from neural_network.scripts.run_10h_2p_then_4p_target import _evaluate_state, _load_config, _prepare_config
from neural_network.src.model import ModelConfig, NeuralNetworkModel, load_compatible_state_dict
from neural_network.src.storage import load_checkpoint
from neural_network.src.trainer import _infer_input_dim


def _load_model(path: Path, cfg: dict) -> NeuralNetworkModel:
    model = NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(cfg), hidden_dim=int(cfg.get("hidden_dim", 320))))
    state, _ = load_checkpoint(str(path))
    load_compatible_state_dict(model, state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a 4p checkpoint on a frozen evaluation pool.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed-offset", type=int, default=8800042)
    parser.add_argument("--run-name", default="validation_4p")
    args = parser.parse_args()

    cfg = _prepare_config(_load_config(args.config), argparse.Namespace(duration_minutes=600, workers=4, stage1_target=0.85, stage2_target=0.70), args.run_name)
    # Keep validation detached from training behavior, but reuse the same game setup.
    cfg["dense_reward_enabled"] = True
    cfg["game_engine"] = "official_fast"
    cfg["official_fast_c_accel"] = True
    model = _load_model(Path(args.checkpoint), cfg)
    record = _evaluate_state(
        model.state_dict(),
        cfg,
        stage="validation_4p",
        n_players=4,
        pool=["random", "greedy", "starter"],
        episodes=args.episodes,
        seed_offset=args.seed_offset,
    )
    print(json.dumps(record, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
