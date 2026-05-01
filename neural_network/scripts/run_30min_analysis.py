from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_network.src.utils import load_json, ensure_dir
from neural_network.src.trainer import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="neural_network/configs/default_config.json")
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    args = parser.parse_args()
    cfg = load_json(args.config)
    cfg["train_steps"] = max(1, int(args.duration_minutes * 10))
    cfg["max_turns"] = int(cfg.get("max_turns", 100))
    cfg["gamma"] = float(cfg.get("gamma", 0.99))
    cfg["learning_rate"] = float(cfg.get("learning_rate", 0.0003))
    ensure_dir(cfg["checkpoint_dir"])
    ensure_dir(cfg["log_dir"])
    result = run_training(cfg, resume=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
