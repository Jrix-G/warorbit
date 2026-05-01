from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_network.src.utils import load_json, ensure_dir
from neural_network.src.trainer import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)
    ensure_dir(config["checkpoint_dir"])
    ensure_dir(config["log_dir"])
    print(run_training(config, resume=not args.no_resume))


if __name__ == "__main__":
    main()

