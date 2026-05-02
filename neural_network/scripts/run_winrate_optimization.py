from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_network.src.utils import load_json, ensure_dir
from neural_network.src.trainer import run_training


def _prepare_config(cfg: dict, duration_minutes: float, eval_episodes: int) -> dict:
    cfg = dict(cfg)
    cfg["train_steps"] = max(1, int(duration_minutes * 10))
    cfg["batch_size"] = max(16, int(cfg.get("batch_size", 32)))
    cfg["eval_episodes"] = max(1, int(eval_episodes))
    cfg["benchmark_games"] = max(1, int(eval_episodes))
    cfg.setdefault("baseline_momentum", 0.05)
    cfg.setdefault("eval_every_updates", 4)
    cfg.setdefault("temperature_start", 1.2)
    cfg.setdefault("temperature_end", 0.35)
    cfg.setdefault("entropy_coef_start", 0.01)
    cfg.setdefault("entropy_coef_end", 0.002)
    cfg.setdefault("exploration_decay", "linear")
    cfg.setdefault("promotion_margin", 0.02)
    cfg.setdefault("promotion_min_eval_std", 0.20)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize neural_network checkpoints for winrate.")
    parser.add_argument("--config", default="neural_network/configs/default_config.json")
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = load_json(args.config)
    cfg = _prepare_config(cfg, args.duration_minutes, args.eval_episodes)
    ensure_dir(cfg["checkpoint_dir"])
    ensure_dir(cfg["log_dir"])

    best = Path(cfg["best_checkpoint"])
    latest = Path(cfg["latest_checkpoint"])
    if best.exists():
        cfg["resume_checkpoint"] = str(best)
    elif latest.exists():
        cfg["resume_checkpoint"] = str(latest)

    result = run_training(cfg, resume=not args.no_resume)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
