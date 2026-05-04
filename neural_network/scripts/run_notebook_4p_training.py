from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_network.src.utils import load_json, ensure_dir
from neural_network.src.notebook_4p_training import run_notebook_4p_training


def _prepare_config(cfg: dict, duration_minutes: float, eval_episodes: int) -> dict:
    cfg = dict(cfg)
    cfg["train_steps"] = max(1, int(duration_minutes * 10))
    cfg["eval_episodes"] = max(1, int(eval_episodes))
    cfg["benchmark_games"] = max(1, int(eval_episodes))
    cfg["curriculum_enabled"] = True
    cfg["curriculum_early_4p_ratio"] = 1.0
    cfg["curriculum_mid_4p_ratio"] = 1.0
    cfg["curriculum_late_4p_ratio"] = 1.0
    cfg["four_player_ratio"] = 1.0
    cfg["eval_four_player_ratio"] = 1.0
    cfg["notebook_pool_limit"] = 15
    cfg["train_notebook_opponents"] = 3
    cfg["hidden_dim"] = max(320, int(cfg.get("hidden_dim", 320)))
    cfg["max_actions_per_turn"] = 4
    cfg["eval_every"] = max(1, int(cfg["train_steps"] // 10))
    cfg.setdefault("temperature_start", 1.2)
    cfg.setdefault("temperature_end", 0.35)
    cfg.setdefault("send_ratios", [0.25, 0.5, 0.75])
    cfg.setdefault("policy_prior_strength", 0.8)
    cfg.setdefault("value_loss_coef", 0.25)
    cfg.setdefault("promotion_margin", 0.05)
    cfg.setdefault("promotion_min_eval_std", 0.20)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the neural network in 4p against notebook opponents.")
    parser.add_argument("--config", default="neural_network/configs/default_config.json")
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = load_json(args.config)
    cfg = _prepare_config(cfg, args.duration_minutes, args.eval_episodes)
    ensure_dir(cfg["checkpoint_dir"])
    ensure_dir(cfg["log_dir"])
    result = run_notebook_4p_training(cfg, resume=not args.no_resume)
    print(json.dumps(result, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
