from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PACKAGE_DIR = Path(__file__).resolve().parents[1]

from neural_network.src.population_4p_training import run_population_4p_training
from neural_network.src.utils import ensure_dir, load_json


def _resolve_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    parts = path.parts
    if parts and parts[0] == "neural_network":
        return str(PACKAGE_DIR.joinpath(*parts[1:]))
    return str(path)


def _load_config(path: str | None) -> dict:
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.append(PACKAGE_DIR / "configs" / "default_config.json")
    candidates.append(ROOT / "neural_network" / "configs" / "default_config.json")
    for candidate in candidates:
        if candidate.exists():
            return load_json(str(candidate))
    raise FileNotFoundError(f"Config not found. Tried: {[str(candidate) for candidate in candidates]}")


def _prepare_config(cfg: dict, workers: int, eval_episodes: int) -> dict:
    cfg = dict(cfg)
    cfg["duration_minutes"] = 20.0
    cfg["workers"] = max(1, int(workers))
    cfg["hidden_dim"] = max(320, int(cfg.get("hidden_dim", 320)))
    cfg["learning_rate"] = min(float(cfg.get("learning_rate", 0.0003)), 0.00025)
    cfg["train_steps"] = 200
    cfg["worker_train_steps"] = max(12, int(cfg.get("worker_train_steps", 12)))
    cfg["eval_episodes"] = max(4, int(eval_episodes))
    cfg["candidate_eval_episodes"] = max(4, min(cfg["eval_episodes"], int(cfg.get("candidate_eval_episodes", 4))))
    cfg["promotion_eval_episodes"] = max(12, cfg["eval_episodes"])
    cfg["benchmark_games"] = cfg["eval_episodes"]
    cfg["curriculum_enabled"] = True
    cfg["curriculum_early_4p_ratio"] = 1.0
    cfg["curriculum_mid_4p_ratio"] = 1.0
    cfg["curriculum_late_4p_ratio"] = 1.0
    cfg["four_player_ratio"] = 1.0
    cfg["eval_four_player_ratio"] = 1.0
    cfg["notebook_pool_limit"] = 15
    cfg["notebook_pool_limit_max"] = 15
    cfg["train_notebook_opponents"] = 3
    cfg["train_stop_on_elimination"] = bool(cfg.get("train_stop_on_elimination", True))
    cfg["game_engine"] = str(cfg.get("game_engine", "official_fast"))
    cfg["official_fast_c_accel"] = bool(cfg.get("official_fast_c_accel", True))
    cfg["max_actions_per_turn"] = 4
    cfg["dense_reward_enabled"] = bool(cfg.get("dense_reward_enabled", True))
    cfg.setdefault("imitation_warmstart_steps", 4)
    cfg["opponent_curriculum_enabled"] = bool(cfg.get("opponent_curriculum_enabled", True))
    cfg.setdefault("opponent_curriculum_start_tier", 0)
    cfg.setdefault("opponent_curriculum_state", "neural_network/logs/opponent_curriculum_state.json")
    cfg.setdefault("resume_from_tier_best", True)
    cfg.setdefault("tier_checkpoint_dir", "neural_network/checkpoints/tiers")
    cfg["temperature_start"] = float(cfg.get("temperature_start", 1.15))
    cfg["temperature_end"] = float(cfg.get("temperature_end", 0.25))
    cfg.setdefault("send_ratios", [0.25, 0.5, 0.75])
    cfg.setdefault("policy_prior_strength", 0.8)
    cfg["promotion_margin"] = float(cfg.get("promotion_margin", 0.0))
    cfg["bootstrap_promote_without_confirmation"] = bool(cfg.get("bootstrap_promote_without_confirmation", True))
    for key in ("checkpoint_dir", "log_dir", "candidate_checkpoint", "best_checkpoint", "latest_checkpoint", "tier_checkpoint_dir", "export_path", "opponent_curriculum_state"):
        if key in cfg:
            cfg[key] = _resolve_path(str(cfg[key]))
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fast 20-minute 4p population training job.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = _prepare_config(_load_config(args.config), args.workers, args.eval_episodes)
    ensure_dir(cfg["checkpoint_dir"])
    ensure_dir(cfg["log_dir"])
    result = run_population_4p_training(cfg, resume=not args.no_resume)
    print(json.dumps(result, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
