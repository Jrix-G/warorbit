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


def _prepare_config(cfg: dict, duration_minutes: float, eval_episodes: int, curriculum: bool) -> dict:
    cfg = dict(cfg)
    cfg["train_steps"] = max(1, int(duration_minutes * 10))
    cfg["eval_episodes"] = max(1, int(eval_episodes))
    cfg["curriculum_enabled"] = bool(curriculum)
    cfg.setdefault("curriculum_early_4p_ratio", 0.3)
    cfg.setdefault("curriculum_mid_4p_ratio", 0.6)
    cfg.setdefault("curriculum_late_4p_ratio", 1.0)
    cfg.setdefault("curriculum_phase1_steps", max(1, int(cfg["train_steps"] * 0.35)))
    cfg.setdefault("curriculum_phase2_steps", max(2, int(cfg["train_steps"] * 0.70)))
    cfg.setdefault("promotion_margin", 0.05)
    cfg.setdefault("promotion_min_eval_std", 0.25)
    cfg.setdefault("candidate_checkpoint", str(Path(cfg["checkpoint_dir"]) / "candidate.npz"))
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a breakthrough curriculum training cycle for neural_network.")
    parser.add_argument("--config", default="neural_network/configs/default_config.json")
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = load_json(args.config)
    cfg = _prepare_config(cfg, args.duration_minutes, args.eval_episodes, args.curriculum)
    ensure_dir(cfg["checkpoint_dir"])
    ensure_dir(cfg["log_dir"])
    result = run_training(cfg, resume=not args.no_resume)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
