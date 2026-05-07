from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PACKAGE_DIR = Path(__file__).resolve().parents[1]

from neural_network.src.population_4p_training import configure_run_logging, run_population_4p_training
from neural_network.src.utils import ensure_dir, load_json, save_json

MAX_DURATION_MINUTES = 600.0


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


def _run_tag() -> str:
    return datetime.now(timezone.utc).strftime("run_10h_target_%Y%m%d_%H%M%S")


def _emit_direct_log(path: str, message: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(message + "\n")


def _prepare_config(
    cfg: dict,
    *,
    duration_minutes: float,
    workers: int,
    eval_episodes: int,
    promotion_eval_episodes: int,
    run_name: str,
) -> dict:
    cfg = dict(cfg)
    duration_minutes = min(float(duration_minutes), MAX_DURATION_MINUTES)

    cfg["duration_minutes"] = duration_minutes
    cfg["workers"] = max(1, int(workers))
    cfg["hidden_dim"] = max(320, int(cfg.get("hidden_dim", 320)))
    cfg["learning_rate"] = min(float(cfg.get("learning_rate", 0.00025)), 0.0002)
    cfg["train_steps"] = max(1, int(duration_minutes * 8))
    cfg["worker_train_steps"] = max(48, min(96, int(cfg.get("worker_train_steps", 64))))

    # Warmstart every candidate from deterministic heuristic teachers before RL.
    cfg["imitation_warmstart_steps"] = max(96, min(192, int(cfg.get("imitation_warmstart_steps", 128))))
    cfg["policy_prior_strength"] = max(1.35, float(cfg.get("policy_prior_strength", 1.35)))

    cfg["eval_episodes"] = max(32, int(eval_episodes))
    cfg["candidate_eval_episodes"] = cfg["eval_episodes"]
    cfg["promotion_eval_episodes"] = max(cfg["eval_episodes"], int(promotion_eval_episodes))
    cfg["promotion_min_remaining_minutes"] = max(20.0, float(cfg.get("promotion_min_remaining_minutes", 20.0)))
    cfg["min_generation_remaining_minutes"] = max(4.0, min(12.0, duration_minutes * 0.02))
    cfg["benchmark_games"] = max(128, cfg["promotion_eval_episodes"])

    cfg["curriculum_enabled"] = True
    cfg["opponent_curriculum_enabled"] = True
    cfg["opponent_curriculum_start_tier"] = 0
    cfg["curriculum_early_4p_ratio"] = 1.0
    cfg["curriculum_mid_4p_ratio"] = 1.0
    cfg["curriculum_late_4p_ratio"] = 1.0
    cfg["four_player_ratio"] = 1.0
    cfg["eval_four_player_ratio"] = 1.0
    cfg["notebook_pool_limit"] = 0
    cfg["notebook_pool_limit_max"] = 0
    cfg["train_notebook_opponents"] = 3
    cfg["game_engine"] = "official_fast"
    cfg["official_fast_c_accel"] = bool(cfg.get("official_fast_c_accel", True))
    cfg["train_stop_on_elimination"] = True
    cfg["max_actions_per_turn"] = 4
    cfg["max_turns"] = min(100, int(cfg.get("max_turns", 100)))
    cfg["min_expand_attack_ships"] = max(4, int(cfg.get("min_expand_attack_ships", 4)))
    cfg["send_ratios"] = [0.45, 0.65, 0.85]

    cfg["value_loss_coef"] = float(cfg.get("value_loss_coef", 0.25))
    cfg["dense_reward_enabled"] = True
    cfg["dense_planet_coef"] = 0.05
    cfg["dense_production_coef"] = 0.04
    cfg["dense_ship_share_coef"] = 0.14
    cfg["dense_score_coef"] = 0.10
    cfg["dense_survival_coef"] = 0.05
    cfg["dense_reward_clip"] = 0.30
    cfg["train_target_do_nothing_rate"] = 0.48
    cfg["train_noop_penalty_coef"] = 0.45
    cfg["train_action_bonus_coef"] = 0.09
    cfg["train_ships_sent_bonus_coef"] = 0.05
    cfg["train_activity_reward_clip"] = 0.35

    cfg["temperature_start"] = max(float(cfg.get("temperature_start", 1.1)), 1.10)
    cfg["temperature_end"] = min(float(cfg.get("temperature_end", 0.20)), 0.20)
    cfg["entropy_coef_start"] = min(float(cfg.get("entropy_coef_start", 0.035)), 0.035)
    cfg["entropy_coef_end"] = min(float(cfg.get("entropy_coef_end", 0.004)), 0.004)
    cfg["baseline_momentum"] = max(float(cfg.get("baseline_momentum", 0.08)), 0.10)

    # Explicitly avoid the implicit 0.125 bootstrap gate that blocked the previous run.
    cfg["promotion_min_winrate"] = max(0.03, float(cfg.get("promotion_min_winrate", 0.03)))
    cfg["promotion_margin"] = max(0.008, float(cfg.get("promotion_margin", 0.008)))
    cfg["promotion_max_do_nothing_rate"] = min(0.78, float(cfg.get("promotion_max_do_nothing_rate", 0.78)))
    cfg["promotion_min_avg_ships_sent"] = max(1.0, float(cfg.get("promotion_min_avg_ships_sent", 1.0)))
    cfg["bootstrap_promote_without_confirmation"] = True
    cfg["resume_from_tier_best"] = True

    cfg["opponent_curriculum_tiers"] = [
        {
            "name": "bootstrap_easy",
            "label": "bootstrap: random greedy starter",
            "opponents": ["random", "greedy", "starter"],
            "min_generations": 1,
            "advance_score": 0.12,
            "advance_winrate": 0.15,
            "advance_rank_mean": 2.85,
            "advance_do_nothing_rate": 0.70,
            "candidate_eval_episodes": 32,
        },
        {
            "name": "easy_heuristics",
            "label": "easy heuristics",
            "opponents": ["random", "greedy", "starter", "distance"],
            "min_generations": 2,
            "advance_score": 0.20,
            "advance_winrate": 0.30,
            "advance_rank_mean": 2.55,
            "advance_do_nothing_rate": 0.62,
            "candidate_eval_episodes": 32,
        },
        {
            "name": "mixed_heuristics",
            "label": "mixed public heuristics",
            "opponents": ["random", "greedy", "starter", "distance", "structured", "sun_dodge", "orbit_stars"],
            "min_generations": 2,
            "advance_score": 0.34,
            "advance_winrate": 0.50,
            "advance_rank_mean": 2.25,
            "advance_do_nothing_rate": 0.56,
            "candidate_eval_episodes": 32,
        },
        {
            "name": "target_70_validation",
            "label": "target 70 percent validation pool",
            "opponents": ["random", "greedy", "starter", "distance", "structured", "sun_dodge", "orbit_stars"],
            "min_generations": 0,
            "advance_score": 1.0,
            "advance_winrate": 0.70,
            "advance_rank_mean": 1.75,
            "advance_do_nothing_rate": 0.45,
            "candidate_eval_episodes": 32,
        },
    ]
    cfg["target_winrate"] = 0.70
    cfg["run_name"] = run_name

    logs_dir = PACKAGE_DIR / "logs" / "runs" / run_name
    checkpoints_dir = PACKAGE_DIR / "checkpoints" / "runs" / run_name
    ensure_dir(logs_dir)
    ensure_dir(checkpoints_dir)
    cfg["checkpoint_dir"] = str(checkpoints_dir)
    cfg["log_dir"] = str(logs_dir)
    cfg["candidate_checkpoint"] = str(checkpoints_dir / "candidate.npz")
    cfg["best_checkpoint"] = str(checkpoints_dir / "best.npz")
    cfg["latest_checkpoint"] = str(checkpoints_dir / "latest.npz")
    cfg["tier_checkpoint_dir"] = str(checkpoints_dir / "tiers")
    cfg["export_path"] = str(checkpoints_dir / "export.npz")
    cfg["opponent_curriculum_state"] = str(logs_dir / "opponent_curriculum_state.json")
    cfg["run_manifest_path"] = str(logs_dir / "run_manifest.json")
    cfg["log_direct_path"] = str(logs_dir / "log_direct.txt")
    cfg["resume_checkpoint"] = ""
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 10h target-winrate 4p training job.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--duration-minutes", type=float, default=600.0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--promotion-eval-episodes", type=int, default=128)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    run_name = args.run_name or _run_tag()
    cfg = _prepare_config(
        _load_config(args.config),
        duration_minutes=args.duration_minutes,
        workers=args.workers,
        eval_episodes=args.eval_episodes,
        promotion_eval_episodes=args.promotion_eval_episodes,
        run_name=run_name,
    )
    save_json(cfg["run_manifest_path"], cfg)
    configure_run_logging(cfg["log_direct_path"])
    start_line = (
        f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] "
        f"run_start run_name={run_name} target_winrate={cfg['target_winrate']:.2f} "
        f"log_direct={cfg['log_direct_path']}"
    )
    print(start_line, flush=True)
    _emit_direct_log(cfg["log_direct_path"], start_line)
    result = run_population_4p_training(cfg, resume=not args.no_resume)
    save_json(Path(cfg["log_dir"]) / "run_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
