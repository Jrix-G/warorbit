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

MAX_DURATION_MINUTES = 480.0


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
    candidates.append(ROOT / "configs" / "default_config.json")
    candidates.append(ROOT / "neural_network" / "configs" / "default_config.json")
    for candidate in candidates:
        if candidate.exists():
            return load_json(str(candidate))
    raise FileNotFoundError(f"Config not found. Tried: {[str(candidate) for candidate in candidates]}")


def _prepare_config(cfg: dict, duration_minutes: float, workers: int, eval_episodes: int) -> dict:
    cfg = dict(cfg)
    cfg["duration_minutes"] = min(float(duration_minutes), MAX_DURATION_MINUTES)
    cfg["workers"] = max(1, int(workers))
    cfg["hidden_dim"] = max(320, int(cfg.get("hidden_dim", 320)))
    cfg["learning_rate"] = min(float(cfg.get("learning_rate", 0.0003)), 0.00025)
    fast_run = float(duration_minutes) <= 30.0
    cfg["train_steps"] = max(1, int(duration_minutes * 10))
    cfg["worker_train_steps"] = max(16 if fast_run else 24, int(cfg.get("worker_train_steps", 16 if fast_run else 24)))
    cfg["eval_episodes"] = max(16 if fast_run else 32, int(eval_episodes))
    cfg["candidate_eval_episodes"] = max(
        16 if fast_run else 32,
        min(cfg["eval_episodes"], max(16 if fast_run else 32, int(cfg.get("candidate_eval_episodes", 16 if fast_run else 32)))),
    )
    cfg["promotion_eval_episodes"] = max(32 if fast_run else 64, cfg["eval_episodes"])
    cfg["promotion_min_remaining_minutes"] = max(4.0 if fast_run else 8.0, float(cfg.get("promotion_min_remaining_minutes", 8.0 if fast_run else 12.0)))
    cfg["min_generation_remaining_minutes"] = max(
        1.5 if fast_run else 2.0,
        min(6.0 if fast_run else 18.0, float(duration_minutes) * (0.05 if fast_run else 0.08)),
    )
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
    cfg["game_engine"] = "official_fast"
    cfg["official_fast_c_accel"] = bool(cfg.get("official_fast_c_accel", True))
    cfg["max_actions_per_turn"] = 4
    cfg["min_expand_attack_ships"] = max(6, int(cfg.get("min_expand_attack_ships", 6)))
    cfg["value_loss_coef"] = float(cfg.get("value_loss_coef", 0.25))
    cfg["dense_reward_enabled"] = bool(cfg.get("dense_reward_enabled", True))
    cfg["dense_planet_coef"] = 0.05 if fast_run else 0.04
    cfg["dense_production_coef"] = 0.04 if fast_run else 0.03
    cfg["dense_ship_share_coef"] = 0.15 if fast_run else 0.12
    cfg["dense_score_coef"] = 0.10 if fast_run else 0.08
    cfg["dense_survival_coef"] = 0.08 if fast_run else 0.05
    cfg["dense_reward_clip"] = 0.40 if fast_run else 0.35
    cfg["imitation_warmstart_steps"] = max(256, int(cfg.get("imitation_warmstart_steps", 256)))
    cfg["opponent_curriculum_enabled"] = bool(cfg.get("opponent_curriculum_enabled", True))
    cfg.setdefault("opponent_curriculum_start_tier", 0)
    cfg.setdefault("opponent_curriculum_state", "neural_network/logs/opponent_curriculum_state.json")
    cfg.setdefault("resume_from_tier_best", True)
    cfg.setdefault("tier_checkpoint_dir", "neural_network/checkpoints/tiers")
    cfg["temperature_start"] = max(float(cfg.get("temperature_start", 1.15)), 1.20 if fast_run else 1.15)
    cfg["temperature_end"] = min(float(cfg.get("temperature_end", 0.25)), 0.20 if fast_run else 0.25)
    cfg["baseline_momentum"] = max(float(cfg.get("baseline_momentum", 0.05)), 0.08 if fast_run else 0.05)
    cfg["send_ratios"] = [0.5, 0.7, 0.9]
    cfg["policy_prior_strength"] = max(1.2, float(cfg.get("policy_prior_strength", 1.2)))
    cfg["promotion_margin"] = max(0.02, float(cfg.get("promotion_margin", 0.02)))
    cfg["bootstrap_promote_without_confirmation"] = bool(cfg.get("bootstrap_promote_without_confirmation", True))
    if fast_run:
        cfg["max_turns"] = min(int(cfg.get("max_turns", 100)), 80)
        cfg["opponent_curriculum_tiers"] = [
            {
                "name": "basic_300",
                "label": "300 Elo: random/greedy/starter",
                "opponents": ["random", "greedy", "starter"],
                "min_generations": 1,
                "advance_score": 0.18,
                "advance_winrate": 0.10,
                "advance_rank_mean": 3.30,
                "advance_do_nothing_rate": 0.70,
                "candidate_eval_episodes": 4,
            },
            {
                "name": "heuristic_500",
                "label": "500 Elo: public heuristics",
                "opponents": ["greedy", "starter", "distance", "sun_dodge", "structured", "orbit_stars"],
                "min_generations": 1,
                "advance_score": 0.20,
                "advance_winrate": 0.08,
                "advance_rank_mean": 3.15,
                "advance_do_nothing_rate": 0.66,
                "candidate_eval_episodes": 6,
            },
            {
                "name": "mixed_700",
                "label": "700 Elo: heuristics plus starter notebooks",
                "opponents": [
                    "distance",
                    "sun_dodge",
                    "structured",
                    "orbit_stars",
                    "notebook_kashiwaba_orbit_wars_reinforcement_learning_tutorial",
                    "notebook_sigmaborov_orbit_wars_2026_starter",
                    "notebook_pilkwang_orbit_wars_structured_baseline",
                    "notebook_djenkivanov_orbit_wars_optimized_nearest_planet_sniper",
                ],
                "min_generations": 1,
                "advance_score": 0.22,
                "advance_winrate": 0.06,
                "advance_rank_mean": 3.00,
                "advance_do_nothing_rate": 0.60,
                "candidate_eval_episodes": 6,
            },
            {
                "name": "notebook_core4",
                "label": "core notebooks plus heuristics",
                "opponents": [
                    "distance",
                    "sun_dodge",
                    "structured",
                    "orbit_stars",
                    "notebook_orbitbotnext",
                    "notebook_distance_prioritized",
                    "notebook_physics_accurate",
                    "notebook_tactical_heuristic",
                ],
                "min_generations": 1,
                "advance_score": 0.24,
                "advance_winrate": 0.05,
                "advance_rank_mean": 2.90,
                "advance_do_nothing_rate": 0.55,
                "candidate_eval_episodes": 8,
            },
            {
                "name": "notebook_mid8",
                "label": "first eight notebook opponents",
                "opponents": "notebook_pool:8",
                "min_generations": 1,
                "advance_score": 0.26,
                "advance_winrate": 0.04,
                "advance_rank_mean": 2.85,
                "advance_do_nothing_rate": 0.52,
                "candidate_eval_episodes": 8,
            },
            {
                "name": "notebook_open",
                "label": "full notebook pool",
                "opponents": "notebook_pool",
                "min_generations": 0,
                "advance_score": 1.0,
                "advance_winrate": 1.0,
                "advance_rank_mean": 1.0,
                "advance_do_nothing_rate": 0.0,
                "candidate_eval_episodes": 8,
            },
        ]
    for key in ("checkpoint_dir", "log_dir", "candidate_checkpoint", "best_checkpoint", "latest_checkpoint", "tier_checkpoint_dir", "export_path", "opponent_curriculum_state"):
        if key in cfg:
            cfg[key] = _resolve_path(str(cfg[key]))
    if cfg.get("best_checkpoint"):
        cfg["resume_checkpoint"] = cfg["best_checkpoint"]
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run up to 8 hours of 4p population training with six neural agents.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--duration-minutes", type=float, default=90.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = _prepare_config(_load_config(args.config), args.duration_minutes, args.workers, args.eval_episodes)
    ensure_dir(cfg["checkpoint_dir"])
    ensure_dir(cfg["log_dir"])
    result = run_population_4p_training(cfg, resume=not args.no_resume)
    print(json.dumps(result, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
