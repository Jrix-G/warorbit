#!/usr/bin/env python3
"""Train and evaluate the V9 War Orbit agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from war_orbit.agents.v9.policy import V9Weights, load_checkpoint, get_weights
from war_orbit.config.v9_config import V9Config
from war_orbit.evaluation.benchmark import benchmark_v9, print_report
from war_orbit.training.curriculum import build_role_pools
from war_orbit.training.trainer import V9Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V9 self-improving training and evaluation.")
    parser.add_argument("--minutes", type=float, default=60.0)
    parser.add_argument("--hard-timeout-minutes", type=float, default=60.0)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--games-per-eval", type=int, default=2)
    parser.add_argument("--eval-games", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--eval-max-steps", type=int, default=220)
    parser.add_argument("--four-player-ratio", type=float, default=0.80)
    parser.add_argument("--eval-four-player-ratio", type=float, default=None)
    parser.add_argument("--benchmark-four-player-ratio", type=float, default=0.80)
    parser.add_argument("--sigma", type=float, default=0.07)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--l2", type=float, default=0.0004)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--benchmark-every", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--game-engine", choices=("kaggle", "official_fast", "kaggle_fast"), default="official_fast")
    parser.add_argument("--train-only", action="store_true", help="Skip in-loop eval/benchmark and optimize for training volume.")
    parser.add_argument("--benchmark-games", type=int, default=128)
    parser.add_argument("--min-promotion-benchmark-games", type=int, default=128)
    parser.add_argument("--benchmark-progress-every", type=int, default=1)
    parser.add_argument("--min-benchmark-score", type=float, default=0.15)
    parser.add_argument("--max-generalization-gap", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=9009)
    parser.add_argument("--checkpoint", default="evaluations/v9_robust_policy_latest.npz")
    parser.add_argument("--best-checkpoint", default="evaluations/v9_robust_policy_best.npz")
    parser.add_argument("--export-checkpoint", default="evaluations/v9_robust_policy.npz")
    parser.add_argument("--log-jsonl", default="evaluations/v9_robust_train.jsonl")
    parser.add_argument("--pool-limit", type=int, default=15)
    parser.add_argument("--search-width", type=int, default=8)
    parser.add_argument("--simulation-depth", type=int, default=3)
    parser.add_argument("--simulation-rollouts", type=int, default=2)
    parser.add_argument("--train-search-width", type=int, default=4)
    parser.add_argument("--train-simulation-depth", type=int, default=1)
    parser.add_argument("--train-simulation-rollouts", type=int, default=0)
    parser.add_argument("--train-opponent-samples", type=int, default=1)
    parser.add_argument("--candidate-diversity", type=float, default=1.15)
    parser.add_argument("--opening-punch-turns", type=int, default=55)
    parser.add_argument("--opening-min-capture-send-2p", type=int, default=14)
    parser.add_argument("--opening-min-capture-send-4p", type=int, default=16)
    parser.add_argument("--midgame-min-capture-send-4p", type=int, default=24)
    parser.add_argument("--capture-garrison-margin", type=float, default=0.22)
    parser.add_argument("--capture-target-ship-margin", type=float, default=0.15)
    parser.add_argument("--midgame-capture-target-margin-4p", type=float, default=0.35)
    parser.add_argument("--opening-close-neutral-dist-4p", type=float, default=42.0)
    parser.add_argument("--opening-long-attack-risk-dist-4p", type=float, default=55.0)
    parser.add_argument("--opening-source-commit-frac", type=float, default=1.0)
    parser.add_argument("--front-lock-turns", type=int, default=24)
    parser.add_argument("--target-active-fronts", type=float, default=2.0)
    parser.add_argument("--target-backbone-turn-frac", type=float, default=0.15)
    parser.add_argument("--front-penalty-weight", type=float, default=0.055)
    parser.add_argument("--front-penalty-cap", type=float, default=0.12)
    parser.add_argument("--front-ok-bonus", type=float, default=0.045)
    parser.add_argument("--front-partial-bonus", type=float, default=0.025)
    parser.add_argument("--backbone-penalty-weight", type=float, default=0.080)
    parser.add_argument("--backbone-bonus-weight", type=float, default=0.060)
    parser.add_argument("--front-pressure-plan-bias", type=float, default=0.12)
    parser.add_argument("--front-pressure-attack-penalty", type=float, default=0.12)
    parser.add_argument("--guardian-enabled", type=int, default=1)
    parser.add_argument("--guardian-min-benchmark-4p", type=float, default=0.42)
    parser.add_argument("--guardian-min-benchmark-backbone", type=float, default=0.08)
    parser.add_argument("--guardian-max-benchmark-fronts", type=float, default=2.70)
    parser.add_argument("--guardian-max-generalization-gap", type=float, default=0.18)
    parser.add_argument("--export-best-on-finish", type=int, default=1)
    parser.add_argument("--snapshot-every", type=int, default=1, help="Save recoverable generation snapshots every N generations; 0 disables.")
    parser.add_argument("--snapshot-dir", default=None, help="Directory for generation snapshots. Defaults next to --checkpoint.")
    parser.add_argument("--strict-single-target-4p", type=int, default=0)
    parser.add_argument("--disable-snipe-4p", type=int, default=0)
    parser.add_argument("--max-focus-targets-4p", type=int, default=2)
    parser.add_argument("--exploration-rate", type=float, default=0.08)
    parser.add_argument("--confidence-l2", type=float, default=0.0025)
    parser.add_argument("--reward-noise", type=float, default=0.015)
    parser.add_argument("--four-p-signal-boost", type=float, default=1.4)
    parser.add_argument("--train-state-perturbation", type=float, default=0.035)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--opponents", nargs="*", default=None, help="Alias for --train-opponents.")
    parser.add_argument("--train-opponents", nargs="*", default=None)
    parser.add_argument("--eval-opponents", nargs="*", default=None)
    parser.add_argument("--benchmark-opponents", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = V9Config(
        minutes=args.minutes,
        hard_timeout_minutes=args.hard_timeout_minutes,
        pairs=args.pairs,
        games_per_eval=args.games_per_eval,
        eval_games=args.eval_games,
        max_steps=args.max_steps,
        eval_max_steps=args.eval_max_steps,
        four_player_ratio=args.four_player_ratio,
        eval_four_player_ratio=args.eval_four_player_ratio,
        sigma=args.sigma,
        learning_rate=args.lr,
        l2=args.l2,
        confidence_l2=args.confidence_l2,
        reward_noise=args.reward_noise,
        four_p_signal_boost=args.four_p_signal_boost,
        train_state_perturbation=args.train_state_perturbation,
        eval_every=args.eval_every,
        benchmark_every=args.benchmark_every,
        workers=args.workers,
        game_engine=args.game_engine,
        train_only=args.train_only,
        benchmark_games=args.benchmark_games,
        min_promotion_benchmark_games=args.min_promotion_benchmark_games,
        benchmark_progress_every=args.benchmark_progress_every,
        benchmark_four_player_ratio=args.benchmark_four_player_ratio,
        min_benchmark_score=args.min_benchmark_score,
        max_generalization_gap=args.max_generalization_gap,
        seed=args.seed,
        checkpoint=args.checkpoint,
        best_checkpoint=args.best_checkpoint,
        export_checkpoint=args.export_checkpoint,
        log_jsonl=args.log_jsonl,
        opponent_pool_limit=args.pool_limit,
        search_width=args.search_width,
        simulation_depth=args.simulation_depth,
        simulation_rollouts=args.simulation_rollouts,
        train_search_width=args.train_search_width,
        train_simulation_depth=args.train_simulation_depth,
        train_simulation_rollouts=args.train_simulation_rollouts,
        train_opponent_samples=args.train_opponent_samples,
        candidate_diversity=args.candidate_diversity,
        exploration_rate=args.exploration_rate,
        opening_punch_turns=args.opening_punch_turns,
        opening_min_capture_send_2p=args.opening_min_capture_send_2p,
        opening_min_capture_send_4p=args.opening_min_capture_send_4p,
        midgame_min_capture_send_4p=args.midgame_min_capture_send_4p,
        capture_garrison_margin=args.capture_garrison_margin,
        capture_target_ship_margin=args.capture_target_ship_margin,
        midgame_capture_target_margin_4p=args.midgame_capture_target_margin_4p,
        opening_close_neutral_dist_4p=args.opening_close_neutral_dist_4p,
        opening_long_attack_risk_dist_4p=args.opening_long_attack_risk_dist_4p,
        opening_source_commit_frac=args.opening_source_commit_frac,
        front_lock_turns=args.front_lock_turns,
        target_active_fronts=args.target_active_fronts,
        target_backbone_turn_frac=args.target_backbone_turn_frac,
        front_penalty_weight=args.front_penalty_weight,
        front_penalty_cap=args.front_penalty_cap,
        front_ok_bonus=args.front_ok_bonus,
        front_partial_bonus=args.front_partial_bonus,
        backbone_penalty_weight=args.backbone_penalty_weight,
        backbone_bonus_weight=args.backbone_bonus_weight,
        front_pressure_plan_bias=args.front_pressure_plan_bias,
        front_pressure_attack_penalty=args.front_pressure_attack_penalty,
        guardian_enabled=bool(args.guardian_enabled),
        guardian_min_benchmark_4p=args.guardian_min_benchmark_4p,
        guardian_min_benchmark_backbone=args.guardian_min_benchmark_backbone,
        guardian_max_benchmark_fronts=args.guardian_max_benchmark_fronts,
        guardian_max_generalization_gap=args.guardian_max_generalization_gap,
        export_best_on_finish=bool(args.export_best_on_finish),
        snapshot_every=args.snapshot_every,
        snapshot_dir=args.snapshot_dir,
        strict_single_target_4p=bool(args.strict_single_target_4p),
        disable_snipe_4p=bool(args.disable_snipe_4p),
        max_focus_targets_4p=args.max_focus_targets_4p,
    )
    if args.opponents:
        config.training_opponents = list(args.opponents)
    if args.train_opponents:
        config.training_opponents = list(args.train_opponents)
    if args.eval_opponents:
        config.eval_opponents = list(args.eval_opponents)
    if args.benchmark_opponents:
        config.benchmark_opponents = list(args.benchmark_opponents)

    weights = V9Weights.defaults()
    if not args.skip_training:
        trainer = V9Trainer(config, resume=not args.no_resume)
        train_summary = trainer.train()
        weights = trainer.weights
        if train_summary.get("stop_reason") == "timeout":
            args.skip_eval = True
    else:
        loaded = load_checkpoint(config.export_checkpoint) or load_checkpoint(config.best_checkpoint) or load_checkpoint(config.checkpoint)
        weights = get_weights() if loaded else weights

    if not args.skip_eval:
        eval_opponents = build_role_pools(config)["benchmark"]
        report = benchmark_v9(
            weights,
            config,
            eval_opponents,
            games=max(1, config.benchmark_games),
            max_steps=config.eval_max_steps,
            four_player_ratio=config.benchmark_four_player_ratio,
            seed_offset=88000,
        )
        print_report(report)


if __name__ == "__main__":
    main()
