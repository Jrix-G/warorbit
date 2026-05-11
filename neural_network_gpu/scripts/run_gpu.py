"""GPU vectorized training: N parallel env workers + GPU inference server + GPU trainer."""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACKAGE_DIR = Path(__file__).resolve().parents[2]

from neural_network.src.model import ModelConfig, NeuralNetworkModel, load_compatible_state_dict, count_parameters
from neural_network.src.notebook_4p_training import _infer_input_dim
from neural_network.src.storage import append_jsonl, load_checkpoint, save_checkpoint
from neural_network.src.utils import ensure_dir, save_json
from neural_network.src.population_4p_training import configure_run_logging

from neural_network_gpu.src.vec_worker import worker_fn
from neural_network_gpu.src.inference_server import inference_server_fn
from neural_network_gpu.src.gpu_trainer import train_on_episodes


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(path: Path, msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    with path.open("a") as f:
        f.write(line + "\n")


def _run_tag() -> str:
    return datetime.now(timezone.utc).strftime("gpu_%Y%m%d_%H%M%S")


def _model_state_np(model: NeuralNetworkModel) -> Dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _auto_tune_training(
    cfg: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    train_history: List[Dict[str, float]],
    metrics: Dict[str, float],
    mission_counts: Dict[str, int],
) -> Dict[str, Any]:
    if not bool(cfg.get("auto_tune_training", False)):
        return {}

    patch: Dict[str, Any] = {}
    reasons: List[str] = []
    lr = float(optimizer.param_groups[0]["lr"])
    recent = train_history[-5:] or [metrics]
    entropy = float(np.mean([row.get("entropy", 0.0) for row in recent]))
    clip_frac = float(np.mean([row.get("clip_frac", 0.0) for row in recent]))
    approx_kl = float(np.mean([row.get("approx_kl", 0.0) for row in recent]))
    ratio_std = float(np.mean([row.get("ratio_std", 0.0) for row in recent]))
    param_delta = float(np.mean([row.get("param_relative_delta", 0.0) for row in recent]))
    log_ratio_abs = float(np.mean([row.get("log_ratio_abs_max", 0.0) for row in recent]))
    total_missions = max(1, sum(int(v) for v in mission_counts.values()))
    noop_rate = float(mission_counts.get("do_nothing", 0)) / float(total_missions)

    min_lr = float(cfg.get("min_lr", 1e-6))
    max_lr = float(cfg.get("max_lr", 2e-4))
    new_lr = lr

    enough_history = len(train_history) >= 3
    if enough_history and clip_frac < 0.003 and ratio_std < 0.025 and param_delta < 0.018:
        new_lr = min(max_lr, new_lr * 1.06)
        reasons.append("update_too_small")
    if clip_frac > 0.075 or approx_kl > 0.010 or log_ratio_abs > 1.25:
        new_lr = max(min_lr, new_lr * 0.90)
        reasons.append("update_too_large")
    if abs(new_lr - lr) > 1e-12:
        _set_optimizer_lr(optimizer, new_lr)
        patch["learning_rate"] = new_lr

    if enough_history and entropy > 0.0 and entropy < 1.70:
        patch["entropy_coef_start"] = _clamp_float(float(cfg.get("entropy_coef_start", 0.10)) * 1.05, 0.04, 0.18)
        patch["temperature_end"] = _clamp_float(float(cfg.get("temperature_end", 0.18)) * 1.05, 0.18, 0.65)
        patch["temperature_start"] = _clamp_float(float(cfg.get("temperature_start", 1.05)) * 1.02, 0.75, 1.35)
        reasons.append("entropy_low")
    elif enough_history and entropy > 2.90 and clip_frac > 0.020:
        patch["entropy_coef_start"] = _clamp_float(float(cfg.get("entropy_coef_start", 0.10)) * 0.96, 0.04, 0.18)
        patch["temperature_start"] = _clamp_float(float(cfg.get("temperature_start", 1.05)) * 0.98, 0.75, 1.35)
        reasons.append("entropy_high")

    target_noop = float(cfg.get("train_target_do_nothing_rate", 0.18))
    avg_noop_rate = float(np.mean([row.get("noop_rate", noop_rate) for row in recent]))
    if enough_history and avg_noop_rate > target_noop + 0.25:
        patch["train_noop_penalty_coef"] = _clamp_float(float(cfg.get("train_noop_penalty_coef", 0.40)) * 1.04, 0.10, 0.90)
        patch["train_action_bonus_coef"] = _clamp_float(float(cfg.get("train_action_bonus_coef", 0.28)) * 1.03, 0.05, 0.55)
        patch["train_ships_sent_bonus_coef"] = _clamp_float(float(cfg.get("train_ships_sent_bonus_coef", 0.18)) * 1.03, 0.03, 0.40)
        reasons.append("noop_high")

    if patch:
        cfg.update(patch)
        patch["auto_tune_reasons"] = ",".join(reasons)
        patch["auto_tune_noop_rate"] = avg_noop_rate
    return patch


def _weighted_pool_from_counts(counts: Dict[str, int]) -> List[str]:
    pool: List[str] = []
    for name in ("random", "greedy", "starter"):
        pool.extend([name] * max(0, int(counts.get(name, 0))))
    return pool or ["random", "greedy", "starter"]


def _auto_tune_opponent_mix(
    cfg: Dict[str, Any],
    eval_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not bool(cfg.get("auto_tune_training", False)) or not eval_history:
        return {}

    current_counts = dict(cfg.get("opponent_mix_counts", {"random": 6, "greedy": 13, "starter": 1}))
    latest = eval_history[-1]
    by_opp = latest.get("by_opponent", {})
    wr = {name: float(by_opp.get(name, {}).get("winrate", 0.0)) for name in ("random", "greedy", "starter")}
    noop = float(latest.get("eval_do_nothing_rate", 1.0))

    new_counts = dict(current_counts)
    reasons: List[str] = []

    if wr["random"] >= 0.70 and wr["greedy"] < 0.55:
        new_counts = {"random": 4, "greedy": 15, "starter": 1}
        reasons.append("focus_greedy")
    elif wr["random"] >= 0.70 and wr["greedy"] >= 0.60 and wr["starter"] < 0.35:
        new_counts = {"random": 4, "greedy": 14, "starter": 2}
        reasons.append("introduce_starter")
    elif wr["random"] >= 0.75 and wr["greedy"] >= 0.70 and wr["starter"] >= 0.35:
        new_counts = {"random": 3, "greedy": 13, "starter": 4}
        reasons.append("starter_ramp")
    elif wr["random"] < 0.45:
        new_counts = {"random": 8, "greedy": 11, "starter": 1}
        reasons.append("repair_random")

    if noop > float(cfg.get("max_eval_do_nothing_rate", 0.55)):
        new_counts["starter"] = min(new_counts.get("starter", 1), 2)
        reasons.append("hold_starter_noop_high")

    if new_counts == current_counts:
        return {}

    pool = _weighted_pool_from_counts(new_counts)
    cfg["opponent_mix_counts"] = new_counts
    cfg["stage1_pool"] = pool
    if cfg.get("curriculum_tiers"):
        cfg["curriculum_tiers"][0]["opponents"] = pool
    return {
        "opponent_mix_counts": new_counts,
        "stage1_pool": pool,
        "auto_tune_reasons": ",".join(reasons),
    }


def _wilson_ci(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 0.0
    p = float(wins) / float(games)
    denom = 1.0 + (z * z) / games
    centre = p + (z * z) / (2.0 * games)
    spread = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * games)) / games)
    return max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom)


def _load_model_from_checkpoint(path: Path, config: Dict[str, Any], device: torch.device) -> NeuralNetworkModel:
    model = NeuralNetworkModel(ModelConfig(
        input_dim=_infer_input_dim(config),
        hidden_dim=int(config["hidden_dim"]),
    ))
    state, _ = load_checkpoint(path)
    load_compatible_state_dict(model, state)
    return model.to(device).eval()


def _checkpoint_opponent_paths(run_dir: Path, max_items: int) -> List[str]:
    archive_dir = run_dir / "archive"
    if max_items <= 0 or not archive_dir.exists():
        return []
    paths = sorted(archive_dir.glob("validated_*.npz"))
    return [f"checkpoint:{path}" for path in paths[-max_items:]]


def _parse_opponent_list(raw_value: str, fallback: List[str]) -> List[str]:
    opponents = [
        item.strip()
        for item in str(raw_value).split(",")
        if item.strip()
    ]
    return opponents or list(fallback)


def _build_config(args: argparse.Namespace) -> Dict[str, Any]:
    default_opponents = ["random", "greedy", "starter"]
    simple_opponents = _parse_opponent_list(args.simple_opponents, default_opponents)
    eval_opponents = _parse_opponent_list(args.eval_opponents, list(dict.fromkeys(simple_opponents)))
    opponent_mix_counts = {name: simple_opponents.count(name) for name in ("random", "greedy", "starter")}
    return {
        # Game
        "game_engine": "official_fast",
        "official_fast_c_accel": True,
        "max_turns": 100,
        "max_actions_per_turn": 4,
        "min_expand_attack_ships": 6,
        "send_ratios": [0.25, 0.35, 0.50, 0.65, 0.80, 0.95],
        "allow_support_actions": not bool(args.disable_support_actions),
        "policy_prior_strength": 0.0,
        # Encoder
        "max_planets": 64,
        "max_fleets": 128,
        "max_players": 4,
        "board_scale": 100.0,
        "ship_scale": 2000.0,
        "production_scale": 10.0,
        "radius_scale": 10.0,
        "horizon_scale": 100.0,
        "planet_id_scale": 64.0,
        # Model
        "hidden_dim": 320,
        # Training
        "learning_rate": args.learning_rate,
        "gamma": 0.99,
        "value_loss_coef": 0.25,
        "entropy_coef_start": 0.10,
        "baseline_momentum": 0.10,
        "max_grad_norm": 1.0,
        # PPO
        "ppo_clip_eps": args.ppo_clip_eps,
        "ppo_epochs": args.ppo_epochs,
        "ppo_minibatch_size": args.ppo_minibatch_size,
        # Activity shaping
        "dense_reward_enabled": True,
        "dense_planet_coef": 0.05,
        "dense_production_coef": 0.04,
        "dense_ship_share_coef": 0.14,
        "dense_score_coef": 0.10,
        "dense_survival_coef": 0.05,
        "dense_reward_clip": 0.40,
        "train_target_do_nothing_rate": 0.18,
        "train_noop_penalty_coef": 0.40,
        "train_action_bonus_coef": 0.28,
        "train_ships_sent_bonus_coef": 0.18,
        "train_activity_reward_clip": 0.55,
        # Temperature
        "temperature_start": 1.05,
        "temperature_end": 0.18,
        "temperature_decay_updates": args.temperature_decay_updates,
        # Pool
        "stage1_pool": simple_opponents,
        "eval_opponents": eval_opponents,
        "opponent_mix_counts": opponent_mix_counts,
        "simple_2p_only": True,
        "league_archive_size": args.league_archive_size,
        "curriculum_tiers": [
            {"name": "simple_2p", "opponents": simple_opponents},
        ],
        "curriculum_tier": 0,
        # GPU
        "n_workers": args.workers,
        "train_every": args.train_every,
        "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes,
        "max_batch_size": args.batch_size,
        "batch_timeout": args.batch_timeout,
        "device": args.device,
        # Run
        "duration_minutes": args.duration_minutes,
        "target_winrate": args.target_winrate,
        "n_players": args.n_players,
        "seed": 42,
        "eval_seed_start": args.eval_seed_start,
        "promotion_margin": args.promotion_margin,
        "rollback_margin": args.rollback_margin,
        "min_lr": args.min_lr,
        "max_lr": args.max_lr,
        "promotion_lr_mult": args.promotion_lr_mult,
        "rollback_lr_mult": args.rollback_lr_mult,
        "max_eval_do_nothing_rate": args.max_eval_do_nothing_rate,
        "min_eval_avg_ships_sent": args.min_eval_avg_ships_sent,
        "max_opponent_regression": args.max_opponent_regression,
        "min_ci_promotion_games": args.min_ci_promotion_games,
        "auto_tune_training": bool(args.auto_tune_training),
    }


def _evaluate(
    model: NeuralNetworkModel,
    config: Dict[str, Any],
    device: torch.device,
    episodes: int,
    seed_start: int,
    pool: List[str],
    n_players: int,
    eval_history_path: Path | None = None,
    checkpoint_label: str = "candidate",
    episode_count: int = 0,
    lr: float = 0.0,
    train_metrics: Dict[str, float] | None = None,
    progress_log_path: Path | None = None,
) -> Dict[str, Any]:
    from neural_network.src.notebook_4p_training import (
        _build_agents, run_match, _action_summary,
    )

    model.eval()
    eval_cfg = dict(config)
    eval_cfg["temperature_end"] = 0.0
    eval_cfg["temperature_start"] = 0.0
    eval_cfg["policy_prior_strength"] = float(config.get("policy_prior_strength", 0.0))

    opponents = list(pool) or ["random"]
    per_opponent: Dict[str, Any] = {}
    all_wins: List[float] = []
    all_do_nothing_rates: List[float] = []
    all_avg_ships_sent: List[float] = []
    train_metrics = train_metrics or {}

    for opponent_idx, opponent in enumerate(opponents):
        if progress_log_path is not None:
            _log(progress_log_path, f"eval_start checkpoint={checkpoint_label} opponent={opponent} games={episodes}")
        wins: List[float] = []
        do_nothing_rates: List[float] = []
        avg_ships_sent: List[float] = []
        seed_base = int(seed_start) + opponent_idx * max(100000, episodes + 1)
        for i in range(episodes):
            seed = seed_base + i
            our_index = i % n_players
            action_records: List = []
            agents, _, action_records, _ = _build_agents(
                model, eval_cfg, seed, our_index,
                temperature=0.0, pool=[opponent], explore=False,
                n_players=n_players,
            )
            if len(agents) != n_players:
                raise ValueError(f"_build_agents returned {len(agents)} agents, expected {n_players}")
            result = run_match(
                agents, seed=seed, n_players=n_players,
                max_steps=int(config.get("max_turns", 100)),
                stop_player=our_index,
                game_engine=str(config.get("game_engine", "official_fast")),
                use_c_accel=bool(config.get("official_fast_c_accel", True)),
            )
            winner = int(result.get("winner", -1))
            wins.append(float(winner == our_index))
            metrics = _action_summary(action_records)
            do_nothing_rates.append(float(metrics.get("do_nothing_rate", 1.0)))
            avg_ships_sent.append(float(metrics.get("avg_ships_sent", 0.0)))
            if progress_log_path is not None and ((i + 1) == episodes or (i + 1) % max(1, episodes // 4) == 0):
                _log(
                    progress_log_path,
                    f"eval_progress checkpoint={checkpoint_label} opponent={opponent} "
                    f"games={i + 1}/{episodes} wins={int(sum(wins))}",
                )

        win_count = int(sum(wins))
        ci_low, ci_high = _wilson_ci(win_count, episodes)
        record = {
            "episode": int(episode_count),
            "checkpoint": checkpoint_label,
            "opponent": str(opponent),
            "games": int(episodes),
            "wins": win_count,
            "losses": int(episodes - win_count),
            "winrate": float(np.mean(wins)) if wins else 0.0,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "do_nothing_rate": float(np.mean(do_nothing_rates)) if do_nothing_rates else 1.0,
            "avg_ships_sent": float(np.mean(avg_ships_sent)) if avg_ships_sent else 0.0,
            "avg_reward": float(train_metrics.get("mean_reward", 0.0)),
            "terminal_reward": float(train_metrics.get("terminal_reward_mean", 0.0)),
            "dense_reward": float(train_metrics.get("dense_reward_mean", 0.0)),
            "activity_reward": float(train_metrics.get("activity_reward_mean", 0.0)),
            "lr": float(lr),
            "temperature": float(train_metrics.get("mean_sample_temperature", 0.0)),
            "entropy": float(train_metrics.get("entropy", 0.0)),
            "approx_kl": float(train_metrics.get("approx_kl", 0.0)),
            "clip_frac": float(train_metrics.get("clip_frac", 0.0)),
            "grad_norm": float(train_metrics.get("grad_norm", 0.0)),
            "policy_version": int(train_metrics.get("policy_version", 0.0)),
            "seed_start": int(seed_base),
            "seed_count": int(episodes),
        }
        per_opponent[str(opponent)] = record
        if progress_log_path is not None:
            _log(
                progress_log_path,
                f"eval_done checkpoint={checkpoint_label} opponent={opponent} "
                f"winrate={record['winrate']:.3f} noop={record['do_nothing_rate']:.3f} ships={record['avg_ships_sent']:.2f}",
            )
        if eval_history_path is not None:
            append_jsonl(eval_history_path, record)
        all_wins.extend(wins)
        all_do_nothing_rates.extend(do_nothing_rates)
        all_avg_ships_sent.extend(avg_ships_sent)

    total_games = len(all_wins)
    total_wins = int(sum(all_wins))
    ci_low, ci_high = _wilson_ci(total_wins, total_games)
    return {
        "winrate": float(np.mean(all_wins)) if all_wins else 0.0,
        "wins": total_wins,
        "losses": int(total_games - total_wins),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "eval_do_nothing_rate": float(np.mean(all_do_nothing_rates)) if all_do_nothing_rates else 1.0,
        "eval_avg_ships_sent": float(np.mean(all_avg_ships_sent)) if all_avg_ships_sent else 0.0,
        "eval_episodes": int(episodes),
        "eval_games": int(total_games),
        "seed_start": int(seed_start),
        "seed_count": int(episodes),
        "by_opponent": per_opponent,
    }


def _promotion_decision(
    candidate_eval: Dict[str, Any],
    best_eval: Dict[str, Any],
    config: Dict[str, Any],
) -> tuple[bool, List[str], bool]:
    margin = float(config.get("promotion_margin", 0.03))
    rollback_margin = float(config.get("rollback_margin", 0.08))
    max_do_nothing = float(config.get("max_eval_do_nothing_rate", 0.75))
    min_avg_ships = float(config.get("min_eval_avg_ships_sent", 0.0))
    max_opp_regression = float(config.get("max_opponent_regression", 0.12))
    min_ci_games = int(config.get("min_ci_promotion_games", 96))
    reasons: List[str] = []

    cand_wr = float(candidate_eval.get("winrate", 0.0))
    best_wr = float(best_eval.get("winrate", 0.0))
    delta = cand_wr - best_wr
    promote = True
    if delta < margin:
        promote = False
        reasons.append(f"delta {delta:.4f} < margin {margin:.4f}")
    eval_games = int(candidate_eval.get("eval_games", 0))
    if eval_games >= min_ci_games and float(candidate_eval.get("ci_low", 0.0)) <= best_wr:
        promote = False
        reasons.append(
            f"candidate ci_low {float(candidate_eval.get('ci_low', 0.0)):.4f} <= best_winrate {best_wr:.4f}"
        )
    if float(candidate_eval.get("eval_do_nothing_rate", 1.0)) > max_do_nothing:
        promote = False
        reasons.append("do_nothing gate failed")
    if float(candidate_eval.get("eval_avg_ships_sent", 0.0)) < min_avg_ships:
        promote = False
        reasons.append("avg_ships_sent gate failed")

    rollback = delta <= -rollback_margin
    cand_by_opp = candidate_eval.get("by_opponent", {})
    best_by_opp = best_eval.get("by_opponent", {})
    for opponent, cand_record in cand_by_opp.items():
        best_record = best_by_opp.get(opponent)
        if not best_record:
            continue
        opp_delta = float(cand_record.get("winrate", 0.0)) - float(best_record.get("winrate", 0.0))
        if opp_delta < -max_opp_regression:
            promote = False
            rollback = True
            reasons.append(f"{opponent} regression {opp_delta:.4f}")
    return promote, reasons, rollback


def _archive_validated_checkpoint(run_dir: Path, total_episodes: int, model: NeuralNetworkModel, metadata: Dict[str, Any]) -> Path:
    archive_dir = run_dir / "archive"
    ensure_dir(archive_dir)
    path = archive_dir / f"validated_{int(total_episodes):09d}.npz"
    save_checkpoint(path, _model_state_np(model), metadata)
    return path


def main() -> None:
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-minutes", type=float, default=600.0)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--train-every", type=int, default=64, help="Train after N episodes")
    parser.add_argument("--eval-every", type=int, default=256, help="Eval after N episodes")
    parser.add_argument("--eval-episodes", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.00006)
    parser.add_argument("--ppo-clip-eps", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--ppo-minibatch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch-timeout", type=float, default=0.010)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-winrate", type=float, default=0.85)
    parser.add_argument("--n-players", type=int, default=2)
    parser.add_argument("--eval-seed-start", type=int, default=900000)
    parser.add_argument("--promotion-margin", type=float, default=0.03)
    parser.add_argument("--rollback-margin", type=float, default=0.08)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--max-lr", type=float, default=2e-4)
    parser.add_argument("--promotion-lr-mult", type=float, default=1.05)
    parser.add_argument("--rollback-lr-mult", type=float, default=0.5)
    parser.add_argument("--max-eval-do-nothing-rate", type=float, default=0.75)
    parser.add_argument("--min-eval-avg-ships-sent", type=float, default=0.0)
    parser.add_argument("--max-opponent-regression", type=float, default=0.12)
    parser.add_argument("--min-ci-promotion-games", type=int, default=96)
    parser.add_argument("--league-archive-size", type=int, default=4)
    parser.add_argument("--simple-opponents", default="random,greedy,starter")
    parser.add_argument("--eval-opponents", default="")
    parser.add_argument("--disable-support-actions", action="store_true")
    parser.add_argument("--auto-tune-training", action="store_true")
    parser.add_argument("--temperature-decay-updates", type=int, default=200)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--runs-root", default="", help="Override output directory for runs, useful on Kaggle")
    args = parser.parse_args()

    cfg = _build_config(args)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    resume_meta: Dict[str, Any] = {}
    if args.resume_checkpoint and Path(args.resume_checkpoint).exists():
        _, resume_meta = load_checkpoint(args.resume_checkpoint)
        adaptive_config = resume_meta.get("adaptive_config", {})
        if isinstance(adaptive_config, dict):
            for key in (
                "learning_rate",
                "entropy_coef_start",
                "temperature_start",
                "temperature_end",
                "train_noop_penalty_coef",
                "train_action_bonus_coef",
                "train_ships_sent_bonus_coef",
            ):
                if key in adaptive_config:
                    cfg[key] = adaptive_config[key]
            mix_counts = adaptive_config.get("opponent_mix_counts")
            if isinstance(mix_counts, dict):
                cfg["opponent_mix_counts"] = {str(k): int(v) for k, v in mix_counts.items()}
                cfg["stage1_pool"] = _weighted_pool_from_counts(cfg["opponent_mix_counts"])
                if cfg.get("curriculum_tiers"):
                    cfg["curriculum_tiers"][0]["opponents"] = list(cfg["stage1_pool"])

    run_name = args.run_name or (Path(args.resume_checkpoint).resolve().parent.name if args.resume_checkpoint and Path(args.resume_checkpoint).exists() else _run_tag())

    runs_root = Path(args.runs_root).expanduser() if args.runs_root else PACKAGE_DIR / "runs"
    run_dir = runs_root / run_name
    ensure_dir(run_dir)
    log_path = run_dir / "gpu_train.log"
    checkpoint_path = run_dir / "best.npz"
    best_validated_path = run_dir / "best_validated.npz"
    candidate_path = run_dir / "candidate.npz"
    latest_path = run_dir / "latest.npz"
    eval_history_path = run_dir / "eval_history.jsonl"

    save_json(run_dir / "config.json", cfg)
    _log(log_path, f"run_start name={run_name} device={device} workers={cfg['n_workers']} target={cfg['target_winrate']}")
    _log(log_path, f"policy_prior_strength={cfg['policy_prior_strength']} (external prior disabled; prior remains candidate feature)")

    # Build model
    model = NeuralNetworkModel(ModelConfig(
        input_dim=_infer_input_dim(cfg),
        hidden_dim=int(cfg["hidden_dim"]),
    ))
    if args.resume_checkpoint and Path(args.resume_checkpoint).exists():
        state, _ = load_checkpoint(args.resume_checkpoint)
        load_compatible_state_dict(model, state)
        _log(log_path, f"resumed from {args.resume_checkpoint}")
    model = model.to(device)
    model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    _log(log_path, f"params={count_parameters(model)} lr={cfg['learning_rate']}")
    if not best_validated_path.exists():
        save_checkpoint(
            best_validated_path,
            _model_state_np(model),
            {
                "winrate": float(resume_meta.get("winrate", -1.0)),
                "baseline": float(resume_meta.get("baseline", 0.0)),
                "total_episodes": int(resume_meta.get("total_episodes", 0)),
                "run_name": run_name,
                "seed_start": int(cfg["eval_seed_start"]),
                "note": "initial best_validated seeded from resume/current model",
            },
        )
        save_checkpoint(checkpoint_path, _model_state_np(model), {"winrate": float(resume_meta.get("winrate", -1.0)), "run_name": run_name})

    # Queues
    obs_queue: mp.Queue = mp.Queue(maxsize=cfg["n_workers"] * 8)
    action_queues: Dict[int, mp.Queue] = {i: mp.Queue(maxsize=8) for i in range(cfg["n_workers"])}
    control_queues: Dict[int, mp.Queue] = {i: mp.Queue(maxsize=4) for i in range(cfg["n_workers"])}
    result_queue: mp.Queue = mp.Queue(maxsize=cfg["n_workers"] * 4)
    model_update_queue: mp.Queue = mp.Queue(maxsize=2)
    stop_event: mp.Event = mp.Event()

    tiers = list(cfg.get("curriculum_tiers", []))
    tier_idx = max(0, min(int(cfg.get("curriculum_tier", 0)), len(tiers) - 1)) if tiers else 0
    pool = list(tiers[tier_idx].get("opponents", cfg["stage1_pool"])) if tiers else list(cfg["stage1_pool"])
    n_players = int(cfg["n_players"])
    if n_players < 2:
        raise ValueError(f"n_players must be >= 2, got {n_players}")
    use_simple_2p_only = bool(cfg.get("simple_2p_only", False)) and n_players == 2
    if best_validated_path.exists() and not use_simple_2p_only:
        pool.append(f"checkpoint:{best_validated_path}")
    if not use_simple_2p_only:
        pool.extend(_checkpoint_opponent_paths(run_dir, int(cfg.get("league_archive_size", 0))))
    _log(log_path, f"curriculum_tier={tiers[tier_idx].get('name', 'stage1') if tiers else 'stage1'} train_pool={pool}")

    # Start inference server
    initial_state = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    inf_proc = mp.Process(
        target=inference_server_fn,
        args=(initial_state, cfg, obs_queue, action_queues, model_update_queue, stop_event),
        kwargs={
            "device_str": str(device),
            "max_batch_size": cfg["max_batch_size"],
            "batch_timeout": float(cfg["batch_timeout"]),
        },
        daemon=True,
    )
    inf_proc.start()
    _log(log_path, "inference_server started")

    # Start env workers
    worker_procs = []
    for wid in range(cfg["n_workers"]):
        p = mp.Process(
            target=worker_fn,
            args=(
                wid, cfg, pool, n_players,
                obs_queue, action_queues[wid],
                result_queue, control_queues[wid], stop_event,
                int(cfg["seed"]) + wid * 1000,
            ),
            daemon=True,
        )
        p.start()
        worker_procs.append(p)
    _log(log_path, f"{cfg['n_workers']} workers started")

    started_at = time.time()
    deadline = started_at + float(cfg["duration_minutes"]) * 60.0

    baseline = float(resume_meta.get("baseline", 0.0))
    best_meta: Dict[str, Any] = {}
    if best_validated_path.exists():
        _, best_meta = load_checkpoint(best_validated_path)
    best_winrate = float(best_meta.get("winrate", resume_meta.get("winrate", -1.0)))
    total_episodes = int(resume_meta.get("total_episodes", 0))
    last_eval_episode = int(resume_meta.get("last_eval_episode", 0))
    pending_episodes: List[Dict[str, Any]] = []
    policy_version = 0
    last_train_metrics: Dict[str, float] = {}
    train_history = deque(maxlen=12)
    eval_history = deque(maxlen=8)
    consecutive_regressions = 0

    try:
        while time.time() < deadline:
            # Collect episode results
            try:
                ep = result_queue.get(timeout=1.0)
                pending_episodes.append(ep)
                total_episodes += 1
            except Empty:
                continue

            # Train every N episodes
            if len(pending_episodes) >= cfg["train_every"]:
                train_started = time.time()
                baseline, metrics = train_on_episodes(
                    model, optimizer, pending_episodes, cfg, device,
                    baseline, float(cfg["baseline_momentum"]),
                )
                train_seconds = max(1e-6, time.time() - train_started)
                elapsed = (time.time() - started_at) / 60.0
                episodes_per_hour = total_episodes / max(1e-6, (time.time() - started_at) / 3600.0)
                wins = [ep["win"] for ep in pending_episodes]
                last_train_metrics = dict(metrics)
                last_train_metrics["train_seconds"] = float(train_seconds)
                last_train_metrics["episodes_per_hour"] = float(episodes_per_hour)
                policy_version += 1
                mission_counts: Dict[str, int] = {}
                for ep in pending_episodes:
                    for key, value in ep.get("action_metrics", {}).get("mission_counts", {}).items():
                        mission_counts[key] = mission_counts.get(key, 0) + int(value)
                total_missions = max(1, sum(int(v) for v in mission_counts.values()))
                train_record = dict(metrics)
                train_record["noop_rate"] = float(mission_counts.get("do_nothing", 0)) / float(total_missions)
                train_record["train_winrate"] = float(np.mean(wins)) if wins else 0.0
                train_history.append(train_record)
                config_patch = _auto_tune_training(cfg, optimizer, list(train_history), metrics, mission_counts)
                if config_patch:
                    _log(
                        log_path,
                        "AUTO_TUNE "
                        f"reasons={config_patch.get('auto_tune_reasons', '')} "
                        f"noop={config_patch.get('auto_tune_noop_rate', 0.0):.3f} "
                        f"lr={float(optimizer.param_groups[0]['lr']):.8f} "
                        f"entropy_coef={float(cfg.get('entropy_coef_start', 0.0)):.4f} "
                        f"temp=[{float(cfg.get('temperature_start', 0.0)):.3f},{float(cfg.get('temperature_end', 0.0)):.3f}] "
                        f"noop_penalty={float(cfg.get('train_noop_penalty_coef', 0.0)):.3f} "
                        f"action_bonus={float(cfg.get('train_action_bonus_coef', 0.0)):.3f} "
                        f"ship_bonus={float(cfg.get('train_ships_sent_bonus_coef', 0.0)):.3f}",
                    )
                _log(
                    log_path,
                    f"train episodes={total_episodes} elapsed={elapsed:.1f}m "
                    f"eps_per_hour={episodes_per_hour:.1f} "
                    f"train_s={train_seconds:.2f} "
                    f"samples={metrics.get('train_samples', 0):.0f} "
                    f"minibatches={metrics.get('train_minibatches', 0):.0f} "
                    f"winrate={np.mean(wins):.3f} "
                    f"policy_loss={metrics.get('policy_loss', 0):.3f} "
                    f"value_loss={metrics.get('value_loss', 0):.3f} "
                    f"total_loss={metrics.get('total_loss', 0):.3f} "
                    f"entropy={metrics.get('entropy', 0):.3f} "
                    f"kl={metrics.get('approx_kl', 0):.4f} "
                    f"clip_frac={metrics.get('clip_frac', 0):.3f} "
                    f"ratio_std={metrics.get('ratio_std', 0):.6f} "
                    f"ratio_range=[{metrics.get('ratio_min', 0):.4f},{metrics.get('ratio_max', 0):.4f}] "
                    f"logr_max={metrics.get('log_ratio_abs_max', 0):.6f} "
                    f"param_delta={metrics.get('param_relative_delta', 0):.8f} "
                    f"grad_norm={metrics.get('grad_norm', 0):.3f} "
                    f"reward={metrics.get('mean_reward', 0):.3f} "
                    f"terminal={metrics.get('terminal_reward_mean', 0):.3f} "
                    f"dense={metrics.get('dense_reward_mean', 0):.3f} "
                    f"activity={metrics.get('activity_reward_mean', 0):.3f} "
                    f"temp={metrics.get('mean_sample_temperature', 0):.3f} "
                    f"skipped_oldlp={metrics.get('skipped_missing_old_log_prob', 0):.0f} "
                    f"missions={mission_counts} "
                    f"baseline={baseline:.3f}",
                )
                # Push updated weights to inference server
                new_state = {
                    "state": _model_state_np(model),
                    "policy_version": policy_version,
                    "config_patch": config_patch,
                }
                try:
                    model_update_queue.put_nowait(new_state)
                except Exception:
                    pass
                if config_patch:
                    control_msg = {"config_patch": dict(config_patch)}
                    for queue in control_queues.values():
                        try:
                            queue.put_nowait(control_msg)
                        except Exception:
                            pass

                save_checkpoint(
                    latest_path,
                    _model_state_np(model),
                    {
                        "winrate": best_winrate,
                        "baseline": baseline,
                        "total_episodes": total_episodes,
                        "last_eval_episode": last_eval_episode,
                        "run_name": run_name,
                        "policy_version": policy_version,
                        "train_metrics": metrics,
                        "adaptive_config": {
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "entropy_coef_start": float(cfg.get("entropy_coef_start", 0.0)),
                            "temperature_start": float(cfg.get("temperature_start", 0.0)),
                            "temperature_end": float(cfg.get("temperature_end", 0.0)),
                            "train_noop_penalty_coef": float(cfg.get("train_noop_penalty_coef", 0.0)),
                            "train_action_bonus_coef": float(cfg.get("train_action_bonus_coef", 0.0)),
                            "train_ships_sent_bonus_coef": float(cfg.get("train_ships_sent_bonus_coef", 0.0)),
                            "opponent_mix_counts": dict(cfg.get("opponent_mix_counts", {})),
                        },
                    },
                )
                pending_episodes = []

            # Eval every N episodes
            if total_episodes - last_eval_episode >= cfg["eval_every"]:
                last_eval_episode = total_episodes
                current_lr = float(optimizer.param_groups[0]["lr"])
                eval_pool = list(cfg.get("eval_opponents", []))
                if not use_simple_2p_only:
                    eval_pool += _checkpoint_opponent_paths(
                        run_dir, int(cfg.get("league_archive_size", 0))
                    )
                save_checkpoint(
                    candidate_path,
                    _model_state_np(model),
                    {
                        "winrate": best_winrate,
                        "baseline": baseline,
                        "total_episodes": total_episodes,
                        "last_eval_episode": last_eval_episode,
                        "run_name": run_name,
                        "policy_version": policy_version,
                        "train_metrics": last_train_metrics,
                    },
                )
                eval_result = _evaluate(
                    model, cfg, device,
                    episodes=cfg["eval_episodes"],
                    seed_start=int(cfg["eval_seed_start"]),
                    pool=eval_pool,
                    n_players=n_players,
                    eval_history_path=eval_history_path,
                    checkpoint_label="candidate",
                    episode_count=total_episodes,
                    lr=current_lr,
                    train_metrics=last_train_metrics,
                    progress_log_path=log_path,
                )
                eval_history.append(eval_result)
                mix_patch = _auto_tune_opponent_mix(cfg, list(eval_history))
                if mix_patch:
                    pool = list(mix_patch["stage1_pool"])
                    control_msg = {"pool": pool}
                    for queue in control_queues.values():
                        try:
                            queue.put_nowait(control_msg)
                        except Exception:
                            pass
                    _log(
                        log_path,
                        "AUTO_MIX "
                        f"reasons={mix_patch.get('auto_tune_reasons', '')} "
                        f"counts={mix_patch.get('opponent_mix_counts', {})} "
                        f"pool={pool}",
                    )
                    save_json(run_dir / "config.json", cfg)
                best_model = _load_model_from_checkpoint(best_validated_path, cfg, device)
                best_eval = _evaluate(
                    best_model, cfg, device,
                    episodes=cfg["eval_episodes"],
                    seed_start=int(cfg["eval_seed_start"]),
                    pool=eval_pool,
                    n_players=n_players,
                    eval_history_path=eval_history_path,
                    checkpoint_label="best_validated",
                    episode_count=total_episodes,
                    lr=current_lr,
                    train_metrics=last_train_metrics,
                    progress_log_path=log_path,
                )
                elapsed = (time.time() - started_at) / 60.0
                promote, promotion_reasons, rollback = _promotion_decision(eval_result, best_eval, cfg)
                _log(
                    log_path,
                    f"eval episodes={total_episodes} elapsed={elapsed:.1f}m "
                    f"winrate={eval_result['winrate']:.3f} "
                    f"ci=[{eval_result['ci_low']:.3f},{eval_result['ci_high']:.3f}] "
                    f"best_eval={best_eval['winrate']:.3f} "
                    f"best_ci=[{best_eval['ci_low']:.3f},{best_eval['ci_high']:.3f}] "
                    f"noop={eval_result['eval_do_nothing_rate']:.3f} "
                    f"ships={eval_result['eval_avg_ships_sent']:.2f} "
                    f"lr={current_lr:.8f} "
                    f"best={best_winrate:.3f} target={cfg['target_winrate']}",
                )
                if promote:
                    best_winrate = eval_result["winrate"]
                    metadata = {
                        **eval_result,
                        "baseline": baseline,
                        "total_episodes": total_episodes,
                        "last_eval_episode": last_eval_episode,
                        "run_name": run_name,
                        "policy_version": policy_version,
                        "train_metrics": last_train_metrics,
                        "adaptive_config": {
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "entropy_coef_start": float(cfg.get("entropy_coef_start", 0.0)),
                            "temperature_start": float(cfg.get("temperature_start", 0.0)),
                            "temperature_end": float(cfg.get("temperature_end", 0.0)),
                            "train_noop_penalty_coef": float(cfg.get("train_noop_penalty_coef", 0.0)),
                            "train_action_bonus_coef": float(cfg.get("train_action_bonus_coef", 0.0)),
                            "train_ships_sent_bonus_coef": float(cfg.get("train_ships_sent_bonus_coef", 0.0)),
                            "opponent_mix_counts": dict(cfg.get("opponent_mix_counts", {})),
                        },
                    }
                    save_checkpoint(best_validated_path, _model_state_np(model), metadata)
                    save_checkpoint(checkpoint_path, _model_state_np(model), metadata)
                    archive_path = _archive_validated_checkpoint(run_dir, total_episodes, model, metadata)
                    new_lr = min(float(cfg["max_lr"]), current_lr * float(cfg["promotion_lr_mult"]))
                    _set_optimizer_lr(optimizer, new_lr)
                    consecutive_regressions = 0
                    _log(
                        log_path,
                        f"PROMOTED winrate={best_winrate:.4f} archive={archive_path} lr={current_lr:.8f}->{new_lr:.8f}",
                    )
                else:
                    _log(log_path, f"not promoted reasons={promotion_reasons}")
                    if rollback:
                        consecutive_regressions += 1
                        failed_path = run_dir / f"candidate_failed_{int(total_episodes):09d}.npz"
                        save_checkpoint(
                            failed_path,
                            _model_state_np(model),
                            {**eval_result, "failure_reasons": promotion_reasons, "total_episodes": total_episodes},
                        )
                        state, _ = load_checkpoint(best_validated_path)
                        load_compatible_state_dict(model, state)
                        model = model.to(device)
                        model.eval()
                        new_lr = max(float(cfg["min_lr"]), current_lr * float(cfg["rollback_lr_mult"]))
                        optimizer = torch.optim.Adam(model.parameters(), lr=new_lr)
                        policy_version += 1
                        try:
                            model_update_queue.put_nowait({"state": _model_state_np(model), "policy_version": policy_version})
                        except Exception:
                            pass
                        _log(
                            log_path,
                            f"ROLLBACK failed={failed_path} regressions={consecutive_regressions} "
                            f"lr={current_lr:.8f}->{new_lr:.8f}",
                        )
                if best_winrate >= cfg["target_winrate"]:
                    _log(log_path, f"TARGET REACHED winrate={best_winrate:.4f}")
                    break

    finally:
        stop_event.set()
        for p in worker_procs:
            p.join(timeout=5)
        inf_proc.join(timeout=5)

    elapsed = (time.time() - started_at) / 60.0
    result = {
        "run_name": run_name,
        "elapsed_minutes": elapsed,
        "total_episodes": total_episodes,
        "best_winrate": best_winrate,
        "target_winrate": cfg["target_winrate"],
        "target_reached": best_winrate >= cfg["target_winrate"],
        "best_checkpoint": str(best_validated_path) if best_validated_path.exists() else "",
        "legacy_best_checkpoint": str(checkpoint_path) if checkpoint_path.exists() else "",
        "eval_history": str(eval_history_path),
    }
    save_json(run_dir / "result.json", result)
    _log(log_path, f"run_done best_winrate={best_winrate:.4f} episodes={total_episodes}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
