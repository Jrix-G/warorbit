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
from neural_network_gpu.src.action_metrics import summarize_action_records, summarize_fleet_events


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


def _put_latest_model_update(queue: mp.Queue, msg: Dict[str, Any]) -> tuple[bool, int]:
    dropped = 0
    while True:
        try:
            queue.get_nowait()
            dropped += 1
        except Empty:
            break
        except Exception:
            break
    try:
        queue.put_nowait(msg)
        return True, dropped
    except Exception:
        return False, dropped


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _raise_ratio_floor(ratios: List[float], step: float = 0.10, floor_cap: float = 0.55) -> List[float]:
    adjusted = []
    for ratio in ratios:
        value = float(ratio)
        if value < floor_cap:
            value = min(floor_cap, value + step)
        adjusted.append(round(value, 3))
    return sorted(set(adjusted))


def _lower_ratio_ceiling(ratios: List[float], step: float = 0.08, ceiling_floor: float = 0.65) -> List[float]:
    adjusted = []
    for ratio in ratios:
        value = float(ratio)
        if value > ceiling_floor:
            value = max(ceiling_floor, value - step)
        adjusted.append(round(value, 3))
    return sorted(set(adjusted))


def _weighted_eval_score(result: Dict[str, Any], weights: Dict[str, float] | None = None) -> float:
    by_opp = result.get("by_opponent", {})
    weights = weights or {"random": 0.15, "greedy": 0.25, "starter": 0.40, "distance": 0.20}
    total_weight = 0.0
    score = 0.0
    for opponent, weight in weights.items():
        if opponent not in by_opp:
            continue
        total_weight += float(weight)
        record = by_opp.get(opponent, {})
        score += float(weight) * float(record.get("valid_winrate", record.get("winrate", 0.0)))
    if total_weight <= 0.0:
        return float(result.get("valid_winrate", result.get("winrate", 0.0)))
    return score / total_weight


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
        patch["entropy_coef_start"] = _clamp_float(float(cfg.get("entropy_coef_start", 0.10)) * 1.05, 0.010, 0.18)
        patch["temperature_end"] = _clamp_float(float(cfg.get("temperature_end", 0.18)) * 1.05, 0.18, 0.65)
        patch["temperature_start"] = _clamp_float(float(cfg.get("temperature_start", 1.05)) * 1.02, 0.75, 1.35)
        reasons.append("entropy_low")
    elif enough_history and entropy > 2.90 and clip_frac > 0.020:
        patch["entropy_coef_start"] = _clamp_float(float(cfg.get("entropy_coef_start", 0.10)) * 0.96, 0.010, 0.18)
        patch["temperature_start"] = _clamp_float(float(cfg.get("temperature_start", 1.05)) * 0.98, 0.75, 1.35)
        reasons.append("entropy_high")

    # Prefer the legal (avoidable) no-op rate as the auto-tune driver. Falls
    # back to raw rate for warm-up evals or rows missing the new metric.
    avg_noop_rate = float(np.mean([
        row.get("legal_noop_rate", row.get("noop_rate", noop_rate)) for row in recent
    ]))
    target_noop = float(cfg.get("train_target_legal_noop_rate", 0.10))
    if enough_history and avg_noop_rate > target_noop + 0.05:
        patch["train_noop_penalty_coef"] = _clamp_float(float(cfg.get("train_noop_penalty_coef", 0.70)) * 1.04, 0.10, 2.20)
        patch["train_passivity_penalty_coef"] = _clamp_float(float(cfg.get("train_passivity_penalty_coef", 0.55)) * 1.06, 0.10, 2.00)
        patch["do_nothing_logit_penalty"] = _clamp_float(float(cfg.get("do_nothing_logit_penalty", 0.50)) + 0.10, 0.0, 2.0)
        if not bool(cfg.get("train_event_shaping_enabled", True)):
            patch["train_action_bonus_coef"] = _clamp_float(float(cfg.get("train_action_bonus_coef", 0.28)) * 1.03, 0.05, 0.70)
            patch["train_ships_sent_bonus_coef"] = _clamp_float(float(cfg.get("train_ships_sent_bonus_coef", 0.18)) * 1.03, 0.03, float(cfg.get("stabilizer_max_ship_bonus", 0.80)))
        patch["policy_prior_strength"] = _clamp_float(float(cfg.get("policy_prior_strength", 0.0)) + 0.02, 0.0, 0.55)
        reasons.append("noop_high")

    if patch:
        cfg.update(patch)
        patch["auto_tune_reasons"] = ",".join(reasons)
        patch["auto_tune_noop_rate"] = avg_noop_rate
    return patch


def _weighted_pool_from_counts(counts: Dict[str, int]) -> List[str]:
    pool: List[str] = []
    for name in ("random", "greedy", "starter", "distance"):
        pool.extend([name] * max(0, int(counts.get(name, 0))))
    return pool or ["random", "greedy", "starter", "distance"]


def _auto_tune_opponent_mix(
    cfg: Dict[str, Any],
    eval_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not bool(cfg.get("auto_tune_training", False)) or not eval_history:
        return {}
    if bool(cfg.get("eval_stabilizer_enabled", True)):
        return {}

    current_counts = dict(cfg.get("opponent_mix_counts", {"random": 2, "greedy": 6, "starter": 8, "distance": 4}))
    latest = eval_history[-1]
    by_opp = latest.get("by_opponent", {})
    wr = {name: float(by_opp.get(name, {}).get("winrate", 0.0)) for name in ("random", "greedy", "starter", "distance")}
    noop = float(latest.get("eval_do_nothing_rate", 1.0))

    new_counts = dict(current_counts)
    reasons: List[str] = []

    if wr["random"] >= 0.70 and wr["greedy"] < 0.55:
        new_counts = {"random": 2, "greedy": 8, "starter": 7, "distance": 3}
        reasons.append("focus_greedy")
    elif wr["random"] >= 0.70 and wr["greedy"] >= 0.60 and wr["starter"] < 0.35:
        new_counts = {"random": 2, "greedy": 6, "starter": 9, "distance": 3}
        reasons.append("introduce_starter")
    elif wr["random"] >= 0.75 and wr["greedy"] >= 0.70 and wr["starter"] >= 0.35:
        new_counts = {"random": 2, "greedy": 5, "starter": 8, "distance": 5}
        reasons.append("starter_ramp")
    elif wr["random"] < 0.45:
        new_counts = {"random": 4, "greedy": 7, "starter": 6, "distance": 3}
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


def _eval_stabilizer(
    cfg: Dict[str, Any],
    eval_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if (
        not bool(cfg.get("auto_tune_training", False))
        or not bool(cfg.get("eval_stabilizer_enabled", True))
        or not bool(cfg.get("behavior_supervisor_enabled", True))
    ):
        return {}
    if not eval_history:
        return {}

    patch: Dict[str, Any] = {}
    reasons: List[str] = []
    eval_count = int(cfg.get("stabilizer_eval_count", 0)) + 1
    cfg["stabilizer_eval_count"] = eval_count
    patch["stabilizer_eval_count"] = eval_count

    latest = eval_history[-1]
    latest_raw_noop = float(latest.get("eval_do_nothing_rate", 1.0))
    latest_noop = float(latest.get("eval_legal_noop_rate", latest_raw_noop))
    latest_ships = float(latest.get("eval_avg_ships_sent", 0.0))
    latest_score = _weighted_eval_score(latest)
    latest_by_opp = latest.get("by_opponent", {})
    latest_wr = {name: float(latest_by_opp.get(name, {}).get("winrate", 0.0)) for name in ("random", "greedy", "starter")}
    degenerate_noop = float(cfg.get("degenerate_noop_rate", 0.92))
    degenerate_ships = float(cfg.get("degenerate_max_avg_ships_sent", 1.50))
    # Degenerate-noop exploit detection uses the *raw* rate: an agent that
    # genuinely produces no actions is degenerate regardless of legal/forced.
    if latest_raw_noop >= degenerate_noop and latest_ships <= degenerate_ships:
        patch.update({
            "stabilizer_action": "emergency_activity_reset",
            "stabilizer_reasons": "degenerate_noop_exploit",
            "stabilizer_score": latest_score,
            "stabilizer_noop": latest_noop,
            "stabilizer_ships": latest_ships,
            "stabilizer_passivity": float(latest.get("train_passivity_rate", 1.0)),
            "stabilizer_real_moves": float(latest.get("train_real_moves_per_turn", 0.0)),
            "stabilizer_cooldown_remaining": 0,
            "train_noop_penalty_coef": _clamp_float(float(cfg.get("train_noop_penalty_coef", 0.70)) * 1.30, 0.10, 2.20),
            "train_passivity_penalty_coef": _clamp_float(float(cfg.get("train_passivity_penalty_coef", 0.55)) * 1.20, 0.10, 2.00),
            "do_nothing_logit_penalty": _clamp_float(float(cfg.get("do_nothing_logit_penalty", 0.50)) + 0.35, 0.0, 2.50),
            "train_action_bonus_coef": _clamp_float(float(cfg.get("train_action_bonus_coef", 0.28)) * 1.12, 0.05, 0.70),
            "train_ships_sent_bonus_coef": _clamp_float(
                float(cfg.get("train_ships_sent_bonus_coef", 0.18)) * 1.25,
                0.03,
                float(cfg.get("stabilizer_max_ship_bonus", 0.80)),
            ),
            "min_expand_attack_ships": _clamp_int(
                int(cfg.get("min_expand_attack_ships", 2)) + 1,
                2,
                int(cfg.get("stabilizer_max_min_ships", 8)),
            ),
            "send_ratios": _raise_ratio_floor(
                list(cfg.get("send_ratios", [0.25, 0.35, 0.50, 0.65, 0.80, 0.95])),
                step=float(cfg.get("stabilizer_ratio_floor_step", 0.10)),
                floor_cap=float(cfg.get("stabilizer_ratio_floor_cap", 0.55)),
            ),
        })
        cfg.update(patch)
        return patch

    max_supervised_ships = float(cfg.get("supervisor_max_avg_ships_sent", 24.0))
    if latest_ships >= max_supervised_ships and (latest_wr["starter"] < 0.40 or latest_wr["greedy"] < 0.55):
        patch.update({
            "stabilizer_action": "reduce_ship_volume",
            "stabilizer_reasons": "overcommit_push",
            "stabilizer_score": latest_score,
            "stabilizer_noop": latest_noop,
            "stabilizer_ships": latest_ships,
            "stabilizer_passivity": float(latest.get("train_passivity_rate", 1.0)),
            "stabilizer_real_moves": float(latest.get("train_real_moves_per_turn", 0.0)),
            "stabilizer_cooldown_remaining": 1,
            "train_ships_sent_bonus_coef": _clamp_float(float(cfg.get("train_ships_sent_bonus_coef", 0.18)) * 0.70, 0.03, float(cfg.get("stabilizer_max_ship_bonus", 0.80))),
            "train_action_bonus_coef": _clamp_float(float(cfg.get("train_action_bonus_coef", 0.28)) * 0.94, 0.05, 0.70),
            "min_expand_attack_ships": _clamp_int(int(cfg.get("min_expand_attack_ships", 2)) - 1, 2, int(cfg.get("stabilizer_max_min_ships", 8))),
            "send_ratios": _lower_ratio_ceiling(
                list(cfg.get("send_ratios", [0.25, 0.35, 0.50, 0.65, 0.80, 0.95])),
                step=float(cfg.get("supervisor_ratio_ceiling_step", 0.08)),
                ceiling_floor=float(cfg.get("supervisor_ratio_ceiling_floor", 0.65)),
            ),
        })
        cfg.update(patch)
        return patch

    cooldown = int(cfg.get("stabilizer_cooldown_remaining", 0))
    if cooldown > 0:
        cooldown -= 1
        cfg["stabilizer_cooldown_remaining"] = cooldown
        patch["stabilizer_cooldown_remaining"] = cooldown
        patch["stabilizer_reasons"] = "cooldown"
        patch["stabilizer_action"] = "hold"
        return patch

    window = int(cfg.get("stabilizer_window_evals", 3))
    recent = eval_history[-window:]
    if len(recent) < window:
        patch["stabilizer_reasons"] = f"warming_up_{len(recent)}/{window}"
        patch["stabilizer_action"] = "hold"
        return patch

    score = float(np.mean([_weighted_eval_score(row) for row in recent]))
    # Stabilizer reads legal_noop (avoidable-only) and the legal train rate so
    # decisions are not dictated by forced no-ops.
    noop = float(np.mean([
        row.get("eval_legal_noop_rate", row.get("eval_do_nothing_rate", 1.0)) for row in recent
    ]))
    ships = float(np.mean([row.get("eval_avg_ships_sent", 0.0) for row in recent]))
    passivity = float(np.mean([
        row.get("train_legal_noop_rate", row.get("train_passivity_rate", 1.0)) for row in recent
    ]))
    real_moves = float(np.mean([row.get("train_real_moves_per_turn", 0.0) for row in recent]))
    latest_by_opp = recent[-1].get("by_opponent", {})
    wr = {name: float(latest_by_opp.get(name, {}).get("winrate", 0.0)) for name in ("random", "greedy", "starter")}

    # Targets now refer to legal_noop (avoidable). Defaults moved from raw
    # thresholds (~0.60) to legal thresholds (~0.15) — the raw plancher of
    # forced no-ops is no longer in the signal.
    target_noop = float(cfg.get("stabilizer_target_legal_noop", cfg.get("stabilizer_target_noop", 0.15)))
    target_passivity = float(cfg.get("stabilizer_target_legal_passivity", cfg.get("stabilizer_target_passivity", 0.12)))
    target_ships = float(cfg.get("stabilizer_target_avg_ships_sent", 3.0))
    target_real_moves = float(cfg.get("stabilizer_target_real_moves_turn", 1.20))
    poor_score = score < float(cfg.get("stabilizer_min_weighted_score", 0.55))
    passive_disguised = real_moves < target_real_moves * 0.75 and ships < target_ships * 0.70

    action = "hold"
    if passive_disguised:
        action = "increase_action_availability"
        reasons.append("passive_disguised")
        patch["train_noop_penalty_coef"] = _clamp_float(float(cfg.get("train_noop_penalty_coef", 0.70)) * 1.08, 0.10, 1.80)
        patch["train_passivity_penalty_coef"] = _clamp_float(float(cfg.get("train_passivity_penalty_coef", 0.55)) * 1.08, 0.10, 1.80)
        patch["do_nothing_logit_penalty"] = _clamp_float(float(cfg.get("do_nothing_logit_penalty", 0.50)) + 0.12, 0.0, 2.20)
        patch["train_action_bonus_coef"] = _clamp_float(float(cfg.get("train_action_bonus_coef", 0.28)) * 1.08, 0.05, 0.65)
        patch["min_expand_attack_ships"] = _clamp_int(int(cfg.get("min_expand_attack_ships", 2)) - 1, 2, 6)
    elif ships < target_ships and noop <= target_noop + 0.20:
        action = "increase_ship_volume"
        reasons.append("ships_low")
        patch["train_ships_sent_bonus_coef"] = _clamp_float(
            float(cfg.get("train_ships_sent_bonus_coef", 0.18)) * 1.18,
            0.03,
            float(cfg.get("stabilizer_max_ship_bonus", 0.80)),
        )
        if real_moves >= target_real_moves * 0.90 and passivity <= target_passivity + 0.08:
            patch["min_expand_attack_ships"] = _clamp_int(
                int(cfg.get("min_expand_attack_ships", 2)) + 1,
                2,
                int(cfg.get("stabilizer_max_min_ships", 6)),
            )
        patch["send_ratios"] = _raise_ratio_floor(
            list(cfg.get("send_ratios", [0.25, 0.35, 0.50, 0.65, 0.80, 0.95])),
            step=float(cfg.get("stabilizer_ratio_floor_step", 0.10)),
            floor_cap=float(cfg.get("stabilizer_ratio_floor_cap", 0.55)),
        )
    elif noop > target_noop and passivity > target_passivity and ships >= max(2.5, target_ships * 0.80):
        action = "reduce_noop"
        reasons.append("noop_high")
        patch["train_noop_penalty_coef"] = _clamp_float(float(cfg.get("train_noop_penalty_coef", 0.70)) * 1.03, 0.10, 1.40)
        patch["do_nothing_logit_penalty"] = _clamp_float(float(cfg.get("do_nothing_logit_penalty", 0.50)) + 0.05, 0.0, 2.0)
        patch["train_action_bonus_coef"] = _clamp_float(float(cfg.get("train_action_bonus_coef", 0.28)) * 1.02, 0.05, 0.55)
    elif poor_score and passivity < target_passivity - 0.05 and ships < target_ships:
        action = "relax_overcorrection"
        reasons.append("active_but_weak")
        patch["train_passivity_penalty_coef"] = _clamp_float(float(cfg.get("train_passivity_penalty_coef", 0.55)) * 0.94, 0.10, 1.60)
        patch["train_action_bonus_coef"] = _clamp_float(float(cfg.get("train_action_bonus_coef", 0.28)) * 0.96, 0.05, 0.55)
        patch["do_nothing_logit_penalty"] = _clamp_float(float(cfg.get("do_nothing_logit_penalty", 0.50)) - 0.05, 0.0, 2.0)
    elif real_moves < target_real_moves and noop > target_noop:
        action = "increase_action_availability"
        reasons.append("moves_low_noop_high")
        patch["min_expand_attack_ships"] = _clamp_int(int(cfg.get("min_expand_attack_ships", 2)) - 1, 2, 6)
        patch["train_action_bonus_coef"] = _clamp_float(float(cfg.get("train_action_bonus_coef", 0.28)) * 1.03, 0.05, 0.55)
    elif wr["random"] >= 0.65 and wr["greedy"] >= 0.60 and wr["starter"] < 0.34:
        action = "starter_focus"
        reasons.append("starter_low")
        counts = dict(cfg.get("opponent_mix_counts", {"random": 2, "greedy": 6, "starter": 8, "distance": 4}))
        counts["starter"] = min(int(counts.get("starter", 1)) + 2, int(cfg.get("stabilizer_max_starter_count", 5)))
        if int(counts.get("greedy", 0)) > 8:
            counts["greedy"] = int(counts.get("greedy", 0)) - 1
        if int(counts.get("random", 0)) > 3:
            counts["random"] = int(counts.get("random", 0)) - 1
        patch["opponent_mix_counts"] = {str(k): int(v) for k, v in counts.items()}
        patch["stage1_pool"] = _weighted_pool_from_counts(patch["opponent_mix_counts"])

    if action == "hold":
        patch["stabilizer_reasons"] = "constraints_ok_or_waiting"
        patch["stabilizer_action"] = "hold"
        patch["stabilizer_score"] = score
        patch["stabilizer_noop"] = noop
        patch["stabilizer_ships"] = ships
        patch["stabilizer_passivity"] = passivity
        patch["stabilizer_real_moves"] = real_moves
        return patch

    cooldown = int(cfg.get("stabilizer_cooldown_evals", 2))
    patch["stabilizer_cooldown_remaining"] = cooldown
    patch["stabilizer_reasons"] = ",".join(reasons)
    patch["stabilizer_action"] = action
    patch["stabilizer_score"] = score
    patch["stabilizer_noop"] = noop
    patch["stabilizer_ships"] = ships
    patch["stabilizer_passivity"] = passivity
    patch["stabilizer_real_moves"] = real_moves
    cfg.update(patch)
    if "opponent_mix_counts" in patch:
        cfg["stage1_pool"] = list(patch["stage1_pool"])
        if cfg.get("curriculum_tiers"):
            cfg["curriculum_tiers"][0]["opponents"] = list(patch["stage1_pool"])
    return patch


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


def _parse_float_list(raw_value: str, fallback: List[float]) -> List[float]:
    values: List[float] = []
    for item in str(raw_value).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if 0.0 < value < 1.0:
            values.append(value)
    return values or list(fallback)


def _build_config(args: argparse.Namespace) -> Dict[str, Any]:
    default_opponents = ["random", "greedy", "starter", "distance"]
    simple_opponents = _parse_opponent_list(args.simple_opponents, default_opponents)
    eval_opponents = _parse_opponent_list(args.eval_opponents, list(dict.fromkeys(simple_opponents)))
    opponent_mix_counts = {name: simple_opponents.count(name) for name in ("random", "greedy", "starter", "distance")}
    return {
        # Game
        "game_engine": "official_fast",
        "official_fast_c_accel": True,
        "max_turns": 100,
        "max_actions_per_turn": args.max_actions_per_turn,
        "min_expand_attack_ships": args.min_expand_attack_ships,
        "send_ratios": _parse_float_list(args.send_ratios, [0.25, 0.35, 0.50, 0.65, 0.80, 0.95]),
        "allow_support_actions": not bool(args.disable_support_actions),
        "policy_prior_strength": args.policy_prior_strength,
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
        "train_return_gamma": args.train_return_gamma,
        "train_return_clip": args.train_return_clip,
        "value_loss_coef": 0.25,
        "entropy_coef_start": args.entropy_coef_start,
        "baseline_momentum": 0.10,
        "max_grad_norm": 1.0,
        # PPO
        "ppo_clip_eps": args.ppo_clip_eps,
        "ppo_epochs": args.ppo_epochs,
        "ppo_minibatch_size": args.ppo_minibatch_size,
        # Activity shaping
        "dense_reward_enabled": True,
        "dense_planet_coef": args.dense_planet_coef,
        "dense_production_coef": args.dense_production_coef,
        "dense_ship_share_coef": args.dense_ship_share_coef,
        "dense_score_coef": args.dense_score_coef,
        "dense_survival_coef": 0.05,
        "dense_reward_clip": args.dense_reward_clip,
        "train_target_do_nothing_rate": 0.35,  # raw (legacy), kept for fallback
        "train_target_legal_noop_rate": args.train_target_legal_noop_rate,
        "train_noop_penalty_coef": 0.70,
        "train_passivity_penalty_coef": 0.30,
        # Per-step shaping (replaces the broadcast episodic passivity penalty)
        "train_episodic_passivity_penalty_enabled": bool(args.enable_episodic_passivity_penalty),
        "train_per_step_legal_noop_penalty": args.per_step_legal_noop_penalty,
        "train_per_step_real_action_bonus": args.per_step_real_action_bonus,
        "train_per_step_ship_volume_bonus": args.per_step_ship_volume_bonus,
        "train_per_step_ship_volume_target": args.per_step_ship_volume_target,
        "train_per_step_shape_clip": args.per_step_shape_clip,
        "train_event_shaping_enabled": not bool(args.disable_event_shaping),
        "train_event_hit_bonus": args.event_hit_bonus,
        "train_event_enemy_hit_bonus": args.event_enemy_hit_bonus,
        "train_event_capture_bonus": args.event_capture_bonus,
        "train_event_support_bonus": args.event_support_bonus,
        "train_event_lost_penalty": args.event_lost_penalty,
        "train_event_pending_penalty": args.event_pending_penalty,
        "train_event_min_shape_clip": args.event_min_shape_clip,
        "train_event_max_flat_action_bonus": args.event_max_flat_action_bonus,
        "train_event_max_ship_volume_bonus": args.event_max_ship_volume_bonus,
        "train_event_max_activity_action_bonus": args.event_max_activity_action_bonus,
        "train_event_max_activity_ships_bonus": args.event_max_activity_ships_bonus,
        "train_passive_win_legal_noop_rate": args.passive_win_legal_noop_rate,
        "do_nothing_logit_penalty": 0.50,
        "do_nothing_prob_cap": 0.45,
        "do_nothing_prob_caps_by_slot": [0.05, 0.08, 0.12, 0.20, 0.32, 0.45, 0.55, 0.65],
        "train_action_bonus_coef": 0.03,
        "train_ships_sent_bonus_coef": 0.0,
        "train_activity_reward_clip": 0.55,
        "train_passive_win_noop_rate": args.valid_win_max_do_nothing_rate,
        "train_passive_win_min_avg_ships_sent": args.valid_win_min_avg_ships_sent,
        "train_passive_win_min_real_moves_turn": args.valid_win_min_real_moves_turn,
        "train_passive_win_terminal_reward": args.passive_win_terminal_reward,
        "deterministic_avoid_noop_if_real": True,
        "train_mission_mix_bonus_coef": args.train_mission_mix_bonus_coef,
        "train_target_support_ratio": args.train_target_support_ratio,
        "train_support_ratio_band": args.train_support_ratio_band,
        "train_min_support_ratio": args.train_min_support_ratio,
        "train_max_attack_ratio": args.train_max_attack_ratio,
        "train_mission_mix_reward_clip": args.train_mission_mix_reward_clip,
        # Slow eval-level stabilizer. It replaces fast train-step behavior
        # shaping when auto-tune is enabled.
        "eval_stabilizer_enabled": True,
        "behavior_supervisor_enabled": not bool(args.disable_behavior_supervisor),
        "stabilizer_window_evals": 2,
        "stabilizer_cooldown_evals": 1,
        "stabilizer_target_noop": args.stabilizer_target_noop,
        "stabilizer_target_passivity": args.stabilizer_target_passivity,
        "stabilizer_target_real_moves_turn": args.stabilizer_target_real_moves_turn,
        "stabilizer_target_avg_ships_sent": args.stabilizer_target_avg_ships_sent,
        "stabilizer_min_weighted_score": args.stabilizer_min_weighted_score,
        "stabilizer_max_ship_bonus": args.stabilizer_max_ship_bonus,
        "stabilizer_max_min_ships": args.stabilizer_max_min_ships,
        "stabilizer_max_starter_count": args.stabilizer_max_starter_count,
        "stabilizer_ratio_floor_step": args.stabilizer_ratio_floor_step,
        "stabilizer_ratio_floor_cap": args.stabilizer_ratio_floor_cap,
        "supervisor_max_avg_ships_sent": args.supervisor_max_avg_ships_sent,
        "supervisor_ratio_ceiling_step": args.supervisor_ratio_ceiling_step,
        "supervisor_ratio_ceiling_floor": args.supervisor_ratio_ceiling_floor,
        "stabilizer_eval_count": 0,
        "stabilizer_cooldown_remaining": 0,
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
        "best_eval_every": args.best_eval_every,
        "max_batch_size": args.batch_size,
        "batch_timeout": args.batch_timeout,
        "device": args.device,
        # Run
        "duration_minutes": args.duration_minutes,
        "target_winrate": args.target_winrate,
        "n_players": args.n_players,
        "seed": 42,
        "eval_seed_start": args.eval_seed_start,
        "eval_seed_stride": args.eval_seed_stride,
        "promotion_margin": args.promotion_margin,
        "rollback_margin": args.rollback_margin,
        "rollback_on_noop_rate": args.rollback_on_noop_rate,
        "min_lr": args.min_lr,
        "max_lr": args.max_lr,
        "promotion_lr_mult": args.promotion_lr_mult,
        "rollback_lr_mult": args.rollback_lr_mult,
        "max_eval_do_nothing_rate": args.max_eval_do_nothing_rate,
        "max_eval_legal_noop_rate": args.max_eval_legal_noop_rate,
        "valid_win_max_do_nothing_rate": args.valid_win_max_do_nothing_rate,
        "valid_win_max_legal_noop_rate": args.valid_win_max_legal_noop_rate,
        "rollback_on_legal_noop_rate": args.rollback_on_legal_noop_rate,
        "stabilizer_target_legal_noop": args.stabilizer_target_legal_noop,
        "stabilizer_target_legal_passivity": args.stabilizer_target_legal_passivity,
        "valid_win_min_avg_ships_sent": args.valid_win_min_avg_ships_sent,
        "valid_win_min_real_moves_turn": args.valid_win_min_real_moves_turn,
        "min_eval_avg_ships_sent": args.min_eval_avg_ships_sent,
        "degenerate_noop_rate": args.degenerate_noop_rate,
        "degenerate_max_avg_ships_sent": args.degenerate_max_avg_ships_sent,
        "degenerate_min_winrate": args.degenerate_min_winrate,
        "collapse_stop_evals": args.collapse_stop_evals,
        "collapse_stop_noop_rate": args.collapse_stop_noop_rate,
        "collapse_stop_max_avg_ships_sent": args.collapse_stop_max_avg_ships_sent,
        "collapse_max_recoveries": args.collapse_max_recoveries,
        "max_consecutive_regressions": args.max_consecutive_regressions,
        "max_opponent_regression": args.max_opponent_regression,
        "min_ci_promotion_games": args.min_ci_promotion_games,
        "auto_tune_training": bool(args.auto_tune_training),
        "teacher_kl_coef": args.teacher_kl_coef,
        "teacher_checkpoint": args.teacher_checkpoint,
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
    from neural_network.src.notebook_4p_training import _build_agents, run_match

    model.eval()
    eval_cfg = dict(config)
    eval_cfg["temperature_end"] = 0.0
    eval_cfg["temperature_start"] = 0.0
    eval_cfg["policy_prior_strength"] = float(config.get("policy_prior_strength", 0.0))

    opponents = list(pool) or ["random"]
    per_opponent: Dict[str, Any] = {}
    all_wins: List[float] = []
    all_valid_wins: List[float] = []
    all_do_nothing_rates: List[float] = []
    all_legal_noop_rates: List[float] = []
    all_forced_noop_rates: List[float] = []
    all_avg_ships_sent: List[float] = []
    all_fleet_hit_rates: List[float] = []
    all_fleet_capture_rates: List[float] = []
    all_fleet_lost_rates: List[float] = []
    all_fleet_pending_rates: List[float] = []
    train_metrics = train_metrics or {}
    valid_win_max_noop = float(config.get("valid_win_max_do_nothing_rate", config.get("max_eval_do_nothing_rate", 0.75)))
    valid_win_max_legal_noop = float(config.get("valid_win_max_legal_noop_rate", 0.30))
    valid_win_min_ships = float(config.get("valid_win_min_avg_ships_sent", config.get("min_eval_avg_ships_sent", 0.0)))
    valid_win_min_real_moves = float(config.get("valid_win_min_real_moves_turn", 0.0))

    for opponent_idx, opponent in enumerate(opponents):
        if progress_log_path is not None:
            _log(progress_log_path, f"eval_start checkpoint={checkpoint_label} opponent={opponent} games={episodes}")
        wins: List[float] = []
        valid_wins: List[float] = []
        do_nothing_rates: List[float] = []
        legal_noop_rates: List[float] = []
        forced_noop_rates: List[float] = []
        avg_ships_sent: List[float] = []
        fleet_hit_rates: List[float] = []
        fleet_capture_rates: List[float] = []
        fleet_lost_rates: List[float] = []
        fleet_pending_rates: List[float] = []
        metric_rows: List[Dict[str, float]] = []
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
            episode_steps = max(1, int(result.get("steps", 1)))
            metrics = summarize_action_records(action_records)
            metrics.update(summarize_fleet_events(result.get("fleet_events", []) or [], our_index))
            metric_rows.append(dict(metrics))
            noop = float(metrics.get("do_nothing_rate", 1.0))
            legal_noop = float(metrics.get("legal_noop_rate", noop))
            forced_noop = float(metrics.get("forced_noop_rate", 0.0))
            ships = float(metrics.get("avg_ships_sent", 0.0))
            fleet_hit = float(metrics.get("fleet_hit_rate", 0.0))
            fleet_capture = float(metrics.get("fleet_capture_rate", 0.0))
            fleet_lost = float(metrics.get("fleet_lost_rate", 0.0))
            fleet_pending = float(metrics.get("fleet_pending_rate", 0.0))
            real_moves_per_turn = float(metrics.get("real_action_count", 0.0)) / episode_steps
            won = float(winner == our_index)
            # Promotion-validity gate uses legal_noop (avoidable-only) so wins on
            # idle-by-necessity states are not penalised.
            valid_won = float(
                bool(won)
                and legal_noop <= valid_win_max_legal_noop
                and noop <= valid_win_max_noop
                and ships >= valid_win_min_ships
                and (valid_win_min_real_moves <= 0.0 or real_moves_per_turn >= valid_win_min_real_moves)
            )
            wins.append(won)
            valid_wins.append(valid_won)
            do_nothing_rates.append(noop)
            legal_noop_rates.append(legal_noop)
            forced_noop_rates.append(forced_noop)
            avg_ships_sent.append(ships)
            fleet_hit_rates.append(fleet_hit)
            fleet_capture_rates.append(fleet_capture)
            fleet_lost_rates.append(fleet_lost)
            fleet_pending_rates.append(fleet_pending)
            if progress_log_path is not None and ((i + 1) == episodes or (i + 1) % max(1, episodes // 4) == 0):
                _log(
                    progress_log_path,
                    f"eval_progress checkpoint={checkpoint_label} opponent={opponent} "
                    f"games={i + 1}/{episodes} wins={int(sum(wins))}",
                )

        win_count = int(sum(wins))
        valid_win_count = int(sum(valid_wins))
        ci_low, ci_high = _wilson_ci(win_count, episodes)
        valid_ci_low, valid_ci_high = _wilson_ci(valid_win_count, episodes)

        def _metric_mean(name: str, default: float = 0.0) -> float:
            values = [float(row.get(name, default)) for row in metric_rows if name in row]
            return float(np.mean(values)) if values else default

        def _metric_max(name: str, default: float = 0.0) -> float:
            values = [float(row.get(name, default)) for row in metric_rows if name in row]
            return float(np.max(values)) if values else default

        record = {
            "episode": int(episode_count),
            "checkpoint": checkpoint_label,
            "opponent": str(opponent),
            "games": int(episodes),
            "wins": win_count,
            "valid_wins": valid_win_count,
            "losses": int(episodes - win_count),
            "winrate": float(np.mean(wins)) if wins else 0.0,
            "valid_winrate": float(np.mean(valid_wins)) if valid_wins else 0.0,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "valid_ci_low": valid_ci_low,
            "valid_ci_high": valid_ci_high,
            "do_nothing_rate": float(np.mean(do_nothing_rates)) if do_nothing_rates else 1.0,
            "legal_noop_rate": float(np.mean(legal_noop_rates)) if legal_noop_rates else 0.0,
            "forced_noop_rate": float(np.mean(forced_noop_rates)) if forced_noop_rates else 0.0,
            "avg_ships_sent": float(np.mean(avg_ships_sent)) if avg_ships_sent else 0.0,
            "fleet_hit_rate": float(np.mean(fleet_hit_rates)) if fleet_hit_rates else 0.0,
            "fleet_capture_rate": float(np.mean(fleet_capture_rates)) if fleet_capture_rates else 0.0,
            "fleet_lost_rate": float(np.mean(fleet_lost_rates)) if fleet_lost_rates else 0.0,
            "fleet_pending_rate": float(np.mean(fleet_pending_rates)) if fleet_pending_rates else 0.0,
            "avg_ships_per_action": _metric_mean("avg_ships_per_action"),
            "median_ships_sent": _metric_mean("median_ships_sent"),
            "ships_sent_p25": _metric_mean("ships_sent_p25"),
            "ships_sent_p75": _metric_mean("ships_sent_p75"),
            "ships_sent_p90": _metric_mean("ships_sent_p90"),
            "ships_sent_max": _metric_max("ships_sent_max"),
            "mission_expand_ships_mean": _metric_mean("mission_expand_ships_mean"),
            "mission_attack_ships_mean": _metric_mean("mission_attack_ships_mean"),
            "mission_support_ships_mean": _metric_mean("mission_support_ships_mean"),
            "tactical_support_defense_rate": _metric_mean("tactical_support_defense_rate"),
            "tactical_support_front_rate": _metric_mean("tactical_support_front_rate"),
            "tactical_support_redistribute_rate": _metric_mean("tactical_support_redistribute_rate"),
            "tactical_support_passive_rate": _metric_mean("tactical_support_passive_rate"),
            "tactical_support_backward_rate": _metric_mean("tactical_support_backward_rate"),
            "tactical_attack_convert_rate": _metric_mean("tactical_attack_convert_rate"),
            "tactical_attack_opportunity_rate": _metric_mean("tactical_attack_opportunity_rate"),
            "tactical_attack_pressure_rate": _metric_mean("tactical_attack_pressure_rate"),
            "candidate_attack_convert_rate": _metric_mean("candidate_attack_convert_rate"),
            "candidate_attack_pressure_rate": _metric_mean("candidate_attack_pressure_rate"),
            "attack_convert_missed_rate": _metric_mean("attack_convert_missed_rate"),
            "good_attack_missed_rate": _metric_mean("good_attack_missed_rate"),
            "tactical_expand_front_rate": _metric_mean("tactical_expand_front_rate"),
            "tactical_expand_safe_rate": _metric_mean("tactical_expand_safe_rate"),
            "fleet_hit_rate": _metric_mean("fleet_hit_rate"),
            "fleet_capture_rate": _metric_mean("fleet_capture_rate"),
            "fleet_lost_rate": _metric_mean("fleet_lost_rate"),
            "fleet_pending_rate": _metric_mean("fleet_pending_rate"),
            "slot0_ships_mean": _metric_mean("slot0_ships_mean"),
            "slot1_ships_mean": _metric_mean("slot1_ships_mean"),
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
                f"winrate={record['winrate']:.3f} valid={record['valid_winrate']:.3f} "
                f"noop={record['do_nothing_rate']:.3f} ships={record['avg_ships_sent']:.2f} "
                f"fleet_hit={record.get('fleet_hit_rate', 0.0):.3f} "
                f"fleet_capture={record.get('fleet_capture_rate', 0.0):.3f} "
                f"fleet_lost={record.get('fleet_lost_rate', 0.0):.3f} "
                f"sup_def={record.get('tactical_support_defense_rate', 0.0):.3f} "
                f"sup_front={record.get('tactical_support_front_rate', 0.0):.3f} "
                f"sup_passive={record.get('tactical_support_passive_rate', 0.0):.3f} "
                f"atk_conv={record.get('tactical_attack_convert_rate', 0.0):.3f} "
                f"atk_press={record.get('tactical_attack_pressure_rate', 0.0):.3f} "
                f"atk_miss={record.get('attack_convert_missed_rate', 0.0):.3f} "
                f"ships_p50={record.get('median_ships_sent', 0.0):.2f} "
                f"ships_p75={record.get('ships_sent_p75', 0.0):.2f} "
                f"ships_p90={record.get('ships_sent_p90', 0.0):.2f} "
                f"ships_max={record.get('ships_sent_max', 0.0):.2f}",
                )
        if eval_history_path is not None:
            append_jsonl(eval_history_path, record)
        all_wins.extend(wins)
        all_valid_wins.extend(valid_wins)
        all_do_nothing_rates.extend(do_nothing_rates)
        all_legal_noop_rates.extend(legal_noop_rates)
        all_forced_noop_rates.extend(forced_noop_rates)
        all_avg_ships_sent.extend(avg_ships_sent)
        all_fleet_hit_rates.extend(fleet_hit_rates)
        all_fleet_capture_rates.extend(fleet_capture_rates)
        all_fleet_lost_rates.extend(fleet_lost_rates)
        all_fleet_pending_rates.extend(fleet_pending_rates)

    total_games = len(all_wins)
    total_wins = int(sum(all_wins))
    total_valid_wins = int(sum(all_valid_wins))
    ci_low, ci_high = _wilson_ci(total_wins, total_games)
    valid_ci_low, valid_ci_high = _wilson_ci(total_valid_wins, total_games)
    aggregate = {
        "winrate": float(np.mean(all_wins)) if all_wins else 0.0,
        "valid_winrate": float(np.mean(all_valid_wins)) if all_valid_wins else 0.0,
        "wins": total_wins,
        "valid_wins": total_valid_wins,
        "losses": int(total_games - total_wins),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "valid_ci_low": valid_ci_low,
        "valid_ci_high": valid_ci_high,
        "eval_do_nothing_rate": float(np.mean(all_do_nothing_rates)) if all_do_nothing_rates else 1.0,
        "eval_legal_noop_rate": float(np.mean(all_legal_noop_rates)) if all_legal_noop_rates else 0.0,
        "eval_forced_noop_rate": float(np.mean(all_forced_noop_rates)) if all_forced_noop_rates else 0.0,
        "eval_avg_ships_sent": float(np.mean(all_avg_ships_sent)) if all_avg_ships_sent else 0.0,
        "eval_fleet_hit_rate": float(np.mean(all_fleet_hit_rates)) if all_fleet_hit_rates else 0.0,
        "eval_fleet_capture_rate": float(np.mean(all_fleet_capture_rates)) if all_fleet_capture_rates else 0.0,
        "eval_fleet_lost_rate": float(np.mean(all_fleet_lost_rates)) if all_fleet_lost_rates else 0.0,
        "eval_fleet_pending_rate": float(np.mean(all_fleet_pending_rates)) if all_fleet_pending_rates else 0.0,
        "eval_episodes": int(episodes),
        "eval_games": int(total_games),
        "seed_start": int(seed_start),
        "seed_count": int(episodes),
        "by_opponent": per_opponent,
        "train_passivity_rate": float(train_metrics.get("do_nothing_rate_mean", 1.0)),
        "train_legal_noop_rate": float(train_metrics.get("legal_noop_rate_mean", train_metrics.get("do_nothing_rate_mean", 1.0))),
        "train_forced_noop_rate": float(train_metrics.get("forced_noop_rate_mean", 0.0)),
        "train_real_moves_per_turn": float(train_metrics.get("mean_real_actions_per_turn", 0.0)),
        "train_real_moves_per_game": float(train_metrics.get("mean_real_actions_per_game", 0.0)),
    }
    aggregate["weighted_score"] = _weighted_eval_score(aggregate)
    return {
        **aggregate,
    }


def _promotion_decision(
    candidate_eval: Dict[str, Any],
    best_eval: Dict[str, Any],
    config: Dict[str, Any],
) -> tuple[bool, List[str], bool]:
    margin = float(config.get("promotion_margin", 0.03))
    rollback_margin = float(config.get("rollback_margin", 0.08))
    max_do_nothing = float(config.get("max_eval_do_nothing_rate", 0.75))
    max_legal_noop = float(config.get("max_eval_legal_noop_rate", 0.30))
    rollback_noop_rate = float(config.get("rollback_on_noop_rate", 0.0))
    rollback_legal_noop_rate = float(config.get("rollback_on_legal_noop_rate", 0.0))
    min_avg_ships = float(config.get("min_eval_avg_ships_sent", 0.0))
    degenerate_noop = float(config.get("degenerate_noop_rate", 0.92))
    degenerate_ships = float(config.get("degenerate_max_avg_ships_sent", 1.50))
    degenerate_winrate = float(config.get("degenerate_min_winrate", 0.70))
    max_opp_regression = float(config.get("max_opponent_regression", 0.12))
    min_ci_games = int(config.get("min_ci_promotion_games", 96))
    reasons: List[str] = []

    cand_wr = float(candidate_eval.get("valid_winrate", candidate_eval.get("winrate", 0.0)))
    best_wr = float(best_eval.get("valid_winrate", best_eval.get("winrate", 0.0)))
    delta = cand_wr - best_wr
    promote = True
    if delta < margin:
        promote = False
        reasons.append(f"delta {delta:.4f} < margin {margin:.4f}")
    eval_games = int(candidate_eval.get("eval_games", 0))
    cand_ci_low = float(candidate_eval.get("valid_ci_low", candidate_eval.get("ci_low", 0.0)))
    if eval_games >= min_ci_games and cand_ci_low <= best_wr:
        promote = False
        reasons.append(
            f"candidate valid_ci_low {cand_ci_low:.4f} <= best_valid_winrate {best_wr:.4f}"
        )
    eval_noop_rate = float(candidate_eval.get("eval_do_nothing_rate", 1.0))
    eval_legal_noop = float(candidate_eval.get("eval_legal_noop_rate", eval_noop_rate))
    if eval_noop_rate > max_do_nothing:
        promote = False
        reasons.append("do_nothing gate failed")
    if eval_legal_noop > max_legal_noop:
        promote = False
        reasons.append(f"legal_noop gate failed {eval_legal_noop:.3f} > {max_legal_noop:.3f}")
    if float(candidate_eval.get("eval_avg_ships_sent", 0.0)) < min_avg_ships:
        promote = False
        reasons.append("avg_ships_sent gate failed")

    rollback = delta <= -rollback_margin
    if cand_wr >= degenerate_winrate and eval_noop_rate >= degenerate_noop and float(candidate_eval.get("eval_avg_ships_sent", 0.0)) <= degenerate_ships:
        promote = False
        rollback = True
        reasons.append(
            f"degenerate noop exploit winrate={cand_wr:.4f} noop={eval_noop_rate:.4f} ships={float(candidate_eval.get('eval_avg_ships_sent', 0.0)):.2f}"
        )
    if rollback_noop_rate > 0.0 and eval_noop_rate > rollback_noop_rate:
        rollback = True
        reasons.append(f"noop rollback {eval_noop_rate:.4f} > {rollback_noop_rate:.4f}")
    if rollback_legal_noop_rate > 0.0 and eval_legal_noop > rollback_legal_noop_rate:
        rollback = True
        reasons.append(f"legal_noop rollback {eval_legal_noop:.4f} > {rollback_legal_noop_rate:.4f}")
    cand_by_opp = candidate_eval.get("by_opponent", {})
    best_by_opp = best_eval.get("by_opponent", {})
    for opponent, cand_record in cand_by_opp.items():
        best_record = best_by_opp.get(opponent)
        if not best_record:
            continue
        opp_delta = (
            float(cand_record.get("valid_winrate", cand_record.get("winrate", 0.0)))
            - float(best_record.get("valid_winrate", best_record.get("winrate", 0.0)))
        )
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
    parser.add_argument(
        "--best-eval-every",
        type=int,
        default=1,
        help="Re-evaluate best_validated every N eval rounds. Cached metadata is used between full best evals.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.00006)
    parser.add_argument("--ppo-clip-eps", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--ppo-minibatch-size", type=int, default=512)
    parser.add_argument("--train-return-gamma", type=float, default=0.997)
    parser.add_argument("--train-return-clip", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch-timeout", type=float, default=0.010)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-winrate", type=float, default=0.85)
    parser.add_argument("--n-players", type=int, default=2)
    parser.add_argument("--max-actions-per-turn", type=int, default=8)
    parser.add_argument("--min-expand-attack-ships", type=int, default=2)
    parser.add_argument("--send-ratios", default="0.25,0.35,0.50,0.65,0.80,0.95")
    parser.add_argument("--eval-seed-start", type=int, default=900000)
    parser.add_argument("--eval-seed-stride", type=int, default=1000000)
    parser.add_argument("--promotion-margin", type=float, default=0.03)
    parser.add_argument("--rollback-margin", type=float, default=0.08)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--max-lr", type=float, default=2e-4)
    parser.add_argument("--promotion-lr-mult", type=float, default=1.05)
    parser.add_argument("--rollback-lr-mult", type=float, default=0.5)
    parser.add_argument("--max-eval-do-nothing-rate", type=float, default=0.75)
    parser.add_argument("--max-eval-legal-noop-rate", type=float, default=0.30,
                        help="Promotion gate: max avoidable-noop rate. The raw gate is now a backstop.")
    parser.add_argument("--rollback-on-noop-rate", type=float, default=0.0)
    parser.add_argument("--rollback-on-legal-noop-rate", type=float, default=0.0,
                        help="Hard rollback when avoidable-noop rate exceeds this. 0 = disabled.")
    parser.add_argument("--train-target-legal-noop-rate", type=float, default=0.10,
                        help="Auto-tune / stabilizer target for the avoidable-noop rate (legal). Defaults moved from 0.35 raw to 0.10 legal because forced no-ops are removed from the metric.")
    parser.add_argument("--passive-win-legal-noop-rate", type=float, default=0.30,
                        help="Wins flagged as passive when legal_noop_rate exceeds this threshold.")
    parser.add_argument("--per-step-legal-noop-penalty", type=float, default=0.020,
                        help="Per-step penalty applied to each avoidable no-op decision.")
    parser.add_argument("--per-step-real-action-bonus", type=float, default=0.001,
                        help="Small exploration bonus for real actions. Event bonuses carry the main signal.")
    parser.add_argument("--per-step-ship-volume-bonus", type=float, default=0.0,
                        help="Extra per-step bonus proportional to ships sent (linear up to target, capped). 0=disabled.")
    parser.add_argument("--per-step-ship-volume-target", type=float, default=8.0,
                        help="Ship count at which the volume bonus is maxed out.")
    parser.add_argument("--per-step-shape-clip", type=float, default=0.04,
                        help="Hard clip on per-step shaping magnitude.")
    parser.add_argument("--disable-event-shaping", action="store_true",
                        help="Disable causal fleet-event shaping and use legacy action-only shaping.")
    parser.add_argument("--event-hit-bonus", type=float, default=0.035)
    parser.add_argument("--event-enemy-hit-bonus", type=float, default=0.045)
    parser.add_argument("--event-capture-bonus", type=float, default=0.10)
    parser.add_argument("--event-support-bonus", type=float, default=0.015)
    parser.add_argument("--event-lost-penalty", type=float, default=0.045)
    parser.add_argument("--event-pending-penalty", type=float, default=0.0)
    parser.add_argument("--event-min-shape-clip", type=float, default=0.12)
    parser.add_argument("--event-max-flat-action-bonus", type=float, default=0.002)
    parser.add_argument("--event-max-ship-volume-bonus", type=float, default=0.0)
    parser.add_argument("--event-max-activity-action-bonus", type=float, default=0.03)
    parser.add_argument("--event-max-activity-ships-bonus", type=float, default=0.0)
    parser.add_argument("--enable-episodic-passivity-penalty", action="store_true",
                        help="Re-enable the legacy episode-broadcast passivity penalty (default: off, per-step shaping replaces it).")
    parser.add_argument("--stabilizer-target-legal-noop", type=float, default=0.15,
                        help="Stabilizer target for eval legal_noop_rate.")
    parser.add_argument("--stabilizer-target-legal-passivity", type=float, default=0.12,
                        help="Stabilizer target for train legal_noop_rate.")
    parser.add_argument("--valid-win-max-legal-noop-rate", type=float, default=0.30,
                        help="Validity gate for promotion: max avoidable-noop rate per game.")
    parser.add_argument("--reset-shaping-coefs", action="store_true",
                        help="On resume, ignore saturated shaping coefficients from adaptive_config and start fresh from CLI defaults.")
    parser.add_argument("--min-eval-avg-ships-sent", type=float, default=0.0)
    parser.add_argument("--valid-win-max-do-nothing-rate", type=float, default=0.84)
    parser.add_argument("--valid-win-min-avg-ships-sent", type=float, default=4.0)
    parser.add_argument("--valid-win-min-real-moves-turn", type=float, default=1.0)
    parser.add_argument("--passive-win-terminal-reward", type=float, default=-0.25)
    parser.add_argument("--degenerate-noop-rate", type=float, default=0.92)
    parser.add_argument("--degenerate-max-avg-ships-sent", type=float, default=1.50)
    parser.add_argument("--degenerate-min-winrate", type=float, default=0.70)
    parser.add_argument("--collapse-stop-evals", type=int, default=2)
    parser.add_argument("--collapse-stop-noop-rate", type=float, default=0.92)
    parser.add_argument("--collapse-stop-max-avg-ships-sent", type=float, default=2.0)
    parser.add_argument("--collapse-max-recoveries", type=int, default=0,
                        help="Max collapse recoveries before hard stop. 0=unlimited recovery.")
    parser.add_argument("--max-consecutive-regressions", type=int, default=0,
                        help="Trigger STUCK_RECOVERY after N consecutive rollbacks. 0=disabled.")
    parser.add_argument("--max-opponent-regression", type=float, default=0.12)
    parser.add_argument("--min-ci-promotion-games", type=int, default=96)
    parser.add_argument("--league-archive-size", type=int, default=4)
    parser.add_argument("--simple-opponents", default="random,greedy,starter,distance")
    parser.add_argument("--eval-opponents", default="")
    parser.add_argument("--disable-support-actions", action="store_true")
    parser.add_argument("--auto-tune-training", action="store_true")
    parser.add_argument("--policy-prior-strength", type=float, default=0.20)
    parser.add_argument("--temperature-decay-updates", type=int, default=200)
    parser.add_argument("--teacher-checkpoint", default="", help="Optional frozen teacher checkpoint for KL distillation")
    parser.add_argument("--teacher-kl-coef", type=float, default=0.0)
    parser.add_argument("--entropy-coef-start", type=float, default=0.10)
    parser.add_argument("--dense-planet-coef", type=float, default=0.05)
    parser.add_argument("--dense-production-coef", type=float, default=0.04)
    parser.add_argument("--dense-ship-share-coef", type=float, default=0.14)
    parser.add_argument("--dense-score-coef", type=float, default=0.10)
    parser.add_argument("--dense-reward-clip", type=float, default=0.40)
    parser.add_argument("--train-mission-mix-bonus-coef", type=float, default=0.0)
    parser.add_argument("--train-target-support-ratio", type=float, default=0.30)
    parser.add_argument("--train-support-ratio-band", type=float, default=0.20)
    parser.add_argument("--train-min-support-ratio", type=float, default=0.12)
    parser.add_argument("--train-max-attack-ratio", type=float, default=0.58)
    parser.add_argument("--train-mission-mix-reward-clip", type=float, default=0.20)
    parser.add_argument("--stabilizer-target-noop", type=float, default=0.60)
    parser.add_argument("--stabilizer-target-passivity", type=float, default=0.50)
    parser.add_argument("--stabilizer-target-real-moves-turn", type=float, default=1.20)
    parser.add_argument("--stabilizer-target-avg-ships-sent", type=float, default=3.0)
    parser.add_argument("--stabilizer-min-weighted-score", type=float, default=0.55)
    parser.add_argument("--stabilizer-max-ship-bonus", type=float, default=1.20)
    parser.add_argument("--stabilizer-max-min-ships", type=int, default=6)
    parser.add_argument("--stabilizer-max-starter-count", type=int, default=5)
    parser.add_argument("--stabilizer-ratio-floor-step", type=float, default=0.10)
    parser.add_argument("--stabilizer-ratio-floor-cap", type=float, default=0.55)
    parser.add_argument("--disable-behavior-supervisor", action="store_true")
    parser.add_argument("--supervisor-max-avg-ships-sent", type=float, default=24.0)
    parser.add_argument("--supervisor-ratio-ceiling-step", type=float, default=0.08)
    parser.add_argument("--supervisor-ratio-ceiling-floor", type=float, default=0.65)
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
        # Keys that drive shaping/penalty intensity. If --reset-shaping-coefs is
        # set we drop these so the run starts from CLI defaults instead of
        # inheriting saturated values from the previous run's supervisor.
        shaping_keys = {
            "do_nothing_logit_penalty",
            "do_nothing_prob_cap",
            "do_nothing_prob_caps_by_slot",
            "train_noop_penalty_coef",
            "train_passivity_penalty_coef",
            "train_action_bonus_coef",
            "train_ships_sent_bonus_coef",
            "train_event_hit_bonus",
            "train_event_enemy_hit_bonus",
            "train_event_capture_bonus",
            "train_event_support_bonus",
            "train_event_lost_penalty",
            "train_event_pending_penalty",
            "train_event_min_shape_clip",
            "train_event_max_flat_action_bonus",
            "train_event_max_ship_volume_bonus",
            "train_event_max_activity_action_bonus",
            "train_event_max_activity_ships_bonus",
            "policy_prior_strength",
            "entropy_coef_start",
            "max_actions_per_turn",
            "min_expand_attack_ships",
            "send_ratios",
        }
        if isinstance(adaptive_config, dict):
            for key in (
                "learning_rate",
                "entropy_coef_start",
                "temperature_start",
                "temperature_end",
                "policy_prior_strength",
                "do_nothing_logit_penalty",
                "do_nothing_prob_cap",
                "do_nothing_prob_caps_by_slot",
                "train_noop_penalty_coef",
                "train_passivity_penalty_coef",
                "train_action_bonus_coef",
                "train_ships_sent_bonus_coef",
                "train_event_hit_bonus",
                "train_event_enemy_hit_bonus",
                "train_event_capture_bonus",
                "train_event_support_bonus",
                "train_event_lost_penalty",
                "train_event_pending_penalty",
                "train_event_min_shape_clip",
                "train_event_max_flat_action_bonus",
                "train_event_max_ship_volume_bonus",
                "train_event_max_activity_action_bonus",
                "train_event_max_activity_ships_bonus",
                "train_mission_mix_bonus_coef",
                "train_target_support_ratio",
                "train_support_ratio_band",
                "train_min_support_ratio",
                "train_max_attack_ratio",
                "train_mission_mix_reward_clip",
                "max_actions_per_turn",
                "min_expand_attack_ships",
                "send_ratios",
                "eval_stabilizer_enabled",
                "stabilizer_eval_count",
                "stabilizer_cooldown_remaining",
            ):
                if key in adaptive_config:
                    if args.reset_shaping_coefs and key in shaping_keys:
                        continue
                    cfg[key] = adaptive_config[key]
            # opponent_mix_counts from checkpoint is intentionally NOT restored:
            # --simple-opponents always wins so the caller controls the training pool.

    run_name = args.run_name or (Path(args.resume_checkpoint).resolve().parent.name if args.resume_checkpoint and Path(args.resume_checkpoint).exists() else _run_tag())
    cfg["resume_checkpoint"] = args.resume_checkpoint or ""

    runs_root = Path(args.runs_root).expanduser() if args.runs_root else PACKAGE_DIR / "runs"
    run_dir = runs_root / run_name
    ensure_dir(run_dir)
    log_path = run_dir / "gpu_train.log"
    checkpoint_path = run_dir / "best.npz"
    best_validated_path = run_dir / "best_validated.npz"
    candidate_path = run_dir / "candidate.npz"
    latest_path = run_dir / "latest.npz"
    eval_history_path = run_dir / "eval_history.jsonl"

    cfg["temperature_start_initial"] = float(cfg.get("temperature_start", 1.05))
    save_json(run_dir / "config.json", cfg)
    _log(log_path, f"run_start name={run_name} device={device} workers={cfg['n_workers']} target={cfg['target_winrate']}")
    _log(log_path, f"policy_prior_strength={cfg['policy_prior_strength']}")

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
    teacher_model = None
    teacher_checkpoint = Path(str(cfg.get("teacher_checkpoint", "")))
    if float(cfg.get("teacher_kl_coef", 0.0)) > 0.0 and teacher_checkpoint.exists():
        teacher_model = _load_model_from_checkpoint(teacher_checkpoint, cfg, device)
        _log(log_path, f"teacher_distillation checkpoint={teacher_checkpoint} kl_coef={float(cfg.get('teacher_kl_coef', 0.0)):.4f}")
    elif float(cfg.get("teacher_kl_coef", 0.0)) > 0.0:
        _log(log_path, f"teacher_distillation disabled missing_checkpoint={teacher_checkpoint}")
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
    best_winrate = float(best_meta.get("valid_winrate", best_meta.get("winrate", resume_meta.get("winrate", -1.0))))
    total_episodes = int(resume_meta.get("total_episodes", 0))
    last_eval_episode = int(resume_meta.get("last_eval_episode", 0))
    pending_episodes: List[Dict[str, Any]] = []
    policy_version = 0
    last_train_metrics: Dict[str, float] = {}
    train_history = deque(maxlen=12)
    eval_history = deque(maxlen=8)
    consecutive_regressions = 0
    collapse_eval_count = 0
    collapse_recovery_count = 0

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
                    teacher_model=teacher_model,
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
                train_record["legal_noop_rate"] = float(metrics.get("legal_noop_rate_mean", train_record["noop_rate"]))
                train_record["forced_noop_rate"] = float(metrics.get("forced_noop_rate_mean", 0.0))
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
                        f"prior={float(cfg.get('policy_prior_strength', 0.0)):.3f} "
                        f"noop_logit_penalty={float(cfg.get('do_nothing_logit_penalty', 0.0)):.3f} "
                        f"noop_prob_cap={float(cfg.get('do_nothing_prob_cap', 1.0)):.3f} "
                        f"noop_slot_caps={cfg.get('do_nothing_prob_caps_by_slot', [])} "
                        f"temp=[{float(cfg.get('temperature_start', 0.0)):.3f},{float(cfg.get('temperature_end', 0.0)):.3f}] "
                        f"noop_penalty={float(cfg.get('train_noop_penalty_coef', 0.0)):.3f} "
                        f"passivity_penalty={float(cfg.get('train_passivity_penalty_coef', 0.0)):.3f} "
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
                    f"teacher_kl={metrics.get('teacher_kl', 0):.4f} "
                    f"entropy={metrics.get('entropy', 0):.3f} "
                    f"kl={metrics.get('approx_kl', 0):.4f} "
                    f"clip_frac={metrics.get('clip_frac', 0):.3f} "
                    f"ratio_std={metrics.get('ratio_std', 0):.6f} "
                    f"ratio_range=[{metrics.get('ratio_min', 0):.4f},{metrics.get('ratio_max', 0):.4f}] "
                    f"logr_max={metrics.get('log_ratio_abs_max', 0):.6f} "
                    f"param_delta={metrics.get('param_relative_delta', 0):.8f} "
                    f"grad_norm={metrics.get('grad_norm', 0):.3f} "
                    f"reward={metrics.get('mean_reward', 0):.3f} "
                    f"return={metrics.get('return_target_mean', 0):.3f}/{metrics.get('return_target_std', 0):.3f} "
                    f"vpred={metrics.get('value_prediction_mean', 0):.3f} "
                    f"adv={metrics.get('raw_advantage_mean', 0):.3f}/{metrics.get('raw_advantage_std', 0):.3f} "
                    f"terminal={metrics.get('terminal_reward_mean', 0):.3f} "
                    f"adj_terminal={metrics.get('adjusted_terminal_reward_mean', 0):.3f} "
                    f"dense={metrics.get('dense_reward_mean', 0):.3f} "
                    f"activity={metrics.get('activity_reward_mean', 0):.3f} "
                    f"event_shape={metrics.get('event_shape_reward_mean', 0):.3f} "
                    f"fleet_hit={metrics.get('fleet_hit_rate_mean', 0):.3f} "
                    f"fleet_capture={metrics.get('fleet_capture_rate_mean', 0):.3f} "
                    f"fleet_lost={metrics.get('fleet_lost_rate_mean', 0):.3f} "
                    f"fleet_pending={metrics.get('fleet_pending_rate_mean', 0):.3f} "
                    f"sup_def={metrics.get('tactical_support_defense_rate_mean', 0):.3f} "
                    f"sup_front={metrics.get('tactical_support_front_rate_mean', 0):.3f} "
                    f"sup_redist={metrics.get('tactical_support_redistribute_rate_mean', 0):.3f} "
                    f"sup_passive={metrics.get('tactical_support_passive_rate_mean', 0):.3f} "
                    f"atk_conv={metrics.get('tactical_attack_convert_rate_mean', 0):.3f} "
                    f"atk_press={metrics.get('tactical_attack_pressure_rate_mean', 0):.3f} "
                    f"atk_miss={metrics.get('attack_convert_missed_rate_mean', 0):.3f} "
                    f"mix={metrics.get('mission_mix_reward_mean', 0):.3f} "
                    f"support={metrics.get('mission_support_ratio_mean', 0):.3f} "
                    f"attack={metrics.get('mission_attack_ratio_mean', 0):.3f} "
                    f"expand={metrics.get('mission_expand_ratio_mean', 0):.3f} "
                    f"passivity_penalty={metrics.get('passivity_penalty_mean', 0):.3f} "
                    f"passivity={metrics.get('do_nothing_rate_mean', 1):.3f} "
                    f"legal_noop={metrics.get('legal_noop_rate_mean', 0):.3f} "
                    f"forced_noop={metrics.get('forced_noop_rate_mean', 0):.3f} "
                    f"step_shape={metrics.get('per_step_shape_sum_mean', 0):.3f} "
                    f"passive_wins={metrics.get('passive_win_rate', 0):.3f} "
                    f"noop_p=[{metrics.get('noop_prob_before_cap_mean', 0):.3f}->{metrics.get('noop_prob_after_cap_mean', 0):.3f}] "
                    f"noop_cap_frac={metrics.get('noop_prob_cap_frac', 0):.3f} "
                    f"slot0_noop={metrics.get('first_slot_noop_rate', 1):.3f} "
                    f"slot0_p=[{metrics.get('slot0_noop_prob_before_cap', 0):.3f}->{metrics.get('slot0_noop_prob_after_cap', 0):.3f}] "
                    f"slot0_cap={metrics.get('slot0_noop_cap_value', 1):.3f} "
                    f"slot0_real={metrics.get('slot0_has_real_candidate', 0):.3f} "
                    f"slot0_real_p=[{metrics.get('slot0_real_noop_prob_before_cap', 0):.3f}->{metrics.get('slot0_real_noop_prob_after_cap', 0):.3f}] "
                    f"slot0_cap_frac={metrics.get('slot0_noop_cap_frac', 0):.3f} "
                    f"slot1_p=[{metrics.get('slot1_noop_prob_before_cap', 0):.3f}->{metrics.get('slot1_noop_prob_after_cap', 0):.3f}] "
                    f"slot1_cap={metrics.get('slot1_noop_cap_value', 1):.3f} "
                    f"slot1_real={metrics.get('slot1_has_real_candidate', 0):.3f} "
                    f"slot1_real_p=[{metrics.get('slot1_real_noop_prob_before_cap', 0):.3f}->{metrics.get('slot1_real_noop_prob_after_cap', 0):.3f}] "
                    f"slot1_cap_frac={metrics.get('slot1_noop_cap_frac', 0):.3f} "
                    f"real_moves_game={metrics.get('mean_real_actions_per_game', 0):.2f} "
                    f"real_moves_turn={metrics.get('mean_real_actions_per_turn', 0):.2f} "
                    f"ships_mean_all={metrics.get('ships_sent_mean_all_steps', 0):.2f} "
                    f"ships_p50={metrics.get('ships_sent_median', 0):.2f} "
                    f"ships_p75={metrics.get('ships_sent_p75', 0):.2f} "
                    f"ships_p90={metrics.get('ships_sent_p90', 0):.2f} "
                    f"ships_max={metrics.get('ships_sent_max', 0):.2f} "
                    f"ep_ship_p50={metrics.get('episode_ships_median_mean', 0):.2f} "
                    f"ep_ship_p90={metrics.get('episode_ships_p90_mean', 0):.2f} "
                    f"ep_ship_max={metrics.get('episode_ships_max_mean', 0):.2f} "
                    f"slot0_ships={metrics.get('slot0_ships_mean', 0):.2f} "
                    f"slot1_ships={metrics.get('slot1_ships_mean', 0):.2f} "
                    f"expand_ships={metrics.get('mission_expand_ships_mean', 0):.2f} "
                    f"attack_ships={metrics.get('mission_attack_ships_mean', 0):.2f} "
                    f"support_ships={metrics.get('mission_support_ships_mean', 0):.2f} "
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
                pushed, dropped_updates = _put_latest_model_update(model_update_queue, new_state)
                if dropped_updates or not pushed:
                    _log(log_path, f"MODEL_UPDATE_QUEUE pushed={int(pushed)} dropped_stale={dropped_updates}")
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
                            "policy_prior_strength": float(cfg.get("policy_prior_strength", 0.0)),
                            "do_nothing_logit_penalty": float(cfg.get("do_nothing_logit_penalty", 0.0)),
                            "do_nothing_prob_cap": float(cfg.get("do_nothing_prob_cap", 1.0)),
                            "do_nothing_prob_caps_by_slot": list(cfg.get("do_nothing_prob_caps_by_slot", [])),
                            "train_noop_penalty_coef": float(cfg.get("train_noop_penalty_coef", 0.0)),
                            "train_passivity_penalty_coef": float(cfg.get("train_passivity_penalty_coef", 0.0)),
                            "train_action_bonus_coef": float(cfg.get("train_action_bonus_coef", 0.0)),
                            "train_ships_sent_bonus_coef": float(cfg.get("train_ships_sent_bonus_coef", 0.0)),
                            "train_event_hit_bonus": float(cfg.get("train_event_hit_bonus", 0.0)),
                            "train_event_enemy_hit_bonus": float(cfg.get("train_event_enemy_hit_bonus", 0.0)),
                            "train_event_capture_bonus": float(cfg.get("train_event_capture_bonus", 0.0)),
                            "train_event_support_bonus": float(cfg.get("train_event_support_bonus", 0.0)),
                            "train_event_lost_penalty": float(cfg.get("train_event_lost_penalty", 0.0)),
                            "train_event_pending_penalty": float(cfg.get("train_event_pending_penalty", 0.0)),
                            "train_event_min_shape_clip": float(cfg.get("train_event_min_shape_clip", 0.0)),
                            "train_event_max_flat_action_bonus": float(cfg.get("train_event_max_flat_action_bonus", 0.0)),
                            "train_event_max_ship_volume_bonus": float(cfg.get("train_event_max_ship_volume_bonus", 0.0)),
                            "train_event_max_activity_action_bonus": float(cfg.get("train_event_max_activity_action_bonus", 0.0)),
                            "train_event_max_activity_ships_bonus": float(cfg.get("train_event_max_activity_ships_bonus", 0.0)),
                            "min_expand_attack_ships": int(cfg.get("min_expand_attack_ships", 2)),
                            "send_ratios": list(cfg.get("send_ratios", [])),
                            "behavior_supervisor_enabled": bool(cfg.get("behavior_supervisor_enabled", True)),
                            "eval_stabilizer_enabled": bool(cfg.get("eval_stabilizer_enabled", True)),
                            "stabilizer_eval_count": int(cfg.get("stabilizer_eval_count", 0)),
                            "stabilizer_cooldown_remaining": int(cfg.get("stabilizer_cooldown_remaining", 0)),
                            "opponent_mix_counts": dict(cfg.get("opponent_mix_counts", {})),
                        },
                    },
                )
                pending_episodes = []

            # Eval every N episodes
            if total_episodes - last_eval_episode >= cfg["eval_every"]:
                last_eval_episode = total_episodes
                current_lr = float(optimizer.param_groups[0]["lr"])
                eval_round = max(0, int(total_episodes // max(1, int(cfg["eval_every"]))) - 1)
                eval_seed_start = int(cfg["eval_seed_start"]) + eval_round * int(cfg.get("eval_seed_stride", 1000000))
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
                    seed_start=eval_seed_start,
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
                collapsed_eval = (
                    float(eval_result.get("eval_do_nothing_rate", 1.0)) >= float(cfg.get("collapse_stop_noop_rate", 0.92))
                    and float(eval_result.get("eval_avg_ships_sent", 0.0)) <= float(cfg.get("collapse_stop_max_avg_ships_sent", 2.0))
                )
                collapse_eval_count = collapse_eval_count + 1 if collapsed_eval else 0
                if collapsed_eval:
                    _log(
                        log_path,
                        f"COLLAPSE_DETECTED count={collapse_eval_count} "
                        f"noop={float(eval_result.get('eval_do_nothing_rate', 1.0)):.3f} "
                        f"ships={float(eval_result.get('eval_avg_ships_sent', 0.0)):.2f} "
                        f"valid={float(eval_result.get('valid_winrate', eval_result.get('winrate', 0.0))):.3f}",
                    )
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
                best_eval_every = max(1, int(cfg.get("best_eval_every", 1)))
                should_eval_best = best_eval_every <= 1 or (eval_round % best_eval_every) == 0 or best_winrate < 0.0
                if should_eval_best:
                    best_model = _load_model_from_checkpoint(best_validated_path, cfg, device)
                    best_eval = _evaluate(
                        best_model, cfg, device,
                        episodes=cfg["eval_episodes"],
                        seed_start=eval_seed_start,
                        pool=eval_pool,
                        n_players=n_players,
                        eval_history_path=eval_history_path,
                        checkpoint_label="best_validated",
                        episode_count=total_episodes,
                        lr=current_lr,
                        train_metrics=last_train_metrics,
                        progress_log_path=log_path,
                    )
                    best_meta = dict(best_eval)
                else:
                    best_eval = dict(best_meta)
                    cached_wr = float(best_eval.get("winrate", best_winrate if best_winrate >= 0.0 else 0.0))
                    cached_valid = float(best_eval.get("valid_winrate", cached_wr))
                    best_eval.setdefault("winrate", cached_wr)
                    best_eval.setdefault("valid_winrate", cached_valid)
                    best_eval.setdefault("weighted_score", _weighted_eval_score(best_eval))
                    best_eval.setdefault("ci_low", cached_wr)
                    best_eval.setdefault("ci_high", cached_wr)
                    best_eval.setdefault("by_opponent", {})
                    _log(
                        log_path,
                        f"eval_skip checkpoint=best_validated reason=cached "
                        f"round={eval_round} best_eval_every={best_eval_every}",
                    )
                elapsed = (time.time() - started_at) / 60.0
                if best_validated_path.exists() and best_winrate < 0.0:
                    best_winrate = float(best_eval.get("valid_winrate", best_eval.get("winrate", best_winrate)))
                promote, promotion_reasons, rollback = _promotion_decision(eval_result, best_eval, cfg)
                _log(
                    log_path,
                    f"eval episodes={total_episodes} elapsed={elapsed:.1f}m "
                    f"winrate={eval_result['winrate']:.3f} "
                    f"valid={eval_result.get('valid_winrate', eval_result['winrate']):.3f} "
                    f"wscore={eval_result.get('weighted_score', eval_result['winrate']):.3f} "
                    f"ci=[{eval_result['ci_low']:.3f},{eval_result['ci_high']:.3f}] "
                    f"best_eval={best_eval['winrate']:.3f} "
                    f"best_valid={best_eval.get('valid_winrate', best_eval['winrate']):.3f} "
                    f"best_wscore={best_eval.get('weighted_score', best_eval['winrate']):.3f} "
                    f"best_ci=[{best_eval['ci_low']:.3f},{best_eval['ci_high']:.3f}] "
                    f"noop={eval_result['eval_do_nothing_rate']:.3f} "
                    f"legal_noop={eval_result.get('eval_legal_noop_rate', 0):.3f} "
                    f"forced_noop={eval_result.get('eval_forced_noop_rate', 0):.3f} "
                    f"ships={eval_result['eval_avg_ships_sent']:.2f} "
                    f"lr={current_lr:.8f} "
                    f"best={best_winrate:.3f} target={cfg['target_winrate']}",
                )
                if promote:
                    best_winrate = float(eval_result.get("valid_winrate", eval_result["winrate"]))
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
                            "policy_prior_strength": float(cfg.get("policy_prior_strength", 0.0)),
                            "do_nothing_logit_penalty": float(cfg.get("do_nothing_logit_penalty", 0.0)),
                            "do_nothing_prob_cap": float(cfg.get("do_nothing_prob_cap", 1.0)),
                            "do_nothing_prob_caps_by_slot": list(cfg.get("do_nothing_prob_caps_by_slot", [])),
                            "train_noop_penalty_coef": float(cfg.get("train_noop_penalty_coef", 0.0)),
                            "train_passivity_penalty_coef": float(cfg.get("train_passivity_penalty_coef", 0.0)),
                            "train_action_bonus_coef": float(cfg.get("train_action_bonus_coef", 0.0)),
                            "train_ships_sent_bonus_coef": float(cfg.get("train_ships_sent_bonus_coef", 0.0)),
                            "train_event_hit_bonus": float(cfg.get("train_event_hit_bonus", 0.0)),
                            "train_event_enemy_hit_bonus": float(cfg.get("train_event_enemy_hit_bonus", 0.0)),
                            "train_event_capture_bonus": float(cfg.get("train_event_capture_bonus", 0.0)),
                            "train_event_support_bonus": float(cfg.get("train_event_support_bonus", 0.0)),
                            "train_event_lost_penalty": float(cfg.get("train_event_lost_penalty", 0.0)),
                            "train_event_pending_penalty": float(cfg.get("train_event_pending_penalty", 0.0)),
                            "train_event_min_shape_clip": float(cfg.get("train_event_min_shape_clip", 0.0)),
                            "train_event_max_flat_action_bonus": float(cfg.get("train_event_max_flat_action_bonus", 0.0)),
                            "train_event_max_ship_volume_bonus": float(cfg.get("train_event_max_ship_volume_bonus", 0.0)),
                            "train_event_max_activity_action_bonus": float(cfg.get("train_event_max_activity_action_bonus", 0.0)),
                            "train_event_max_activity_ships_bonus": float(cfg.get("train_event_max_activity_ships_bonus", 0.0)),
                            "min_expand_attack_ships": int(cfg.get("min_expand_attack_ships", 2)),
                            "send_ratios": list(cfg.get("send_ratios", [])),
                            "eval_stabilizer_enabled": bool(cfg.get("eval_stabilizer_enabled", True)),
                            "stabilizer_eval_count": int(cfg.get("stabilizer_eval_count", 0)),
                            "stabilizer_cooldown_remaining": int(cfg.get("stabilizer_cooldown_remaining", 0)),
                            "opponent_mix_counts": dict(cfg.get("opponent_mix_counts", {})),
                        },
                    }
                    save_checkpoint(best_validated_path, _model_state_np(model), metadata)
                    save_checkpoint(checkpoint_path, _model_state_np(model), metadata)
                    best_meta = dict(metadata)
                    archive_path = _archive_validated_checkpoint(run_dir, total_episodes, model, metadata)
                    new_lr = min(float(cfg["max_lr"]), current_lr * float(cfg["promotion_lr_mult"]))
                    _set_optimizer_lr(optimizer, new_lr)
                    consecutive_regressions = 0
                    _log(
                        log_path,
                        f"PROMOTED valid_winrate={best_winrate:.4f} archive={archive_path} lr={current_lr:.8f}->{new_lr:.8f}",
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
                        policy_version = 0
                        cfg["temperature_start"] = float(cfg.get("temperature_start_initial", 1.05))
                        train_history.clear()
                        rollback_metadata = {
                            "winrate": best_winrate,
                            "baseline": baseline,
                            "total_episodes": total_episodes,
                            "last_eval_episode": last_eval_episode,
                            "run_name": run_name,
                            "policy_version": policy_version,
                            "adaptive_config": {
                                "learning_rate": new_lr,
                                "entropy_coef_start": float(cfg.get("entropy_coef_start", 0.10)),
                                "temperature_start": float(cfg.get("temperature_start", 1.05)),
                                "temperature_end": float(cfg.get("temperature_end", 0.18)),
                                "policy_prior_strength": float(cfg.get("policy_prior_strength", 0.0)),
                                "do_nothing_logit_penalty": float(cfg.get("do_nothing_logit_penalty", 0.0)),
                                "do_nothing_prob_cap": float(cfg.get("do_nothing_prob_cap", 1.0)),
                                "do_nothing_prob_caps_by_slot": list(cfg.get("do_nothing_prob_caps_by_slot", [])),
                                "train_noop_penalty_coef": float(cfg.get("train_noop_penalty_coef", 0.0)),
                                "train_passivity_penalty_coef": float(cfg.get("train_passivity_penalty_coef", 0.0)),
                                "train_action_bonus_coef": float(cfg.get("train_action_bonus_coef", 0.0)),
                                "train_ships_sent_bonus_coef": float(cfg.get("train_ships_sent_bonus_coef", 0.0)),
                                "train_mission_mix_bonus_coef": float(cfg.get("train_mission_mix_bonus_coef", 0.0)),
                                "train_target_support_ratio": float(cfg.get("train_target_support_ratio", 0.0)),
                                "train_support_ratio_band": float(cfg.get("train_support_ratio_band", 0.0)),
                                "train_min_support_ratio": float(cfg.get("train_min_support_ratio", 0.0)),
                                "train_max_attack_ratio": float(cfg.get("train_max_attack_ratio", 0.0)),
                                "train_mission_mix_reward_clip": float(cfg.get("train_mission_mix_reward_clip", 0.0)),
                                "min_expand_attack_ships": int(cfg.get("min_expand_attack_ships", 2)),
                                "send_ratios": list(cfg.get("send_ratios", [])),
                                "behavior_supervisor_enabled": bool(cfg.get("behavior_supervisor_enabled", True)),
                                "eval_stabilizer_enabled": bool(cfg.get("eval_stabilizer_enabled", True)),
                                "stabilizer_eval_count": int(cfg.get("stabilizer_eval_count", 0)),
                                "stabilizer_cooldown_remaining": int(cfg.get("stabilizer_cooldown_remaining", 0)),
                                "opponent_mix_counts": dict(cfg.get("opponent_mix_counts", {})),
                            },
                            "note": "latest reset to best_validated after rollback",
                        }
                        save_checkpoint(latest_path, _model_state_np(model), rollback_metadata)
                        save_checkpoint(checkpoint_path, _model_state_np(model), rollback_metadata)
                        pushed, dropped_updates = _put_latest_model_update(
                            model_update_queue,
                            {"state": _model_state_np(model), "policy_version": policy_version},
                        )
                        if dropped_updates or not pushed:
                            _log(log_path, f"MODEL_UPDATE_QUEUE pushed={int(pushed)} dropped_stale={dropped_updates}")
                        _log(
                            log_path,
                            f"ROLLBACK failed={failed_path} regressions={consecutive_regressions} "
                            f"lr={current_lr:.8f}->{new_lr:.8f}",
                        )
                        max_stuck_regressions = int(cfg.get("max_consecutive_regressions", 0))
                        if max_stuck_regressions > 0 and consecutive_regressions >= max_stuck_regressions:
                            cfg["entropy_coef_start"] = min(0.35, float(cfg.get("entropy_coef_start", 0.10)) * 1.25)
                            cfg["train_per_step_legal_noop_penalty"] = max(0.0001, float(cfg.get("train_per_step_legal_noop_penalty", 0.0012)) * 0.75)
                            cfg["train_per_step_real_action_bonus"] = max(0.0001, float(cfg.get("train_per_step_real_action_bonus", 0.0008)) * 0.75)
                            # Reset LR partway back up so gradient updates are effective.
                            stuck_lr = min(float(cfg["max_lr"]), float(cfg.get("learning_rate", current_lr)) * 0.5)
                            stuck_lr = max(float(cfg["min_lr"]), stuck_lr)
                            _set_optimizer_lr(optimizer, stuck_lr)
                            consecutive_regressions = 0
                            _log(
                                log_path,
                                f"STUCK_RECOVERY entropy={cfg['entropy_coef_start']:.3f} "
                                f"shaping_noop={cfg['train_per_step_legal_noop_penalty']:.5f} "
                                f"shaping_action={cfg['train_per_step_real_action_bonus']:.5f} "
                                f"lr={current_lr:.8f}->{stuck_lr:.8f}",
                            )
                stabilizer_patch = _eval_stabilizer(cfg, list(eval_history))
                if stabilizer_patch:
                    _log(
                        log_path,
                        "SUPERVISOR "
                        f"action={stabilizer_patch.get('stabilizer_action', '')} "
                        f"reasons={stabilizer_patch.get('stabilizer_reasons', '')} "
                        f"score={float(stabilizer_patch.get('stabilizer_score', eval_result.get('weighted_score', 0.0))):.3f} "
                        f"noop={float(stabilizer_patch.get('stabilizer_noop', eval_result.get('eval_do_nothing_rate', 0.0))):.3f} "
                        f"ships={float(stabilizer_patch.get('stabilizer_ships', eval_result.get('eval_avg_ships_sent', 0.0))):.2f} "
                        f"passivity={float(stabilizer_patch.get('stabilizer_passivity', last_train_metrics.get('do_nothing_rate_mean', 0.0))):.3f} "
                        f"moves={float(stabilizer_patch.get('stabilizer_real_moves', last_train_metrics.get('mean_real_actions_per_turn', 0.0))):.2f} "
                        f"min_ships={int(cfg.get('min_expand_attack_ships', 2))} "
                        f"ship_bonus={float(cfg.get('train_ships_sent_bonus_coef', 0.0)):.3f} "
                        f"noop_penalty={float(cfg.get('train_noop_penalty_coef', 0.0)):.3f} "
                        f"passivity_penalty={float(cfg.get('train_passivity_penalty_coef', 0.0)):.3f} "
                        f"ratios={cfg.get('send_ratios', [])} "
                        f"counts={cfg.get('opponent_mix_counts', {})}",
                    )
                    control_msg = {"config_patch": dict(stabilizer_patch)}
                    if "stage1_pool" in stabilizer_patch:
                        control_msg["pool"] = list(stabilizer_patch["stage1_pool"])
                    for queue in control_queues.values():
                        try:
                            queue.put_nowait(control_msg)
                        except Exception:
                            pass
                    pushed, dropped_updates = _put_latest_model_update(
                        model_update_queue,
                        {
                            "state": _model_state_np(model),
                            "policy_version": policy_version,
                            "config_patch": dict(stabilizer_patch),
                        },
                    )
                    if dropped_updates or not pushed:
                        _log(log_path, f"MODEL_UPDATE_QUEUE pushed={int(pushed)} dropped_stale={dropped_updates}")
                    save_json(run_dir / "config.json", cfg)
                max_collapse_recoveries = int(cfg.get("collapse_max_recoveries", 0))
                if int(cfg.get("collapse_stop_evals", 0)) > 0 and collapse_eval_count >= int(cfg.get("collapse_stop_evals", 0)):
                    collapse_recovery_count += 1
                    if max_collapse_recoveries > 0 and collapse_recovery_count > max_collapse_recoveries:
                        _log(
                            log_path,
                            f"STOP collapse_max_recoveries reached recoveries={collapse_recovery_count} "
                            f"noop={float(eval_result.get('eval_do_nothing_rate', 1.0)):.3f} "
                            f"ships={float(eval_result.get('eval_avg_ships_sent', 0.0)):.2f}",
                        )
                        break
                    # Recovery: roll back to best checkpoint, increase entropy, reduce shaping.
                    recovery_checkpoint = best_validated_path if best_validated_path.exists() else Path(cfg.get("resume_checkpoint", ""))
                    if recovery_checkpoint.exists():
                        state, _ = load_checkpoint(recovery_checkpoint)
                        load_compatible_state_dict(model, state)
                        model = model.to(device)
                        model.eval()
                        policy_version = 0
                    cfg["entropy_coef_start"] = min(0.35, float(cfg.get("entropy_coef_start", 0.10)) * 1.50)
                    cfg["train_per_step_legal_noop_penalty"] = max(0.001, float(cfg.get("train_per_step_legal_noop_penalty", 0.006)) * 0.60)
                    cfg["train_per_step_real_action_bonus"] = max(0.001, float(cfg.get("train_per_step_real_action_bonus", 0.004)) * 0.60)
                    cfg["train_per_step_shape_clip"] = max(0.005, float(cfg.get("train_per_step_shape_clip", 0.02)) * 0.60)
                    new_lr = max(float(cfg["min_lr"]), current_lr * float(cfg.get("rollback_lr_mult", 0.5)))
                    optimizer = torch.optim.Adam(model.parameters(), lr=new_lr)
                    train_history.clear()
                    eval_history.clear()
                    collapse_eval_count = 0
                    consecutive_regressions = 0
                    pushed, dropped_updates = _put_latest_model_update(
                        model_update_queue,
                        {"state": _model_state_np(model), "policy_version": policy_version},
                    )
                    if dropped_updates or not pushed:
                        _log(log_path, f"MODEL_UPDATE_QUEUE pushed={int(pushed)} dropped_stale={dropped_updates}")
                    _log(
                        log_path,
                        f"COLLAPSE_RECOVERY count={collapse_recovery_count}/{max_collapse_recoveries or 'inf'} "
                        f"entropy={cfg['entropy_coef_start']:.3f} "
                        f"shaping_noop={cfg['train_per_step_legal_noop_penalty']:.4f} "
                        f"shaping_action={cfg['train_per_step_real_action_bonus']:.4f} "
                        f"lr={current_lr:.8f}->{new_lr:.8f} "
                        f"checkpoint={recovery_checkpoint}",
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
