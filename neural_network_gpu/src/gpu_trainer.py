from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_network.src.model import NeuralNetworkModel
from neural_network.src.population_4p_training import _activity_shaping_reward
from neural_network_gpu.src.action_metrics import summarize_action_records
from neural_network_gpu.src.probability import cap_do_nothing_probability_with_info, caps_for_action_slots


def _mission_mix_shaping_reward(action_metrics: Dict[str, Any], config: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    counts = action_metrics.get("mission_counts", {}) or {}
    expand = float(counts.get("expand", 0))
    attack = float(counts.get("attack", 0))
    support = float(counts.get("support", 0))
    real = expand + attack + support
    if real <= 0.0:
        return 0.0, {
            "mission_mix_reward": 0.0,
            "mission_expand_ratio": 0.0,
            "mission_attack_ratio": 0.0,
            "mission_support_ratio": 0.0,
        }

    expand_ratio = expand / real
    attack_ratio = attack / real
    support_ratio = support / real
    coef = max(0.0, float(config.get("train_mission_mix_bonus_coef", 0.0)))
    target_support = max(0.0, min(1.0, float(config.get("train_target_support_ratio", 0.30))))
    support_band = max(1e-6, float(config.get("train_support_ratio_band", 0.20)))
    min_support = max(0.0, min(1.0, float(config.get("train_min_support_ratio", 0.12))))
    max_attack = max(0.0, min(1.0, float(config.get("train_max_attack_ratio", 0.58))))
    clip = max(0.0, float(config.get("train_mission_mix_reward_clip", 0.20)))

    support_shape = max(0.0, 1.0 - abs(support_ratio - target_support) / support_band)
    reward = coef * support_shape
    if support_ratio < min_support:
        reward -= coef * ((min_support - support_ratio) / max(1e-6, min_support))
    if attack_ratio > max_attack:
        reward -= coef * ((attack_ratio - max_attack) / max(1e-6, 1.0 - max_attack))
    if clip > 0.0:
        reward = max(-clip, min(clip, reward))

    return float(reward), {
        "mission_mix_reward": float(reward),
        "mission_expand_ratio": float(expand_ratio),
        "mission_attack_ratio": float(attack_ratio),
        "mission_support_ratio": float(support_ratio),
    }


def _counterfactual_selected_bonus(
    step: Dict[str, Any],
    episode_reward: float,
    step_shape: float,
    config: Dict[str, Any],
) -> float:
    """Outcome correction for the actually sampled candidate.

    The full counterfactual target ranks every candidate before the move is
    played. This term then uses causal feedback from the sampled move so the
    target remains anchored to real outcomes instead of pure hand ranking.
    """
    coef = max(0.0, float(config.get("counterfactual_selected_outcome_coef", 1.0)))
    if coef <= 0.0:
        return 0.0

    mission = str(step.get("mission") or "do_nothing")
    ships = int(step.get("ships") or 0)
    is_noop = mission == "do_nothing" or ships <= 0
    bonus = 0.0
    if is_noop:
        if bool(step.get("noop_has_real_candidate", False)):
            bonus -= float(config.get("counterfactual_selected_legal_noop_penalty", 0.35))
    else:
        if bool(step.get("fleet_captured", False)):
            bonus += float(config.get("counterfactual_selected_capture_bonus", 0.90))
        elif bool(step.get("fleet_enemy_hit", False)):
            bonus += float(config.get("counterfactual_selected_enemy_hit_bonus", 0.35))
        elif bool(step.get("fleet_neutral_hit", False)):
            bonus += float(config.get("counterfactual_selected_neutral_hit_bonus", 0.25))
        elif bool(step.get("fleet_supported", False)):
            bonus += float(config.get("counterfactual_selected_support_bonus", 0.20))
        elif bool(step.get("fleet_hit", False)):
            bonus += float(config.get("counterfactual_selected_hit_bonus", 0.12))
        if bool(step.get("fleet_lost", False)):
            bonus -= float(config.get("counterfactual_selected_lost_penalty", 0.80))
        if bool(step.get("fleet_pending", False)):
            bonus -= float(config.get("counterfactual_selected_pending_penalty", 0.03))

    bonus += float(config.get("counterfactual_selected_episode_coef", 0.12)) * float(episode_reward)
    bonus += float(config.get("counterfactual_selected_step_shape_coef", 0.35)) * float(step_shape)
    clip = max(0.0, float(config.get("counterfactual_selected_bonus_clip", 1.20)))
    if clip > 0.0:
        bonus = max(-clip, min(clip, bonus))
    return float(coef * bonus)


def train_on_episodes(
    model: NeuralNetworkModel,
    optimizer: torch.optim.Optimizer,
    episodes: List[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
    baseline: float,
    baseline_momentum: float,
    teacher_model: NeuralNetworkModel | None = None,
) -> Tuple[float, Dict[str, float]]:
    if not episodes:
        return baseline, {}

    entropy_coef = float(config.get("entropy_coef_start", 0.065))
    value_coef = float(config.get("value_loss_coef", 0.25))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    ppo_clip_eps = float(config.get("ppo_clip_eps", 0.2))
    ppo_epochs = int(config.get("ppo_epochs", 3))
    minibatch_size = max(1, int(config.get("ppo_minibatch_size", 512)))
    teacher_kl_coef = max(0.0, float(config.get("teacher_kl_coef", 0.0))) if teacher_model is not None else 0.0
    on_policy_imitation_coef = max(0.0, float(config.get("on_policy_imitation_coef", 0.0)))
    on_policy_imitation_min_margin = max(0.0, float(config.get("on_policy_imitation_min_margin", 0.20)))
    on_policy_imitation_max_weight = max(1.0, float(config.get("on_policy_imitation_max_weight", 2.0)))
    counterfactual_imitation_coef = max(0.0, float(config.get("counterfactual_imitation_coef", 0.0)))
    counterfactual_temperature = max(1e-3, float(config.get("counterfactual_temperature", 0.80)))
    counterfactual_min_margin = max(0.0, float(config.get("counterfactual_min_margin", 0.05)))
    counterfactual_max_weight = max(1.0, float(config.get("counterfactual_max_weight", 2.5)))
    advantage_eps = 1e-6
    return_gamma = max(0.0, min(1.0, float(config.get("train_return_gamma", config.get("gamma", 0.997)))))
    return_clip = max(0.0, float(config.get("train_return_clip", 2.0)))
    event_shaping_enabled = bool(config.get("train_event_shaping_enabled", True))
    effective_activity_action_bonus = float(config.get("train_action_bonus_coef", 0.08))
    effective_activity_ship_bonus = float(config.get("train_ships_sent_bonus_coef", 0.04))
    if event_shaping_enabled:
        effective_activity_action_bonus = min(
            effective_activity_action_bonus,
            max(0.0, float(config.get("train_event_max_activity_action_bonus", 0.03))),
        )
        effective_activity_ship_bonus = min(
            effective_activity_ship_bonus,
            max(0.0, float(config.get("train_event_max_activity_ships_bonus", 0.0))),
        )
    activity_config = config
    if (
        effective_activity_action_bonus != float(config.get("train_action_bonus_coef", 0.08))
        or effective_activity_ship_bonus != float(config.get("train_ships_sent_bonus_coef", 0.04))
    ):
        activity_config = dict(config)
        activity_config["train_action_bonus_coef"] = effective_activity_action_bonus
        activity_config["train_ships_sent_bonus_coef"] = effective_activity_ship_bonus

    # Build per-step dataset across all episodes
    dataset: List[Dict[str, Any]] = []
    rewards: List[float] = []
    terminal_rewards: List[float] = []
    adjusted_terminal_rewards: List[float] = []
    dense_rewards: List[float] = []
    activity_rewards: List[float] = []
    mission_mix_rewards: List[float] = []
    mission_expand_ratios: List[float] = []
    mission_attack_ratios: List[float] = []
    mission_support_ratios: List[float] = []
    event_shape_rewards: List[float] = []
    fleet_launch_mapped_rates: List[float] = []
    fleet_outcome_known_rates: List[float] = []
    fleet_hit_rates: List[float] = []
    fleet_enemy_hit_rates: List[float] = []
    fleet_neutral_hit_rates: List[float] = []
    fleet_support_rates: List[float] = []
    fleet_capture_rates: List[float] = []
    fleet_lost_rates: List[float] = []
    fleet_lost_sun_rates: List[float] = []
    fleet_lost_oob_rates: List[float] = []
    fleet_pending_rates: List[float] = []
    tactical_rate_values: Dict[str, List[float]] = {}
    passivity_penalties: List[float] = []
    passive_win_flags: List[float] = []
    do_nothing_rates: List[float] = []
    legal_noop_rates: List[float] = []
    forced_noop_rates: List[float] = []
    per_step_shape_sums: List[float] = []
    skipped_missing_old_log_prob = 0
    trajectory_lengths: List[int] = []
    episode_turn_lengths: List[int] = []
    all_ships_sent: List[float] = []
    all_real_ships_sent: List[float] = []
    all_real_ships_by_slot: Dict[int, List[float]] = {}
    all_real_ships_by_mission: Dict[str, List[float]] = {}
    episode_ship_medians: List[float] = []
    episode_ship_p90s: List[float] = []
    episode_ship_maxes: List[float] = []
    slot_real_actions: Dict[int, List[float]] = {}
    slot_noop_actions: Dict[int, List[float]] = {}
    inference_noop_probs_before_cap: Dict[int, List[float]] = {}
    inference_noop_probs_after_cap: Dict[int, List[float]] = {}
    inference_noop_cap_values: Dict[int, List[float]] = {}
    inference_noop_has_real_candidate: Dict[int, List[float]] = {}
    inference_noop_cap_applied: Dict[int, List[float]] = {}
    inference_real_noop_probs_before_cap: Dict[int, List[float]] = {}
    inference_real_noop_probs_after_cap: Dict[int, List[float]] = {}

    for ep in episodes:
        trajectory = ep["trajectory"]
        if not trajectory:
            continue
        action_metrics = ep.get("action_metrics", {})
        activity_reward = _activity_shaping_reward(action_metrics, activity_config)
        mission_mix_reward, mission_mix_stats = _mission_mix_shaping_reward(action_metrics, config)
        do_nothing_rate = max(0.0, min(1.0, float(action_metrics.get("do_nothing_rate", 1.0))))
        legal_noop_rate = max(0.0, min(1.0, float(action_metrics.get("legal_noop_rate", do_nothing_rate))))
        forced_noop_rate = max(0.0, min(1.0, float(action_metrics.get("forced_noop_rate", 0.0))))
        # Passivity penalty now reads the legal (avoidable) no-op rate so it
        # cannot be driven by states with no real candidate. The episodic term
        # is gated off by default in favour of per-step shaping below.
        passive_limit = max(0.0, min(1.0, float(config.get("train_target_legal_noop_rate", 0.10))))
        passive_coef = max(0.0, float(config.get("train_passivity_penalty_coef", 0.0)))
        episodic_passivity_enabled = bool(config.get("train_episodic_passivity_penalty_enabled", False))
        passive_excess = max(0.0, legal_noop_rate - passive_limit) / max(1e-6, 1.0 - passive_limit)
        passivity_penalty = passive_coef * passive_excess if episodic_passivity_enabled else 0.0
        terminal_reward = float(ep["terminal_reward"])
        avg_ships_sent = float(action_metrics.get("avg_ships_sent", 0.0))
        real_action_count = float(action_metrics.get("real_action_count", 0.0))
        episode_turns = max(1.0, float(ep.get("episode_length") or 0))
        real_moves_per_turn = real_action_count / episode_turns
        # Passive-win gate switches to legal rate: a "win while idling" is now
        # only flagged when the agent was idling on decisions it could have acted on.
        passive_win_legal_threshold = float(config.get("train_passive_win_legal_noop_rate", 0.30))
        passive_win = (
            terminal_reward > 0.0
            and (
                legal_noop_rate > passive_win_legal_threshold
                or avg_ships_sent < float(config.get("train_passive_win_min_avg_ships_sent", 4.0))
                or real_moves_per_turn < float(config.get("train_passive_win_min_real_moves_turn", 1.0))
            )
        )
        adjusted_terminal_reward = (
            float(config.get("train_passive_win_terminal_reward", -0.25))
            if passive_win
            else terminal_reward
        )
        dense_reward = float(ep.get("dense_reward", 0.0))
        reward = float(adjusted_terminal_reward + 0.15 * dense_reward + 0.05 * activity_reward + mission_mix_reward - passivity_penalty)
        reward = max(-1.2, min(1.2, reward))
        rewards.append(reward)
        terminal_rewards.append(terminal_reward)
        adjusted_terminal_rewards.append(adjusted_terminal_reward)
        dense_rewards.append(dense_reward)
        activity_rewards.append(float(activity_reward))
        mission_mix_rewards.append(float(mission_mix_reward))
        mission_expand_ratios.append(float(mission_mix_stats["mission_expand_ratio"]))
        mission_attack_ratios.append(float(mission_mix_stats["mission_attack_ratio"]))
        mission_support_ratios.append(float(mission_mix_stats["mission_support_ratio"]))
        fleet_launch_mapped_rates.append(float(action_metrics.get("fleet_launch_mapped_rate", 0.0)))
        fleet_outcome_known_rates.append(float(action_metrics.get("fleet_outcome_known_rate", 0.0)))
        fleet_hit_rates.append(float(action_metrics.get("fleet_hit_rate", 0.0)))
        fleet_enemy_hit_rates.append(float(action_metrics.get("fleet_enemy_hit_rate", 0.0)))
        fleet_neutral_hit_rates.append(float(action_metrics.get("fleet_neutral_hit_rate", 0.0)))
        fleet_support_rates.append(float(action_metrics.get("fleet_support_rate", 0.0)))
        fleet_capture_rates.append(float(action_metrics.get("fleet_capture_rate", 0.0)))
        fleet_lost_rates.append(float(action_metrics.get("fleet_lost_rate", 0.0)))
        fleet_lost_sun_rates.append(float(action_metrics.get("fleet_lost_sun_rate", 0.0)))
        fleet_lost_oob_rates.append(float(action_metrics.get("fleet_lost_oob_rate", 0.0)))
        fleet_pending_rates.append(float(action_metrics.get("fleet_pending_rate", 0.0)))
        for tag in (
            "support_defense",
            "support_front",
            "support_redistribute",
            "support_passive",
            "support_backward",
            "attack_convert",
            "attack_opportunity",
            "attack_pressure",
            "attack_poor",
            "expand_front",
            "expand_safe",
        ):
            tactical_rate_values.setdefault(tag, []).append(float(action_metrics.get(f"tactical_{tag}_rate", 0.0)))
        tactical_rate_values.setdefault("candidate_attack_convert", []).append(float(action_metrics.get("candidate_attack_convert_rate", 0.0)))
        tactical_rate_values.setdefault("candidate_attack_pressure", []).append(float(action_metrics.get("candidate_attack_pressure_rate", 0.0)))
        tactical_rate_values.setdefault("attack_convert_missed", []).append(float(action_metrics.get("attack_convert_missed_rate", 0.0)))
        tactical_rate_values.setdefault("good_attack_missed", []).append(float(action_metrics.get("good_attack_missed_rate", 0.0)))
        tactical_rate_values.setdefault("oracle_match", []).append(float(action_metrics.get("tactical_oracle_match_rate", 0.0)))
        tactical_rate_values.setdefault("oracle_real", []).append(float(action_metrics.get("tactical_oracle_real_rate", 0.0)))
        tactical_rate_values.setdefault("oracle_margin", []).append(float(action_metrics.get("tactical_oracle_margin_mean", 0.0)))
        tactical_rate_values.setdefault("counterfactual_match", []).append(float(action_metrics.get("counterfactual_match_rate", 0.0)))
        tactical_rate_values.setdefault("counterfactual_margin", []).append(float(action_metrics.get("counterfactual_margin_mean", 0.0)))
        tactical_rate_values.setdefault("counterfactual_entropy", []).append(float(action_metrics.get("counterfactual_entropy_mean", 0.0)))
        passivity_penalties.append(float(passivity_penalty))
        passive_win_flags.append(1.0 if passive_win else 0.0)
        do_nothing_rates.append(float(do_nothing_rate))
        legal_noop_rates.append(float(legal_noop_rate))
        forced_noop_rates.append(float(forced_noop_rate))
        valid_steps = [
            step for step in trajectory
            if step.get("old_log_prob") is not None and int(step.get("action_idx", -1)) >= 0
        ]
        skipped_missing_old_log_prob += len(trajectory) - len(valid_steps)
        if not valid_steps:
            continue
        step_weight = 1.0 / float(len(valid_steps))
        trajectory_lengths.append(len(valid_steps))
        episode_turn_lengths.append(max(1, int(ep.get("episode_length") or 0)))
        step_summary = summarize_action_records(valid_steps)
        episode_ship_medians.append(float(step_summary.get("median_ships_sent", 0.0)))
        episode_ship_p90s.append(float(step_summary.get("ships_sent_p90", 0.0)))
        episode_ship_maxes.append(float(step_summary.get("ships_sent_max", 0.0)))

        # Per-step shaping is kept local, but the win/loss signal is no longer
        # divided by trajectory length. Each decision trains against:
        #   target_t = episode_outcome + discounted_future_step_shaping_t
        # This keeps terminal outcome on the same scale for long games while
        # still differentiating avoidable no-ops and useful real actions.
        per_step_legal_noop_coef = float(config.get("train_per_step_legal_noop_penalty", 0.020))
        per_step_real_action_coef = float(config.get("train_per_step_real_action_bonus", 0.008))
        per_step_ship_volume_coef = float(config.get("train_per_step_ship_volume_bonus", 0.0))
        if event_shaping_enabled:
            per_step_real_action_coef = min(
                per_step_real_action_coef,
                max(0.0, float(config.get("train_event_max_flat_action_bonus", 0.002))),
            )
            per_step_ship_volume_coef = min(
                per_step_ship_volume_coef,
                max(0.0, float(config.get("train_event_max_ship_volume_bonus", 0.0))),
            )
        event_hit_bonus = max(0.0, float(config.get("train_event_hit_bonus", 0.035)))
        event_enemy_hit_bonus = max(0.0, float(config.get("train_event_enemy_hit_bonus", event_hit_bonus)))
        event_capture_bonus = max(0.0, float(config.get("train_event_capture_bonus", 0.10)))
        event_support_bonus = max(0.0, float(config.get("train_event_support_bonus", 0.015)))
        event_lost_penalty = max(0.0, float(config.get("train_event_lost_penalty", 0.045)))
        event_pending_penalty = max(0.0, float(config.get("train_event_pending_penalty", 0.0)))
        per_step_ship_volume_target = max(1.0, float(config.get("train_per_step_ship_volume_target", 8.0)))
        per_step_shape_clip = float(config.get("train_per_step_shape_clip", 0.04))
        if event_shaping_enabled:
            per_step_shape_clip = max(per_step_shape_clip, float(config.get("train_event_min_shape_clip", 0.12)))
        episode_per_step_shape_total = 0.0
        episode_rows: List[Dict[str, Any]] = []
        for step in valid_steps:
            action_slot = int(step.get("action_slot") or 0)
            mission = str(step.get("mission") or "do_nothing")
            ships = int(step.get("ships") or 0)
            had_real_step = bool(step.get("noop_has_real_candidate", False))
            is_noop_step = mission == "do_nothing" or ships <= 0
            all_ships_sent.append(float(ships))
            if is_noop_step and had_real_step:
                per_step_shape = -per_step_legal_noop_coef  # avoidable no-op
            elif not is_noop_step:
                # The flat/volume terms are capped when event shaping is enabled:
                # successful or failed fleet outcomes carry the real signal.
                volume_bonus = per_step_ship_volume_coef * min(1.0, float(ships) / per_step_ship_volume_target)
                per_step_shape = per_step_real_action_coef + volume_bonus
                event_shape = 0.0
                if event_shaping_enabled:
                    if bool(step.get("fleet_hit", False)):
                        if bool(step.get("fleet_captured", False)):
                            event_shape += event_capture_bonus
                        elif bool(step.get("fleet_supported", False)):
                            event_shape += event_support_bonus
                        elif bool(step.get("fleet_enemy_hit", False)):
                            event_shape += event_enemy_hit_bonus
                        else:
                            event_shape += event_hit_bonus
                    if bool(step.get("fleet_lost", False)):
                        event_shape -= event_lost_penalty
                    if bool(step.get("fleet_pending", False)):
                        event_shape -= event_pending_penalty
                per_step_shape += event_shape
                event_shape_rewards.append(float(event_shape))
                all_real_ships_sent.append(float(ships))
                all_real_ships_by_slot.setdefault(action_slot, []).append(float(ships))
                all_real_ships_by_mission.setdefault(mission, []).append(float(ships))
            else:
                per_step_shape = 0.0  # forced no-op → no signal
            if per_step_shape_clip > 0.0:
                per_step_shape = max(-per_step_shape_clip, min(per_step_shape_clip, per_step_shape))
            episode_per_step_shape_total += per_step_shape
            counterfactual_scores = np.asarray(
                step.get("counterfactual_scores", []),
                dtype=np.float32,
            )
            if counterfactual_scores.shape[:1] != step["candidates"].shape[:1]:
                counterfactual_scores = np.full((int(step["candidates"].shape[0]),), np.nan, dtype=np.float32)
            counterfactual_selected_bonus = _counterfactual_selected_bonus(
                step,
                float(reward),
                float(per_step_shape),
                config,
            )

            episode_rows.append({
                "state": step["state"],
                "candidates": step["candidates"],
                "action_idx": int(step["action_idx"]),
                "old_log_prob": float(step["old_log_prob"]),
                "sample_entropy": float(step.get("entropy") or 0.0),
                "temperature": float(step.get("temperature") or 0.0),
                "policy_version": int(step.get("policy_version") or 0),
                "action_slot": action_slot,
                "reward": 0.0,
                "episode_reward": float(reward),
                "step_shape_reward": float(per_step_shape),
                "step_weight": step_weight,
                "tactical_oracle_idx": int(step.get("tactical_oracle_idx", -1)),
                "tactical_oracle_margin": float(step.get("tactical_oracle_margin", 0.0)),
                "tactical_oracle_tag": str(step.get("tactical_oracle_tag", "unknown")),
                "counterfactual_scores": counterfactual_scores.astype(np.float32, copy=False),
                "counterfactual_selected_bonus": float(counterfactual_selected_bonus),
                "counterfactual_best_idx": int(step.get("counterfactual_best_idx", -1)),
                "counterfactual_margin": float(step.get("counterfactual_margin", 0.0)),
                "counterfactual_entropy": float(step.get("counterfactual_entropy", 0.0)),
            })
            mission = str(step.get("mission") or "do_nothing")
            ships = int(step.get("ships") or 0)
            slot_real_actions.setdefault(action_slot, []).append(1.0 if mission != "do_nothing" and ships > 0 else 0.0)
            slot_noop_actions.setdefault(action_slot, []).append(1.0 if mission == "do_nothing" or ships <= 0 else 0.0)
            noop_before = step.get("noop_prob_before_cap")
            noop_after = step.get("noop_prob_after_cap")
            if noop_before is not None:
                inference_noop_probs_before_cap.setdefault(action_slot, []).append(float(noop_before))
            if noop_after is not None:
                inference_noop_probs_after_cap.setdefault(action_slot, []).append(float(noop_after))
            noop_cap_value = step.get("noop_cap_value")
            if noop_cap_value is not None:
                inference_noop_cap_values.setdefault(action_slot, []).append(float(noop_cap_value))
            has_real_candidate = bool(step.get("noop_has_real_candidate"))
            inference_noop_has_real_candidate.setdefault(action_slot, []).append(1.0 if has_real_candidate else 0.0)
            inference_noop_cap_applied.setdefault(action_slot, []).append(1.0 if step.get("noop_cap_applied") else 0.0)
            if has_real_candidate and noop_before is not None:
                inference_real_noop_probs_before_cap.setdefault(action_slot, []).append(float(noop_before))
            if has_real_candidate and noop_after is not None:
                inference_real_noop_probs_after_cap.setdefault(action_slot, []).append(float(noop_after))
        future_shape_return = 0.0
        for row in reversed(episode_rows):
            future_shape_return = float(row["step_shape_reward"]) + return_gamma * future_shape_return
            return_target = float(row["episode_reward"]) + future_shape_return
            if return_clip > 0.0:
                return_target = max(-return_clip, min(return_clip, return_target))
            row["reward"] = float(return_target)
        dataset.extend(episode_rows)
        per_step_shape_sums.append(float(episode_per_step_shape_total))

    if not dataset:
        return baseline, {"skipped_missing_old_log_prob": float(skipped_missing_old_log_prob)}

    rewards_arr = np.array([d["reward"] for d in dataset], dtype=np.float32)
    value_predictions: List[float] = []
    value_batch_size = max(1, int(config.get("value_bootstrap_batch_size", 4096)))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), value_batch_size):
            batch = dataset[start:start + value_batch_size]
            states = np.stack([d["state"] for d in batch]).astype(np.float32, copy=False)
            state_t = torch.as_tensor(states, dtype=torch.float32, device=device)
            values = model(state_t)["value"].detach().cpu().numpy().astype(np.float32)
            value_predictions.extend(values.tolist())
    value_predictions_arr = np.asarray(value_predictions, dtype=np.float32)
    advantages_arr = rewards_arr - value_predictions_arr
    raw_advantages_arr = advantages_arr.copy()
    adv_std = float(advantages_arr.std()) + advantage_eps
    advantages_arr = (advantages_arr - advantages_arr.mean()) / adv_std

    all_policy_losses: List[float] = []
    all_value_losses: List[float] = []
    all_entropies: List[float] = []
    all_clip_fracs: List[float] = []
    all_total_losses: List[float] = []
    all_approx_kls: List[float] = []
    all_grad_norms: List[float] = []
    all_ratios: List[float] = []
    all_log_ratio_abs: List[float] = []
    all_noop_probs_before_cap: List[float] = []
    all_noop_probs_after_cap: List[float] = []
    all_noop_cap_fracs: List[float] = []
    all_teacher_kls: List[float] = []
    all_on_policy_imitation_losses: List[float] = []
    all_on_policy_oracle_valid_rates: List[float] = []
    all_on_policy_oracle_match_rates: List[float] = []
    all_on_policy_oracle_margins: List[float] = []
    all_counterfactual_imitation_losses: List[float] = []
    all_counterfactual_valid_rates: List[float] = []
    all_counterfactual_margins: List[float] = []
    all_counterfactual_entropies: List[float] = []
    all_counterfactual_top_probs: List[float] = []
    all_counterfactual_policy_match_rates: List[float] = []
    all_counterfactual_selected_probs: List[float] = []
    minibatch_updates = 0

    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    before_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    for _epoch in range(max(1, ppo_epochs)):
        indices = np.random.permutation(len(dataset))
        ep_policy_losses: List[float] = []
        ep_value_losses: List[float] = []
        ep_entropies: List[float] = []
        clipped = 0

        for start in range(0, len(indices), minibatch_size):
            batch_indices = indices[start:start + minibatch_size]
            batch = [dataset[int(i)] for i in batch_indices]
            n_cands = [int(d["candidates"].shape[0]) for d in batch]
            max_n = max(n_cands)
            cand_dim = int(batch[0]["candidates"].shape[-1])

            states = np.stack([d["state"] for d in batch]).astype(np.float32, copy=False)
            cands_padded = np.zeros((len(batch), max_n, cand_dim), dtype=np.float32)
            counterfactual_scores_padded = np.full((len(batch), max_n), np.nan, dtype=np.float32)
            counterfactual_row_valid = np.zeros((len(batch),), dtype=bool)
            mask = np.zeros((len(batch), max_n), dtype=bool)
            for row, (d, n) in enumerate(zip(batch, n_cands)):
                cands_padded[row, :n] = d["candidates"]
                mask[row, :n] = True
                cf_scores = np.asarray(d.get("counterfactual_scores", []), dtype=np.float32)
                if cf_scores.shape[:1] == (n,) and np.isfinite(cf_scores).any():
                    counterfactual_scores_padded[row, :n] = cf_scores[:n]
                    counterfactual_row_valid[row] = True

            state_t = torch.as_tensor(states, dtype=torch.float32, device=device)
            cand_t = torch.as_tensor(cands_padded, dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
            counterfactual_score_t = torch.as_tensor(counterfactual_scores_padded, dtype=torch.float32, device=device)
            counterfactual_row_valid_t = torch.as_tensor(counterfactual_row_valid, dtype=torch.bool, device=device)
            counterfactual_selected_bonus_t = torch.as_tensor(
                [float(d.get("counterfactual_selected_bonus", 0.0)) for d in batch],
                dtype=torch.float32,
                device=device,
            )
            action_t = torch.as_tensor([d["action_idx"] for d in batch], dtype=torch.long, device=device)
            old_lp_t = torch.as_tensor([d["old_log_prob"] for d in batch], dtype=torch.float32, device=device)
            adv_t = torch.as_tensor(advantages_arr[batch_indices], dtype=torch.float32, device=device)
            reward_t = torch.as_tensor([d["reward"] for d in batch], dtype=torch.float32, device=device)
            weight_t = torch.as_tensor([d["step_weight"] for d in batch], dtype=torch.float32, device=device)
            oracle_t = torch.as_tensor([int(d.get("tactical_oracle_idx", -1)) for d in batch], dtype=torch.long, device=device)
            oracle_margin_t = torch.as_tensor(
                [float(d.get("tactical_oracle_margin", 0.0)) for d in batch],
                dtype=torch.float32,
                device=device,
            )
            temp_t = torch.as_tensor(
                [float(d.get("temperature") or 1.0) if float(d.get("temperature") or 0.0) > 0.0 else 1.0 for d in batch],
                dtype=torch.float32,
                device=device,
            )
            action_slot_t = torch.as_tensor([int(d.get("action_slot") or 0) for d in batch], dtype=torch.long, device=device)

            outputs = model(state_t, cand_t)
            logits = outputs["policy_logits"]
            prior_strength = float(config.get("policy_prior_strength", 0.0))
            if prior_strength:
                logits = logits + prior_strength * cand_t[..., -1] * 3.0
            noop_penalty = float(config.get("do_nothing_logit_penalty", 0.0))
            if noop_penalty > 0.0:
                has_real_candidate = mask_t[:, 1:].any(dim=-1) if mask_t.size(-1) > 1 else torch.zeros(mask_t.size(0), dtype=torch.bool, device=device)
                logits[:, 0] = logits[:, 0] - noop_penalty * has_real_candidate.to(dtype=logits.dtype)
            logits = logits.masked_fill(~mask_t, float("-inf"))
            value = outputs["value"]

            action_logits = logits / temp_t.unsqueeze(-1)
            raw_probs = torch.softmax(action_logits, dim=-1)
            noop_cap = caps_for_action_slots(config, action_slot_t)
            if raw_probs.size(-1) > 1:
                has_real_candidate = mask_t[:, 1:].any(dim=-1)
                cap_applies = has_real_candidate & mask_t[:, 0] & (raw_probs[:, 0] > noop_cap.to(device=device, dtype=raw_probs.dtype))
                all_noop_probs_before_cap.extend(raw_probs[:, 0].detach().cpu().numpy().astype(float).tolist())
                all_noop_cap_fracs.append(float(cap_applies.float().mean().item()))
            probs, cap_info = cap_do_nothing_probability_with_info(
                raw_probs,
                mask_t,
                noop_cap,
            )
            if probs.size(-1) > 1:
                all_noop_probs_after_cap.extend(probs[:, 0].detach().cpu().numpy().astype(float).tolist())
            log_probs_all = torch.log(probs.clamp_min(1e-12))
            safe_log_probs = log_probs_all.masked_fill(~mask_t, 0.0)
            entropy_terms = probs * safe_log_probs
            entropy = -entropy_terms.sum(dim=-1)

            new_log_prob = log_probs_all.gather(1, action_t.unsqueeze(-1)).squeeze(-1)
            ratio = torch.exp(new_log_prob - old_lp_t)
            unclipped = ratio * adv_t
            clipped_obj = torch.clamp(ratio, 1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps) * adv_t
            policy_loss_vec = -torch.min(unclipped, clipped_obj) * weight_t
            value_loss_vec = (value - reward_t).pow(2) * weight_t
            entropy_loss_vec = entropy * weight_t

            policy_loss = policy_loss_vec.sum()
            value_loss = value_loss_vec.sum()
            entropy_loss = entropy_loss_vec.sum()
            teacher_kl = torch.zeros((), dtype=torch.float32, device=device)
            if teacher_kl_coef > 0.0 and teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(state_t, cand_t)
                    teacher_logits = teacher_outputs["policy_logits"]
                    if prior_strength:
                        teacher_logits = teacher_logits + prior_strength * cand_t[..., -1] * 3.0
                    if noop_penalty > 0.0:
                        teacher_has_real = (
                            mask_t[:, 1:].any(dim=-1)
                            if mask_t.size(-1) > 1
                            else torch.zeros(mask_t.size(0), dtype=torch.bool, device=device)
                        )
                        teacher_logits[:, 0] = teacher_logits[:, 0] - noop_penalty * teacher_has_real.to(dtype=teacher_logits.dtype)
                    teacher_logits = teacher_logits.masked_fill(~mask_t, float("-inf"))
                    teacher_raw_probs = torch.softmax(teacher_logits / temp_t.unsqueeze(-1), dim=-1)
                    teacher_probs, _ = cap_do_nothing_probability_with_info(
                        teacher_raw_probs,
                        mask_t,
                        noop_cap,
                    )
                    teacher_probs = teacher_probs.nan_to_num(0.0) * mask_t.to(dtype=teacher_probs.dtype)
                    teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    teacher_log_probs = torch.log(teacher_probs.clamp_min(1e-12))
                teacher_kl_vec = (teacher_probs * (teacher_log_probs - log_probs_all)).sum(dim=-1) * weight_t
                teacher_kl = teacher_kl_vec.sum()
                all_teacher_kls.append(float(teacher_kl.detach().item()))

            on_policy_imitation_loss = torch.zeros((), dtype=torch.float32, device=device)
            if on_policy_imitation_coef > 0.0:
                safe_oracle_t = oracle_t.clamp(0, max(0, mask_t.size(1) - 1))
                oracle_in_range = (oracle_t >= 0) & (oracle_t < mask_t.size(1))
                oracle_masked_valid = mask_t.gather(1, safe_oracle_t.unsqueeze(-1)).squeeze(-1)
                oracle_match = oracle_t == action_t
                oracle_confident = oracle_margin_t >= on_policy_imitation_min_margin
                oracle_valid = oracle_in_range & oracle_masked_valid & (oracle_match | oracle_confident)
                all_on_policy_oracle_valid_rates.append(float(oracle_valid.float().mean().detach().item()))
                if oracle_in_range.any():
                    match_rate = (oracle_match & oracle_in_range).float().sum() / oracle_in_range.float().sum().clamp_min(1.0)
                    all_on_policy_oracle_match_rates.append(float(match_rate.detach().item()))
                if oracle_valid.any():
                    oracle_log_prob = log_probs_all.gather(1, safe_oracle_t.unsqueeze(-1)).squeeze(-1)
                    positive_margin = oracle_margin_t.clamp_min(0.0)
                    if on_policy_imitation_min_margin > 0.0:
                        margin_weight = (positive_margin / on_policy_imitation_min_margin).clamp(1.0, on_policy_imitation_max_weight)
                    else:
                        margin_weight = (1.0 + positive_margin).clamp(1.0, on_policy_imitation_max_weight)
                    oracle_weight = weight_t * margin_weight * oracle_valid.to(dtype=weight_t.dtype)
                    on_policy_imitation_loss = (-(oracle_log_prob) * oracle_weight).sum()
                    all_on_policy_imitation_losses.append(float(on_policy_imitation_loss.detach().item()))
                    all_on_policy_oracle_margins.extend(
                        oracle_margin_t[oracle_valid].detach().cpu().numpy().astype(float).tolist()
                    )

            counterfactual_imitation_loss = torch.zeros((), dtype=torch.float32, device=device)
            if counterfactual_imitation_coef > 0.0:
                cf_scores = counterfactual_score_t.clone()
                cf_finite = torch.isfinite(cf_scores)
                cf_scores = torch.where(cf_finite, cf_scores, torch.full_like(cf_scores, -1.0e9))
                cf_scores = cf_scores.masked_fill(~mask_t, -1.0e9)
                safe_action_t = action_t.clamp(0, max(0, mask_t.size(1) - 1))
                action_in_range = (action_t >= 0) & (action_t < mask_t.size(1))
                action_masked_valid = mask_t.gather(1, safe_action_t.unsqueeze(-1)).squeeze(-1)
                selected_valid = action_in_range & action_masked_valid
                if selected_valid.any():
                    cf_scores = cf_scores.scatter_add(
                        1,
                        safe_action_t.unsqueeze(-1),
                        (counterfactual_selected_bonus_t * selected_valid.to(dtype=torch.float32)).unsqueeze(-1),
                    )
                    cf_scores = cf_scores.masked_fill(~mask_t, -1.0e9)

                has_two_candidates = mask_t.sum(dim=-1) >= 2
                if cf_scores.size(1) >= 2:
                    top_values, top_indices = torch.topk(cf_scores, k=2, dim=-1)
                    cf_margin_t = top_values[:, 0] - top_values[:, 1]
                    cf_best_t = top_indices[:, 0]
                else:
                    cf_margin_t = torch.zeros(cf_scores.size(0), dtype=torch.float32, device=device)
                    cf_best_t = torch.zeros(cf_scores.size(0), dtype=torch.long, device=device)
                cf_valid = counterfactual_row_valid_t & has_two_candidates & (cf_margin_t >= counterfactual_min_margin)
                all_counterfactual_valid_rates.append(float(cf_valid.float().mean().detach().item()))

                target_logits = cf_scores / counterfactual_temperature
                target_probs = torch.softmax(target_logits, dim=-1).nan_to_num(0.0)
                target_probs = target_probs * mask_t.to(dtype=target_probs.dtype)
                target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                target_log_probs = torch.log(target_probs.clamp_min(1e-12))
                target_entropy = -(target_probs * target_log_probs).sum(dim=-1)
                target_top_prob = target_probs.gather(1, cf_best_t.unsqueeze(-1)).squeeze(-1)
                policy_best_t = probs.argmax(dim=-1)
                cf_policy_match = policy_best_t == cf_best_t
                selected_target_prob = target_probs.gather(1, safe_action_t.unsqueeze(-1)).squeeze(-1)

                if cf_valid.any():
                    if counterfactual_min_margin > 0.0:
                        cf_weight = (cf_margin_t / counterfactual_min_margin).clamp(1.0, counterfactual_max_weight)
                    else:
                        cf_weight = (1.0 + cf_margin_t.clamp_min(0.0)).clamp(1.0, counterfactual_max_weight)
                    counterfactual_kl = (target_probs * (target_log_probs - log_probs_all)).sum(dim=-1)
                    counterfactual_imitation_loss = (
                        counterfactual_kl
                        * weight_t
                        * cf_weight
                        * cf_valid.to(dtype=weight_t.dtype)
                    ).sum()
                    all_counterfactual_imitation_losses.append(float(counterfactual_imitation_loss.detach().item()))
                    all_counterfactual_margins.extend(cf_margin_t[cf_valid].detach().cpu().numpy().astype(float).tolist())
                    all_counterfactual_entropies.extend(target_entropy[cf_valid].detach().cpu().numpy().astype(float).tolist())
                    all_counterfactual_top_probs.extend(target_top_prob[cf_valid].detach().cpu().numpy().astype(float).tolist())
                    all_counterfactual_policy_match_rates.append(float(cf_policy_match[cf_valid].float().mean().detach().item()))
                    all_counterfactual_selected_probs.extend(selected_target_prob[cf_valid].detach().cpu().numpy().astype(float).tolist())

            loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy_loss
                + teacher_kl_coef * teacher_kl
                + on_policy_imitation_coef * on_policy_imitation_loss
                + counterfactual_imitation_coef * counterfactual_imitation_loss
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            minibatch_updates += 1

            ep_policy_losses.append(float(policy_loss.item()))
            ep_value_losses.append(float(value_loss.item()))
            ep_entropies.append(float(entropy.mean().item()))
            all_total_losses.append(float(loss.item()))
            all_grad_norms.append(float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm))
            log_ratio = new_log_prob.detach() - old_lp_t
            approx_kl = (torch.exp(log_ratio) - 1.0 - log_ratio).detach()
            all_approx_kls.extend(approx_kl.cpu().numpy().astype(float).tolist())
            all_ratios.extend(ratio.detach().cpu().numpy().astype(float).tolist())
            all_log_ratio_abs.extend(log_ratio.abs().cpu().numpy().astype(float).tolist())
            clipped += int((ratio.detach() - 1.0).abs().gt(ppo_clip_eps).sum().item())

        if ep_policy_losses:
            all_policy_losses.append(float(np.mean(ep_policy_losses)))
            all_value_losses.append(float(np.mean(ep_value_losses)))
            all_entropies.append(float(np.mean(ep_entropies)))
            all_clip_fracs.append(clipped / max(1, len(indices)))

    model.eval()

    delta_sq = 0.0
    param_sq = 0.0
    param_count = 0
    with torch.no_grad():
        for before, after in zip(before_params, (p for p in model.parameters() if p.requires_grad)):
            before_cpu = before.detach().float().cpu()
            after_cpu = after.detach().float().cpu()
            diff = after_cpu - before_cpu
            delta_sq += float(diff.pow(2).sum().item())
            param_sq += float(before_cpu.pow(2).sum().item())
            param_count += int(diff.numel())
    param_delta_rms = float(np.sqrt(delta_sq / max(1, param_count)))
    param_relative_delta = float(np.sqrt(delta_sq / max(1e-12, param_sq)))

    # Update baseline (running mean of episode rewards)
    for r in rewards:
        baseline = (1.0 - baseline_momentum) * baseline + baseline_momentum * r

    total_real_actions = float(sum(float(np.sum(values)) for values in slot_real_actions.values()))
    total_turns = float(sum(episode_turn_lengths))

    def _slot_mean(values_by_slot: Dict[int, List[float]], slot: int, default: float = 0.0) -> float:
        values = values_by_slot.get(slot)
        return float(np.mean(values)) if values else default

    def _quantile(values: List[float], q: float, default: float = 0.0) -> float:
        return float(np.quantile(np.asarray(values, dtype=np.float32), q)) if values else default

    metrics = {
        "policy_loss": float(np.mean(all_policy_losses)) if all_policy_losses else 0.0,
        "value_loss": float(np.mean(all_value_losses)) if all_value_losses else 0.0,
        "total_loss": float(np.mean(all_total_losses)) if all_total_losses else 0.0,
        "on_policy_imitation_loss": float(np.mean(all_on_policy_imitation_losses)) if all_on_policy_imitation_losses else 0.0,
        "on_policy_imitation_coef": float(on_policy_imitation_coef),
        "on_policy_oracle_valid_rate": float(np.mean(all_on_policy_oracle_valid_rates)) if all_on_policy_oracle_valid_rates else 0.0,
        "on_policy_oracle_match_rate": float(np.mean(all_on_policy_oracle_match_rates)) if all_on_policy_oracle_match_rates else 0.0,
        "on_policy_oracle_margin_mean": float(np.mean(all_on_policy_oracle_margins)) if all_on_policy_oracle_margins else 0.0,
        "counterfactual_imitation_loss": float(np.mean(all_counterfactual_imitation_losses)) if all_counterfactual_imitation_losses else 0.0,
        "counterfactual_imitation_coef": float(counterfactual_imitation_coef),
        "counterfactual_valid_rate": float(np.mean(all_counterfactual_valid_rates)) if all_counterfactual_valid_rates else 0.0,
        "counterfactual_margin_mean": float(np.mean(all_counterfactual_margins)) if all_counterfactual_margins else 0.0,
        "counterfactual_entropy_mean": float(np.mean(all_counterfactual_entropies)) if all_counterfactual_entropies else 0.0,
        "counterfactual_top_prob_mean": float(np.mean(all_counterfactual_top_probs)) if all_counterfactual_top_probs else 0.0,
        "counterfactual_policy_match_rate": float(np.mean(all_counterfactual_policy_match_rates)) if all_counterfactual_policy_match_rates else 0.0,
        "counterfactual_selected_prob_mean": float(np.mean(all_counterfactual_selected_probs)) if all_counterfactual_selected_probs else 0.0,
        "entropy": float(np.mean(all_entropies)) if all_entropies else 0.0,
        "clip_frac": float(np.mean(all_clip_fracs)) if all_clip_fracs else 0.0,
        "approx_kl": float(np.mean(all_approx_kls)) if all_approx_kls else 0.0,
        "grad_norm": float(np.mean(all_grad_norms)) if all_grad_norms else 0.0,
        "ratio_mean": float(np.mean(all_ratios)) if all_ratios else 0.0,
        "ratio_std": float(np.std(all_ratios)) if all_ratios else 0.0,
        "ratio_min": float(np.min(all_ratios)) if all_ratios else 0.0,
        "ratio_max": float(np.max(all_ratios)) if all_ratios else 0.0,
        "log_ratio_abs_max": float(np.max(all_log_ratio_abs)) if all_log_ratio_abs else 0.0,
        "param_delta_rms": param_delta_rms,
        "param_relative_delta": param_relative_delta,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "median_reward": float(np.median(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "return_target_mean": float(np.mean(rewards_arr)) if rewards_arr.size else 0.0,
        "return_target_std": float(np.std(rewards_arr)) if rewards_arr.size else 0.0,
        "value_prediction_mean": float(np.mean(value_predictions_arr)) if value_predictions_arr.size else 0.0,
        "raw_advantage_mean": float(np.mean(raw_advantages_arr)) if raw_advantages_arr.size else 0.0,
        "raw_advantage_std": float(np.std(raw_advantages_arr)) if raw_advantages_arr.size else 0.0,
        "train_return_gamma": float(return_gamma),
        "train_return_clip": float(return_clip),
        "terminal_reward_mean": float(np.mean(terminal_rewards)) if terminal_rewards else 0.0,
        "adjusted_terminal_reward_mean": float(np.mean(adjusted_terminal_rewards)) if adjusted_terminal_rewards else 0.0,
        "dense_reward_mean": float(np.mean(dense_rewards)) if dense_rewards else 0.0,
        "activity_reward_mean": float(np.mean(activity_rewards)) if activity_rewards else 0.0,
        "effective_activity_action_bonus": float(effective_activity_action_bonus),
        "effective_activity_ship_bonus": float(effective_activity_ship_bonus),
        "mission_mix_reward_mean": float(np.mean(mission_mix_rewards)) if mission_mix_rewards else 0.0,
        "mission_expand_ratio_mean": float(np.mean(mission_expand_ratios)) if mission_expand_ratios else 0.0,
        "mission_attack_ratio_mean": float(np.mean(mission_attack_ratios)) if mission_attack_ratios else 0.0,
        "mission_support_ratio_mean": float(np.mean(mission_support_ratios)) if mission_support_ratios else 0.0,
        "event_shaping_enabled": float(1.0 if event_shaping_enabled else 0.0),
        "event_shape_reward_mean": float(np.mean(event_shape_rewards)) if event_shape_rewards else 0.0,
        "fleet_launch_mapped_rate_mean": float(np.mean(fleet_launch_mapped_rates)) if fleet_launch_mapped_rates else 0.0,
        "fleet_outcome_known_rate_mean": float(np.mean(fleet_outcome_known_rates)) if fleet_outcome_known_rates else 0.0,
        "fleet_hit_rate_mean": float(np.mean(fleet_hit_rates)) if fleet_hit_rates else 0.0,
        "fleet_enemy_hit_rate_mean": float(np.mean(fleet_enemy_hit_rates)) if fleet_enemy_hit_rates else 0.0,
        "fleet_neutral_hit_rate_mean": float(np.mean(fleet_neutral_hit_rates)) if fleet_neutral_hit_rates else 0.0,
        "fleet_support_rate_mean": float(np.mean(fleet_support_rates)) if fleet_support_rates else 0.0,
        "fleet_capture_rate_mean": float(np.mean(fleet_capture_rates)) if fleet_capture_rates else 0.0,
        "fleet_lost_rate_mean": float(np.mean(fleet_lost_rates)) if fleet_lost_rates else 0.0,
        "fleet_lost_sun_rate_mean": float(np.mean(fleet_lost_sun_rates)) if fleet_lost_sun_rates else 0.0,
        "fleet_lost_oob_rate_mean": float(np.mean(fleet_lost_oob_rates)) if fleet_lost_oob_rates else 0.0,
        "fleet_pending_rate_mean": float(np.mean(fleet_pending_rates)) if fleet_pending_rates else 0.0,
        "tactical_support_defense_rate_mean": float(np.mean(tactical_rate_values.get("support_defense", []))) if tactical_rate_values.get("support_defense") else 0.0,
        "tactical_support_front_rate_mean": float(np.mean(tactical_rate_values.get("support_front", []))) if tactical_rate_values.get("support_front") else 0.0,
        "tactical_support_redistribute_rate_mean": float(np.mean(tactical_rate_values.get("support_redistribute", []))) if tactical_rate_values.get("support_redistribute") else 0.0,
        "tactical_support_passive_rate_mean": float(np.mean(tactical_rate_values.get("support_passive", []))) if tactical_rate_values.get("support_passive") else 0.0,
        "tactical_support_backward_rate_mean": float(np.mean(tactical_rate_values.get("support_backward", []))) if tactical_rate_values.get("support_backward") else 0.0,
        "tactical_attack_convert_rate_mean": float(np.mean(tactical_rate_values.get("attack_convert", []))) if tactical_rate_values.get("attack_convert") else 0.0,
        "tactical_attack_opportunity_rate_mean": float(np.mean(tactical_rate_values.get("attack_opportunity", []))) if tactical_rate_values.get("attack_opportunity") else 0.0,
        "tactical_attack_pressure_rate_mean": float(np.mean(tactical_rate_values.get("attack_pressure", []))) if tactical_rate_values.get("attack_pressure") else 0.0,
        "tactical_attack_poor_rate_mean": float(np.mean(tactical_rate_values.get("attack_poor", []))) if tactical_rate_values.get("attack_poor") else 0.0,
        "candidate_attack_convert_rate_mean": float(np.mean(tactical_rate_values.get("candidate_attack_convert", []))) if tactical_rate_values.get("candidate_attack_convert") else 0.0,
        "candidate_attack_pressure_rate_mean": float(np.mean(tactical_rate_values.get("candidate_attack_pressure", []))) if tactical_rate_values.get("candidate_attack_pressure") else 0.0,
        "attack_convert_missed_rate_mean": float(np.mean(tactical_rate_values.get("attack_convert_missed", []))) if tactical_rate_values.get("attack_convert_missed") else 0.0,
        "good_attack_missed_rate_mean": float(np.mean(tactical_rate_values.get("good_attack_missed", []))) if tactical_rate_values.get("good_attack_missed") else 0.0,
        "tactical_oracle_match_rate_mean": float(np.mean(tactical_rate_values.get("oracle_match", []))) if tactical_rate_values.get("oracle_match") else 0.0,
        "tactical_oracle_real_rate_mean": float(np.mean(tactical_rate_values.get("oracle_real", []))) if tactical_rate_values.get("oracle_real") else 0.0,
        "tactical_oracle_margin_mean": float(np.mean(tactical_rate_values.get("oracle_margin", []))) if tactical_rate_values.get("oracle_margin") else 0.0,
        "action_counterfactual_match_rate_mean": float(np.mean(tactical_rate_values.get("counterfactual_match", []))) if tactical_rate_values.get("counterfactual_match") else 0.0,
        "action_counterfactual_margin_mean": float(np.mean(tactical_rate_values.get("counterfactual_margin", []))) if tactical_rate_values.get("counterfactual_margin") else 0.0,
        "action_counterfactual_entropy_mean": float(np.mean(tactical_rate_values.get("counterfactual_entropy", []))) if tactical_rate_values.get("counterfactual_entropy") else 0.0,
        "tactical_expand_front_rate_mean": float(np.mean(tactical_rate_values.get("expand_front", []))) if tactical_rate_values.get("expand_front") else 0.0,
        "tactical_expand_safe_rate_mean": float(np.mean(tactical_rate_values.get("expand_safe", []))) if tactical_rate_values.get("expand_safe") else 0.0,
        "passivity_penalty_mean": float(np.mean(passivity_penalties)) if passivity_penalties else 0.0,
        "passive_win_rate": float(np.mean(passive_win_flags)) if passive_win_flags else 0.0,
        "do_nothing_rate_mean": float(np.mean(do_nothing_rates)) if do_nothing_rates else 1.0,
        "legal_noop_rate_mean": float(np.mean(legal_noop_rates)) if legal_noop_rates else 0.0,
        "forced_noop_rate_mean": float(np.mean(forced_noop_rates)) if forced_noop_rates else 0.0,
        "per_step_shape_sum_mean": float(np.mean(per_step_shape_sums)) if per_step_shape_sums else 0.0,
        "noop_prob_before_cap_mean": float(np.mean(all_noop_probs_before_cap)) if all_noop_probs_before_cap else 0.0,
        "noop_prob_after_cap_mean": float(np.mean(all_noop_probs_after_cap)) if all_noop_probs_after_cap else 0.0,
        "noop_prob_cap_frac": float(np.mean(all_noop_cap_fracs)) if all_noop_cap_fracs else 0.0,
        "ships_sent_mean_all_steps": float(np.mean(all_ships_sent)) if all_ships_sent else 0.0,
        "ships_sent_median": _quantile(all_real_ships_sent, 0.50),
        "ships_sent_p25": _quantile(all_real_ships_sent, 0.25),
        "ships_sent_p75": _quantile(all_real_ships_sent, 0.75),
        "ships_sent_p90": _quantile(all_real_ships_sent, 0.90),
        "ships_sent_max": float(np.max(all_real_ships_sent)) if all_real_ships_sent else 0.0,
        "episode_ships_median_mean": float(np.mean(episode_ship_medians)) if episode_ship_medians else 0.0,
        "episode_ships_p90_mean": float(np.mean(episode_ship_p90s)) if episode_ship_p90s else 0.0,
        "episode_ships_max_mean": float(np.mean(episode_ship_maxes)) if episode_ship_maxes else 0.0,
        "slot0_ships_mean": _slot_mean(all_real_ships_by_slot, 0),
        "slot0_ships_p90": _quantile(all_real_ships_by_slot.get(0, []), 0.90),
        "slot1_ships_mean": _slot_mean(all_real_ships_by_slot, 1),
        "slot1_ships_p90": _quantile(all_real_ships_by_slot.get(1, []), 0.90),
        "mission_expand_ships_mean": float(np.mean(all_real_ships_by_mission.get("expand", []))) if all_real_ships_by_mission.get("expand") else 0.0,
        "mission_attack_ships_mean": float(np.mean(all_real_ships_by_mission.get("attack", []))) if all_real_ships_by_mission.get("attack") else 0.0,
        "mission_support_ships_mean": float(np.mean(all_real_ships_by_mission.get("support", []))) if all_real_ships_by_mission.get("support") else 0.0,
        "first_slot_noop_rate": float(np.mean(slot_noop_actions.get(0, [1.0]))),
        "first_slot_real_action_rate": float(np.mean(slot_real_actions.get(0, [0.0]))),
        "mean_real_actions_per_game": float(total_real_actions / max(1, len(episodes))),
        "mean_real_actions_per_turn": float(total_real_actions / max(1.0, total_turns)),
        "slot0_noop_prob_before_cap": _slot_mean(inference_noop_probs_before_cap, 0),
        "slot0_noop_prob_after_cap": _slot_mean(inference_noop_probs_after_cap, 0),
        "slot0_noop_cap_value": _slot_mean(inference_noop_cap_values, 0, 1.0),
        "slot0_has_real_candidate": _slot_mean(inference_noop_has_real_candidate, 0),
        "slot0_real_noop_prob_before_cap": _slot_mean(inference_real_noop_probs_before_cap, 0),
        "slot0_real_noop_prob_after_cap": _slot_mean(inference_real_noop_probs_after_cap, 0),
        "slot0_noop_cap_frac": _slot_mean(inference_noop_cap_applied, 0),
        "slot1_noop_prob_before_cap": _slot_mean(inference_noop_probs_before_cap, 1),
        "slot1_noop_prob_after_cap": _slot_mean(inference_noop_probs_after_cap, 1),
        "slot1_noop_cap_value": _slot_mean(inference_noop_cap_values, 1, 1.0),
        "slot1_has_real_candidate": _slot_mean(inference_noop_has_real_candidate, 1),
        "slot1_real_noop_prob_before_cap": _slot_mean(inference_real_noop_probs_before_cap, 1),
        "slot1_real_noop_prob_after_cap": _slot_mean(inference_real_noop_probs_after_cap, 1),
        "slot1_noop_cap_frac": _slot_mean(inference_noop_cap_applied, 1),
        "mean_win": float(np.mean([ep["win"] for ep in episodes])),
        "skipped_missing_old_log_prob": float(skipped_missing_old_log_prob),
        "mean_trajectory_len": float(np.mean(trajectory_lengths)) if trajectory_lengths else 0.0,
        "mean_sample_temperature": float(np.mean([d["temperature"] for d in dataset])) if dataset else 0.0,
        "mean_sample_entropy": float(np.mean([d["sample_entropy"] for d in dataset])) if dataset else 0.0,
        "policy_version": float(max(d["policy_version"] for d in dataset)) if dataset else 0.0,
        "train_samples": float(len(dataset)),
        "train_minibatches": float(minibatch_updates),
        "ppo_minibatch_size": float(minibatch_size),
        "teacher_kl": float(np.mean(all_teacher_kls)) if all_teacher_kls else 0.0,
        "teacher_kl_coef": float(teacher_kl_coef),
    }
    return baseline, metrics
