from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_network.src.encoder import encode_game_state
from neural_network.src.orbit_wars_adapter import obs_to_game_dict
from neural_network.src.policy import build_action_candidates, reconstruct_action
from neural_network.src.model import ModelConfig, NeuralNetworkModel, load_compatible_state_dict
from neural_network.src.storage import load_checkpoint
from neural_network.src.notebook_4p_training import (
    _agent_for_name,
    _candidate_move,
    _copy_planning_game,
    _infer_input_dim,
    _make_model_agent,
    _reserve_planned_ships,
    _sample_opponents,
    _send_ratios,
    _min_expand_attack_ships,
    run_match,
)
from neural_network.src.population_4p_training import _strategic_dense_reward
from neural_network_gpu.src.action_metrics import summarize_action_records


_TACTICAL_ORACLE_PRIORITY = {
    "attack_convert": 9.0,
    "attack_pressure": 6.8,
    "support_defense": 6.4,
    "attack_opportunity": 5.6,
    "expand_front": 5.0,
    "support_redistribute": 4.6,
    "support_front": 4.2,
    "expand_safe": 2.6,
    "attack_poor": -1.0,
    "support_backward": -1.5,
    "support_passive": -2.0,
    "noop": 0.0,
    "do_nothing": 0.0,
}


def _candidate_oracle_score(candidate: Any, has_real_candidate: bool) -> float:
    mission = str(getattr(candidate, "mission", "do_nothing") or "do_nothing")
    tag = str(getattr(candidate, "tactical_tag", mission) or mission)
    score = float(_TACTICAL_ORACLE_PRIORITY.get(tag, _TACTICAL_ORACLE_PRIORITY.get(mission, 0.0)))
    score += 0.35 * float(getattr(candidate, "tactical_score", 0.0) or 0.0)
    try:
        score += 0.20 * float(candidate.score_features[-1]) * 3.0
    except Exception:
        pass
    amount = int(getattr(candidate, "amount", 0) or 0)
    if mission in {"do_nothing", "noop"}:
        return score - (3.0 if has_real_candidate else 0.0)
    if amount <= 0:
        score -= 4.0
    elif amount <= 2 and mission in {"attack", "expand"}:
        score -= 0.5
    return float(score)


def _tactical_oracle(candidates: List[Any]) -> tuple[int, float, str]:
    if not candidates:
        return -1, 0.0, "none"
    has_real_candidate = len(candidates) > 1
    best_idx = 0
    best_score = float("-inf")
    best_tag = "noop"
    for idx, candidate in enumerate(candidates):
        score = _candidate_oracle_score(candidate, has_real_candidate)
        tag = str(getattr(candidate, "tactical_tag", getattr(candidate, "mission", "unknown")) or "unknown")
        if score > best_score:
            best_idx = int(idx)
            best_score = float(score)
            best_tag = tag
    return best_idx, best_score, best_tag


def _counterfactual_candidate_scores(candidates: List[Any], config: Dict[str, Any]) -> tuple[np.ndarray, int, float, float]:
    """Score every legal candidate as a soft policy-improvement target.

    This is only a training target. Runtime inference still uses the neural
    network logits. The score deliberately ranks all alternatives instead of
    turning the best action into a brittle one-hot label.
    """
    if not candidates:
        return np.zeros((0,), dtype=np.float32), -1, 0.0, 0.0

    has_real_candidate = len(candidates) > 1
    prior_weight = float(config.get("counterfactual_prior_weight", 0.35))
    tactical_weight = float(config.get("counterfactual_tactical_weight", 1.0))
    oracle_scale = float(config.get("counterfactual_oracle_scale", 0.45))
    amount_weight = float(config.get("counterfactual_amount_weight", 0.20))
    distance_penalty = float(config.get("counterfactual_distance_penalty", 0.10))
    attack_bonus = float(config.get("counterfactual_attack_bonus", 0.10))
    attack_convert_bonus = float(config.get("counterfactual_attack_convert_bonus", 1.20))
    attack_pressure_bonus = float(config.get("counterfactual_attack_pressure_bonus", 0.70))
    attack_opportunity_bonus = float(config.get("counterfactual_attack_opportunity_bonus", 0.45))
    attack_poor_penalty = float(config.get("counterfactual_attack_poor_penalty", 0.75))
    good_attack_compete_penalty = float(config.get("counterfactual_good_attack_compete_penalty", 0.35))
    expand_bonus = float(config.get("counterfactual_expand_bonus", -0.05))
    expand_front_bonus = float(config.get("counterfactual_expand_front_bonus", 0.10))
    expand_safe_penalty = float(config.get("counterfactual_expand_safe_penalty", 0.25))
    support_front_bonus = float(config.get("counterfactual_support_front_bonus", 0.20))
    support_defense_bonus = float(config.get("counterfactual_support_defense_bonus", 0.85))
    support_redistribute_bonus = float(config.get("counterfactual_support_redistribute_bonus", 0.50))
    support_passive_penalty = float(config.get("counterfactual_support_passive_penalty", 0.65))
    support_backward_penalty = float(config.get("counterfactual_support_backward_penalty", 0.45))
    noop_penalty = float(config.get("counterfactual_noop_penalty", 2.0))
    temperature = max(1e-3, float(config.get("counterfactual_temperature", 0.80)))
    candidate_tags = [str(getattr(candidate, "tactical_tag", getattr(candidate, "mission", "unknown")) or "unknown") for candidate in candidates]
    has_attack_convert = any(tag == "attack_convert" for tag in candidate_tags)
    has_good_attack = has_attack_convert or any(tag == "attack_pressure" for tag in candidate_tags)

    scores: List[float] = []
    for candidate in candidates:
        mission = str(getattr(candidate, "mission", "do_nothing") or "do_nothing")
        tag = str(getattr(candidate, "tactical_tag", mission) or mission)
        features = np.asarray(getattr(candidate, "score_features", np.zeros(16, dtype=np.float32)), dtype=np.float32)
        amount = float(getattr(candidate, "amount", 0.0) or 0.0)
        score = tactical_weight * oracle_scale * _candidate_oracle_score(candidate, has_real_candidate)

        if features.size >= 16:
            sent_ratio = float(features[2])
            distance = float(features[4])
            prior = float(features[-1]) * 3.0
            score += prior_weight * prior
            if mission in {"attack", "expand"}:
                score += amount_weight * sent_ratio
                score -= distance_penalty * distance
            elif mission == "support":
                score += 0.5 * amount_weight * sent_ratio
                score -= 0.4 * distance_penalty * distance

        if mission == "attack":
            score += attack_bonus
            if tag == "attack_convert":
                score += attack_convert_bonus
            elif tag == "attack_pressure":
                score += attack_pressure_bonus
            elif tag == "attack_opportunity":
                score += attack_opportunity_bonus
            elif tag == "attack_poor":
                score -= attack_poor_penalty
        elif mission == "expand":
            score += expand_bonus
            if tag == "expand_front":
                score += expand_front_bonus
            elif tag == "expand_safe":
                score -= expand_safe_penalty
        elif mission == "support":
            if tag == "support_defense":
                score += support_defense_bonus
            elif tag == "support_redistribute":
                score += support_redistribute_bonus
            elif tag == "support_front":
                score += support_front_bonus
            elif tag == "support_passive":
                score -= support_passive_penalty
            elif tag == "support_backward":
                score -= support_backward_penalty
        elif mission in {"do_nothing", "noop"} and has_real_candidate:
            score -= noop_penalty

        if has_attack_convert and tag != "attack_convert":
            score -= 0.50 * good_attack_compete_penalty
        elif has_good_attack and mission != "attack":
            score -= good_attack_compete_penalty
        if amount <= 0.0 and mission not in {"do_nothing", "noop"}:
            score -= 3.0
        scores.append(float(score))

    score_arr = np.asarray(scores, dtype=np.float32)
    best_idx = int(np.argmax(score_arr)) if score_arr.size else -1
    sorted_scores = np.sort(score_arr)
    margin = float(sorted_scores[-1] - sorted_scores[-2]) if sorted_scores.size >= 2 else 0.0
    logits = (score_arr - float(np.max(score_arr))) / temperature
    probs = np.exp(logits).astype(np.float64)
    probs /= max(1e-12, float(np.sum(probs)))
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12)))) if probs.size else 0.0
    return score_arr, best_idx, margin, entropy


def _agent_for_pool_name(name: str, config: Dict[str, Any], cache: Dict[str, Any]):
    if isinstance(name, str) and name.startswith("checkpoint:"):
        path = name.split(":", 1)[1]
        if path not in cache:
            checkpoint_path = Path(path)
            if not checkpoint_path.exists():
                cache[path] = _agent_for_name("random")
            else:
                model = NeuralNetworkModel(ModelConfig(
                    input_dim=_infer_input_dim(config),
                    hidden_dim=int(config.get("hidden_dim", 320)),
                ))
                state, _ = load_checkpoint(checkpoint_path)
                load_compatible_state_dict(model, state)
                model.eval()
                cache[path] = _make_model_agent(model, config, temperature=0.0, explore=False)
        return cache[path]
    return _agent_for_name(name)


def _event_type(event: Any) -> str:
    return str(event.get("type", "") if isinstance(event, dict) else getattr(event, "type", ""))


def _event_get(event: Any, key: str, default: Any = None) -> Any:
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)


def _attach_fleet_outcomes(
    trajectory: List[Dict[str, Any]],
    fleet_events: List[Dict[str, Any]],
    player: int,
) -> None:
    """Map official fleet events back to the policy decisions that launched them."""
    if not trajectory:
        return

    launch_events = [
        event for event in fleet_events
        if _event_type(event) == "launch" and int(_event_get(event, "player", -1)) == int(player)
    ]
    hit_by_fleet = {
        int(_event_get(event, "fleet_id", -1)): event
        for event in fleet_events
        if _event_type(event) == "hit"
    }
    loss_by_fleet = {
        int(_event_get(event, "fleet_id", -1)): event
        for event in fleet_events
        if _event_type(event) in {"lost_oob", "lost_sun"}
    }
    combat_by_turn_target = {
        (int(_event_get(event, "turn", -1)), int(_event_get(event, "target_id", -1))): event
        for event in fleet_events
        if _event_type(event) == "combat"
    }

    remaining_launches = list(launch_events)
    for step in trajectory:
        mission = str(step.get("mission") or "do_nothing")
        ships = int(step.get("ships") or 0)
        if mission == "do_nothing" or ships <= 0:
            continue

        step.update({
            "fleet_launch_mapped": False,
            "fleet_outcome_known": False,
            "fleet_hit": False,
            "fleet_lost": False,
            "fleet_lost_sun": False,
            "fleet_lost_oob": False,
            "fleet_pending": False,
            "fleet_captured": False,
            "fleet_supported": False,
            "fleet_enemy_hit": False,
            "fleet_neutral_hit": False,
            "fleet_target_owner_before": -2,
            "fleet_target_owner_after": -2,
            "fleet_id": -1,
            "hit_target_id": -1,
        })

        source_id = int(step.get("source_id", -1))
        match_idx = -1
        for idx, event in enumerate(remaining_launches):
            if int(_event_get(event, "source_id", -2)) == source_id and int(_event_get(event, "ships", -1)) == ships:
                match_idx = idx
                break
        if match_idx < 0 and remaining_launches:
            match_idx = 0
        if match_idx < 0:
            continue

        launch = remaining_launches.pop(match_idx)
        fleet_id = int(_event_get(launch, "fleet_id", -1))
        step["fleet_id"] = fleet_id
        step["fleet_launch_mapped"] = True

        hit = hit_by_fleet.get(fleet_id)
        loss = loss_by_fleet.get(fleet_id)
        if hit is not None:
            turn = int(_event_get(hit, "turn", -1))
            target_id = int(_event_get(hit, "target_id", -1))
            target_owner_before = int(_event_get(hit, "target_owner_before", -2))
            combat = combat_by_turn_target.get((turn, target_id))
            target_owner_after = (
                int(_event_get(combat, "owner_after", target_owner_before))
                if combat is not None
                else target_owner_before
            )
            step.update({
                "fleet_outcome_known": True,
                "fleet_hit": True,
                "hit_target_id": target_id,
                "fleet_target_owner_before": target_owner_before,
                "fleet_target_owner_after": target_owner_after,
                "fleet_captured": target_owner_before != int(player) and target_owner_after == int(player),
                "fleet_supported": target_owner_before == int(player) and target_owner_after == int(player),
                "fleet_enemy_hit": target_owner_before not in (-1, int(player)),
                "fleet_neutral_hit": target_owner_before == -1,
            })
        elif loss is not None:
            loss_type = _event_type(loss)
            step.update({
                "fleet_outcome_known": True,
                "fleet_lost": True,
                "fleet_lost_sun": loss_type == "lost_sun",
                "fleet_lost_oob": loss_type == "lost_oob",
            })
        else:
            step["fleet_pending"] = True


def _make_gpu_agent(
    worker_id: int,
    obs_queue: mp.Queue,
    action_queue: mp.Queue,
    config: Dict[str, Any],
    trajectory: List[Dict[str, Any]],
):
    ratios = _send_ratios(config)
    min_ships = _min_expand_attack_ships(config)
    max_actions = max(1, int(config.get("max_actions_per_turn", 4)))

    def agent(obs, _config=None):
        game = obs_to_game_dict(obs)
        planning_game = _copy_planning_game(game)
        moves: List = []

        for action_slot in range(max_actions):
            encoded = encode_game_state(planning_game, config)
            candidates = build_action_candidates(
                planning_game,
                send_ratios=ratios,
                min_expand_attack_ships=min_ships,
                allow_support=bool(config.get("allow_support_actions", True)),
            )
            if not candidates:
                break
            candidate_tactical_tags = [str(getattr(c, "tactical_tag", c.mission)) for c in candidates]
            candidate_has_attack_convert = any(tag == "attack_convert" for tag in candidate_tactical_tags)
            candidate_has_attack_pressure = any(tag == "attack_pressure" for tag in candidate_tactical_tags)
            candidate_has_good_attack = candidate_has_attack_convert or candidate_has_attack_pressure
            tactical_oracle_idx, tactical_oracle_score, tactical_oracle_tag = _tactical_oracle(candidates)
            counterfactual_scores, counterfactual_best_idx, counterfactual_margin, counterfactual_entropy = (
                _counterfactual_candidate_scores(candidates, config)
            )

            state_features = np.array(encoded.features, dtype=np.float32)
            cand_features = np.stack([c.score_features for c in candidates]).astype(np.float32)

            obs_queue.put({
                "worker_id": worker_id,
                "state": state_features,
                "candidates": cand_features,
                "n_candidates": len(candidates),
                "action_slot": action_slot,
            })

            action_msg = action_queue.get()
            if isinstance(action_msg, dict):
                action_idx = int(action_msg.get("action_idx", -1))
                old_log_prob = action_msg.get("old_log_prob")
                entropy = action_msg.get("entropy")
                temperature = action_msg.get("temperature")
                policy_version = action_msg.get("policy_version")
                noop_prob_before_cap = action_msg.get("noop_prob_before_cap")
                noop_prob_after_cap = action_msg.get("noop_prob_after_cap")
                noop_cap_value = action_msg.get("noop_cap_value")
                noop_has_real_candidate = action_msg.get("noop_has_real_candidate")
                noop_cap_applied = action_msg.get("noop_cap_applied")
            else:
                action_idx = int(action_msg)
                old_log_prob = None
                entropy = None
                temperature = None
                policy_version = None
                noop_prob_before_cap = None
                noop_prob_after_cap = None
                noop_cap_value = None
                noop_has_real_candidate = False
                noop_cap_applied = False

            if action_idx < 0 or action_idx >= len(candidates):
                trajectory.append({
                    "state": state_features,
                    "candidates": cand_features,
                    "action_idx": -1,
                    "mission": "do_nothing",
                    "ships": 0,
                    "source_id": -1,
                    "target_id": -1,
                    "planned_angle": None,
                    "old_log_prob": old_log_prob,
                    "entropy": entropy,
                    "temperature": temperature,
                    "policy_version": policy_version,
                    "action_slot": action_slot,
                    "noop_prob_before_cap": noop_prob_before_cap,
                    "noop_prob_after_cap": noop_prob_after_cap,
                    "noop_cap_value": noop_cap_value,
                    "noop_has_real_candidate": noop_has_real_candidate,
                    "noop_cap_applied": noop_cap_applied,
                    "tactical_tag": "noop_invalid",
                    "tactical_score": 0.0,
                    "tactical_oracle_idx": int(tactical_oracle_idx),
                    "tactical_oracle_score": float(tactical_oracle_score),
                    "tactical_oracle_tag": str(tactical_oracle_tag),
                    "tactical_oracle_margin": float(tactical_oracle_score),
                    "counterfactual_scores": counterfactual_scores,
                    "counterfactual_best_idx": int(counterfactual_best_idx),
                    "counterfactual_margin": float(counterfactual_margin),
                    "counterfactual_entropy": float(counterfactual_entropy),
                    "counterfactual_selected_score": 0.0,
                    "candidate_has_attack_convert": bool(candidate_has_attack_convert),
                    "candidate_has_attack_pressure": bool(candidate_has_attack_pressure),
                    "candidate_has_good_attack": bool(candidate_has_good_attack),
                })
                break

            cand = candidates[action_idx]
            selected_oracle_score = _candidate_oracle_score(cand, len(candidates) > 1)
            action = reconstruct_action(cand, planning_game)
            move = _candidate_move(planning_game, action)
            executed_ships = int(move[0][2]) if move else 0
            planned_angle = float(move[0][1]) if move else None

            trajectory.append({
                "state": state_features,
                "candidates": cand_features,
                "action_idx": action_idx,
                "mission": cand.mission if move else "do_nothing",
                "ships": executed_ships,
                "source_id": int(action[0]) if len(action) > 0 else -1,
                "target_id": int(action[1]) if len(action) > 1 else -1,
                "planned_angle": planned_angle,
                "old_log_prob": old_log_prob,
                "entropy": entropy,
                "temperature": temperature,
                "policy_version": policy_version,
                "action_slot": action_slot,
                "noop_prob_before_cap": noop_prob_before_cap,
                "noop_prob_after_cap": noop_prob_after_cap,
                "noop_cap_value": noop_cap_value,
                "noop_has_real_candidate": noop_has_real_candidate,
                "noop_cap_applied": noop_cap_applied,
                "tactical_tag": str(getattr(cand, "tactical_tag", cand.mission)),
                "tactical_score": float(getattr(cand, "tactical_score", 0.0)),
                "tactical_oracle_idx": int(tactical_oracle_idx),
                "tactical_oracle_score": float(tactical_oracle_score),
                "tactical_oracle_tag": str(tactical_oracle_tag),
                "tactical_oracle_margin": float(tactical_oracle_score - selected_oracle_score),
                "counterfactual_scores": counterfactual_scores,
                "counterfactual_best_idx": int(counterfactual_best_idx),
                "counterfactual_margin": float(counterfactual_margin),
                "counterfactual_entropy": float(counterfactual_entropy),
                "counterfactual_selected_score": float(counterfactual_scores[action_idx]) if 0 <= action_idx < len(counterfactual_scores) else 0.0,
                "candidate_has_attack_convert": bool(candidate_has_attack_convert),
                "candidate_has_attack_pressure": bool(candidate_has_attack_pressure),
                "candidate_has_good_attack": bool(candidate_has_good_attack),
            })

            if not move:
                break

            moves.extend(move)
            _reserve_planned_ships(planning_game, action[0], executed_ships)

        return moves

    return agent


def worker_fn(
    worker_id: int,
    config: Dict[str, Any],
    pool: List[str],
    n_players: int,
    obs_queue: mp.Queue,
    action_queue: mp.Queue,
    result_queue: mp.Queue,
    control_queue: mp.Queue,
    stop_event: mp.Event,
    base_seed: int,
) -> None:
    torch.set_num_threads(1)

    episode = 0
    checkpoint_agent_cache: Dict[str, Any] = {}
    while not stop_event.is_set():
        try:
            while True:
                control_msg = control_queue.get_nowait()
                if isinstance(control_msg, dict):
                    config_patch = control_msg.get("config_patch")
                    if isinstance(config_patch, dict):
                        config.update(config_patch)
                    pool_patch = control_msg.get("pool")
                    if isinstance(pool_patch, list) and pool_patch:
                        pool = [str(item) for item in pool_patch]
        except Empty:
            pass

        seed = base_seed + worker_id * 99991 + episode * 9973
        our_index = (worker_id + episode) % n_players
        trajectory: List[Dict[str, Any]] = []

        gpu_agent = _make_gpu_agent(worker_id, obs_queue, action_queue, config, trajectory)
        opp_names = _sample_opponents(pool, seed, max(1, n_players - 1))
        opp_iter = iter(opp_names)
        agents = []
        for slot in range(n_players):
            if slot == our_index:
                agents.append(gpu_agent)
            else:
                name = next(opp_iter, None) or "random"
                agents.append(_agent_for_pool_name(name, config, checkpoint_agent_cache))

        try:
            result = run_match(
                agents,
                seed=seed,
                n_players=n_players,
                max_steps=int(config.get("max_turns", 100)),
                stop_player=our_index,
                game_engine=str(config.get("game_engine", "official_fast")),
                use_c_accel=bool(config.get("official_fast_c_accel", True)),
            )
        except Exception:
            episode += 1
            continue

        if stop_event.is_set():
            break

        _attach_fleet_outcomes(
            trajectory,
            list(result.get("fleet_events", []) or []),
            our_index,
        )

        winner = int(result.get("winner", -1))
        terminal_reward = 1.0 if winner == our_index else -1.0
        dense_reward = (
            _strategic_dense_reward(result, our_index, config)
            if config.get("dense_reward_enabled", True)
            else 0.0
        )
        action_records = [
            {
                "mission": s["mission"],
                "ships": s["ships"],
                "action_idx": int(s.get("action_idx", -1)),
                "noop_has_real_candidate": bool(s.get("noop_has_real_candidate", False)),
                "action_slot": int(s.get("action_slot", 0)),
                "fleet_launch_mapped": bool(s.get("fleet_launch_mapped", False)),
                "fleet_outcome_known": bool(s.get("fleet_outcome_known", False)),
                "fleet_hit": bool(s.get("fleet_hit", False)),
                "fleet_lost": bool(s.get("fleet_lost", False)),
                "fleet_lost_sun": bool(s.get("fleet_lost_sun", False)),
                "fleet_lost_oob": bool(s.get("fleet_lost_oob", False)),
                "fleet_pending": bool(s.get("fleet_pending", False)),
                "fleet_captured": bool(s.get("fleet_captured", False)),
                "fleet_supported": bool(s.get("fleet_supported", False)),
                "fleet_enemy_hit": bool(s.get("fleet_enemy_hit", False)),
                "fleet_neutral_hit": bool(s.get("fleet_neutral_hit", False)),
                "tactical_tag": str(s.get("tactical_tag", s.get("mission", "unknown"))),
                "tactical_score": float(s.get("tactical_score", 0.0)),
                "tactical_oracle_idx": int(s.get("tactical_oracle_idx", -1)),
                "tactical_oracle_tag": str(s.get("tactical_oracle_tag", "unknown")),
                "tactical_oracle_margin": float(s.get("tactical_oracle_margin", 0.0)),
                "counterfactual_best_idx": int(s.get("counterfactual_best_idx", -1)),
                "counterfactual_margin": float(s.get("counterfactual_margin", 0.0)),
                "counterfactual_entropy": float(s.get("counterfactual_entropy", 0.0)),
                "candidate_has_attack_convert": bool(s.get("candidate_has_attack_convert", False)),
                "candidate_has_attack_pressure": bool(s.get("candidate_has_attack_pressure", False)),
                "candidate_has_good_attack": bool(s.get("candidate_has_good_attack", False)),
            }
            for s in trajectory
        ]

        result_queue.put({
            "worker_id": worker_id,
            "trajectory": trajectory,
            "terminal_reward": terminal_reward,
            "dense_reward": dense_reward,
            "win": float(winner == our_index),
            "opponents": list(opp_names),
            "episode_length": int(result.get("steps", 0)),
            "seed": seed,
            "action_metrics": summarize_action_records(action_records),
        })

        episode += 1
