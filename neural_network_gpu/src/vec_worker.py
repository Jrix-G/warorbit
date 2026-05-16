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
                })
                break

            cand = candidates[action_idx]
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
