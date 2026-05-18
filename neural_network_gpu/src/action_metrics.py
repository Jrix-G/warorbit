from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np


def _quantile(values: List[float], q: float, default: float = 0.0) -> float:
    if not values:
        return default
    return float(np.quantile(np.asarray(values, dtype=np.float32), q))


def summarize_action_records(action_records: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    records = [dict(rec) for rec in action_records]
    if not records:
        return {
            "action_count": 0.0,
            "real_action_count": 0.0,
            "do_nothing_rate": 1.0,
            "legal_noop_rate": 0.0,
            "forced_noop_rate": 0.0,
            "real_action_rate": 0.0,
            "avg_ships_sent": 0.0,
            "avg_ships_per_action": 0.0,
            "median_ships_sent": 0.0,
            "ships_sent_p25": 0.0,
            "ships_sent_p75": 0.0,
            "ships_sent_p90": 0.0,
            "ships_sent_max": 0.0,
            "fleet_hit_rate": 0.0,
            "fleet_lost_rate": 0.0,
            "fleet_capture_rate": 0.0,
        }

    action_count = len(records)
    real_actions: List[Dict[str, Any]] = []
    real_ships: List[float] = []
    all_ships: List[float] = []
    legal_noop_count = 0
    forced_noop_count = 0
    mission_counts: Dict[str, int] = {}
    tactical_counts: Dict[str, int] = {}
    mission_ship_totals: Dict[str, List[float]] = {}
    slot_ship_totals: Dict[int, List[float]] = {}
    slot_counts: Dict[int, int] = {}
    slot_real_counts: Dict[int, int] = {}
    slot_legal_noop_counts: Dict[int, int] = {}
    slot_forced_noop_counts: Dict[int, int] = {}
    fleet_launch_mapped = 0
    fleet_outcome_known = 0
    fleet_hits = 0
    fleet_enemy_hits = 0
    fleet_neutral_hits = 0
    fleet_supports = 0
    fleet_captures = 0
    fleet_losses = 0
    fleet_lost_sun = 0
    fleet_lost_oob = 0
    fleet_pending = 0

    for rec in records:
        mission = str(rec.get("mission") or "do_nothing")
        tactical_tag = str(rec.get("tactical_tag") or mission)
        ships = float(rec.get("ships") or 0.0)
        action_slot = int(rec.get("action_slot") or 0)
        has_real_candidate = bool(rec.get("noop_has_real_candidate", mission != "do_nothing" and ships > 0.0))
        is_noop = mission == "do_nothing" or ships <= 0.0

        mission_counts[mission] = mission_counts.get(mission, 0) + 1
        tactical_counts[tactical_tag] = tactical_counts.get(tactical_tag, 0) + 1
        mission_ship_totals.setdefault(mission, []).append(ships)
        slot_ship_totals.setdefault(action_slot, []).append(ships)
        slot_counts[action_slot] = slot_counts.get(action_slot, 0) + 1
        if has_real_candidate:
            slot_legal_noop_counts[action_slot] = slot_legal_noop_counts.get(action_slot, 0) + int(is_noop)
        else:
            slot_forced_noop_counts[action_slot] = slot_forced_noop_counts.get(action_slot, 0) + int(is_noop)
        slot_real_counts[action_slot] = slot_real_counts.get(action_slot, 0) + int(not is_noop)

        all_ships.append(ships)
        if not is_noop:
            real_actions.append(rec)
            real_ships.append(ships)
            fleet_launch_mapped += int(bool(rec.get("fleet_launch_mapped", False)))
            fleet_outcome_known += int(bool(rec.get("fleet_outcome_known", False)))
            fleet_hits += int(bool(rec.get("fleet_hit", False)))
            fleet_enemy_hits += int(bool(rec.get("fleet_enemy_hit", False)))
            fleet_neutral_hits += int(bool(rec.get("fleet_neutral_hit", False)))
            fleet_supports += int(bool(rec.get("fleet_supported", False)))
            fleet_captures += int(bool(rec.get("fleet_captured", False)))
            fleet_losses += int(bool(rec.get("fleet_lost", False)))
            fleet_lost_sun += int(bool(rec.get("fleet_lost_sun", False)))
            fleet_lost_oob += int(bool(rec.get("fleet_lost_oob", False)))
            fleet_pending += int(bool(rec.get("fleet_pending", False)))
        elif has_real_candidate:
            legal_noop_count += 1
        else:
            forced_noop_count += 1

    real_action_count = len(real_actions)
    legal_decision_count = max(1, real_action_count + legal_noop_count)
    noop_count = action_count - real_action_count

    avg_ships_sent = float(np.mean(real_ships)) if real_ships else 0.0
    avg_ships_per_action = float(np.mean(all_ships)) if all_ships else 0.0

    metrics: Dict[str, float] = {
        "action_count": float(action_count),
        "real_action_count": float(real_action_count),
        "do_nothing_rate": float(noop_count) / float(action_count),
        "legal_noop_rate": float(legal_noop_count) / float(legal_decision_count),
        "forced_noop_rate": float(forced_noop_count) / float(action_count),
        "real_action_rate": float(real_action_count) / float(action_count),
        "avg_ships_sent": avg_ships_sent,
        "avg_ships_per_action": avg_ships_per_action,
        "median_ships_sent": _quantile(real_ships, 0.50),
        "ships_sent_p25": _quantile(real_ships, 0.25),
        "ships_sent_p75": _quantile(real_ships, 0.75),
        "ships_sent_p90": _quantile(real_ships, 0.90),
        "ships_sent_max": float(np.max(real_ships)) if real_ships else 0.0,
        "fleet_launch_mapped_count": float(fleet_launch_mapped),
        "fleet_outcome_known_count": float(fleet_outcome_known),
        "fleet_hit_count": float(fleet_hits),
        "fleet_enemy_hit_count": float(fleet_enemy_hits),
        "fleet_neutral_hit_count": float(fleet_neutral_hits),
        "fleet_support_count": float(fleet_supports),
        "fleet_capture_count": float(fleet_captures),
        "fleet_lost_count": float(fleet_losses),
        "fleet_lost_sun_count": float(fleet_lost_sun),
        "fleet_lost_oob_count": float(fleet_lost_oob),
        "fleet_pending_count": float(fleet_pending),
        "fleet_launch_mapped_rate": float(fleet_launch_mapped) / float(max(1, real_action_count)),
        "fleet_outcome_known_rate": float(fleet_outcome_known) / float(max(1, real_action_count)),
        "fleet_hit_rate": float(fleet_hits) / float(max(1, real_action_count)),
        "fleet_enemy_hit_rate": float(fleet_enemy_hits) / float(max(1, real_action_count)),
        "fleet_neutral_hit_rate": float(fleet_neutral_hits) / float(max(1, real_action_count)),
        "fleet_support_rate": float(fleet_supports) / float(max(1, real_action_count)),
        "fleet_capture_rate": float(fleet_captures) / float(max(1, real_action_count)),
        "fleet_lost_rate": float(fleet_losses) / float(max(1, real_action_count)),
        "fleet_lost_sun_rate": float(fleet_lost_sun) / float(max(1, real_action_count)),
        "fleet_lost_oob_rate": float(fleet_lost_oob) / float(max(1, real_action_count)),
        "fleet_pending_rate": float(fleet_pending) / float(max(1, real_action_count)),
    }

    metrics["mission_counts"] = dict(mission_counts)
    metrics["tactical_counts"] = dict(tactical_counts)

    for mission in ("expand", "attack", "support", "do_nothing"):
        ships_values = mission_ship_totals.get(mission, [])
        metrics[f"mission_{mission}_count"] = float(mission_counts.get(mission, 0))
        metrics[f"mission_{mission}_ships_mean"] = float(np.mean(ships_values)) if ships_values else 0.0

    for tag in (
        "support_defense",
        "support_front",
        "support_redistribute",
        "support_passive",
        "support_backward",
        "attack_opportunity",
        "attack_pressure",
        "expand_front",
        "expand_safe",
    ):
        count = float(tactical_counts.get(tag, 0))
        metrics[f"tactical_{tag}_count"] = count
        metrics[f"tactical_{tag}_rate"] = count / float(max(1, real_action_count))

    for slot in sorted(set(slot_counts.keys()) | set(slot_ship_totals.keys())):
        ships_values = slot_ship_totals.get(slot, [])
        total = max(1, slot_counts.get(slot, 0))
        metrics[f"slot{slot}_action_count"] = float(slot_counts.get(slot, 0))
        metrics[f"slot{slot}_real_action_rate"] = float(slot_real_counts.get(slot, 0)) / float(total)
        metrics[f"slot{slot}_legal_noop_rate"] = float(slot_legal_noop_counts.get(slot, 0)) / float(total)
        metrics[f"slot{slot}_forced_noop_rate"] = float(slot_forced_noop_counts.get(slot, 0)) / float(total)
        metrics[f"slot{slot}_ships_mean"] = float(np.mean(ships_values)) if ships_values else 0.0
        metrics[f"slot{slot}_ships_p90"] = _quantile(ships_values, 0.90)

    return metrics


def summarize_fleet_events(fleet_events: Iterable[Dict[str, Any]], player: int) -> Dict[str, float]:
    events = [dict(event) for event in fleet_events]
    launches = [event for event in events if str(event.get("type")) == "launch" and int(event.get("player", -1)) == int(player)]
    launch_ids = {int(event.get("fleet_id", -1)) for event in launches}
    hits = [
        event for event in events
        if str(event.get("type")) == "hit" and int(event.get("fleet_id", -1)) in launch_ids
    ]
    losses = [
        event for event in events
        if str(event.get("type")) in {"lost_oob", "lost_sun"} and int(event.get("fleet_id", -1)) in launch_ids
    ]
    lost_sun = [event for event in losses if str(event.get("type")) == "lost_sun"]
    lost_oob = [event for event in losses if str(event.get("type")) == "lost_oob"]
    captures = [
        event for event in events
        if (
            str(event.get("type")) == "combat"
            and int(event.get("owner_before", -2)) != int(player)
            and int(event.get("owner_after", -2)) == int(player)
        )
    ]
    launch_count = max(1, len(launches))
    pending = max(0, len(launches) - len(hits) - len(losses))
    return {
        "fleet_launch_count": float(len(launches)),
        "fleet_hit_count": float(len(hits)),
        "fleet_lost_count": float(len(losses)),
        "fleet_lost_sun_count": float(len(lost_sun)),
        "fleet_lost_oob_count": float(len(lost_oob)),
        "fleet_capture_count": float(len(captures)),
        "fleet_pending_count": float(pending),
        "fleet_hit_rate": float(len(hits)) / float(launch_count),
        "fleet_lost_rate": float(len(losses)) / float(launch_count),
        "fleet_lost_sun_rate": float(len(lost_sun)) / float(launch_count),
        "fleet_lost_oob_rate": float(len(lost_oob)) / float(launch_count),
        "fleet_capture_rate": float(len(captures)) / float(launch_count),
        "fleet_pending_rate": float(pending) / float(launch_count),
    }
