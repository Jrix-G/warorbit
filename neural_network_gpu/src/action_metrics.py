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
        }

    action_count = len(records)
    real_actions: List[Dict[str, Any]] = []
    real_ships: List[float] = []
    all_ships: List[float] = []
    legal_noop_count = 0
    forced_noop_count = 0
    mission_counts: Dict[str, int] = {}
    mission_ship_totals: Dict[str, List[float]] = {}
    slot_ship_totals: Dict[int, List[float]] = {}
    slot_counts: Dict[int, int] = {}
    slot_real_counts: Dict[int, int] = {}
    slot_legal_noop_counts: Dict[int, int] = {}
    slot_forced_noop_counts: Dict[int, int] = {}

    for rec in records:
        mission = str(rec.get("mission") or "do_nothing")
        ships = float(rec.get("ships") or 0.0)
        action_slot = int(rec.get("action_slot") or 0)
        has_real_candidate = bool(rec.get("noop_has_real_candidate", mission != "do_nothing" and ships > 0.0))
        is_noop = mission == "do_nothing" or ships <= 0.0

        mission_counts[mission] = mission_counts.get(mission, 0) + 1
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
    }

    metrics["mission_counts"] = dict(mission_counts)

    for mission in ("expand", "attack", "support", "do_nothing"):
        ships_values = mission_ship_totals.get(mission, [])
        metrics[f"mission_{mission}_count"] = float(mission_counts.get(mission, 0))
        metrics[f"mission_{mission}_ships_mean"] = float(np.mean(ships_values)) if ships_values else 0.0

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
