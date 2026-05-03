"""Lightweight checks for replay harvesting helpers."""

from pathlib import Path

from harvest_replays import default_state_path, detect_anomalies


def test_default_state_path():
    p = default_state_path(Path("replay_dataset/compact/episodes.jsonl.gz"))
    assert str(p).endswith("episodes.jsonl.gz.state.json")


def test_detect_anomalies_flags_large_fleet():
    compact = {
        "steps": 500,
        "summary": {
            "max_fleet_ships": 1800,
            "max_total_ships": 9100.0,
            "avg_actions_per_turn": 5.0,
        },
    }
    reasons = detect_anomalies(compact)
    assert "max_fleet_ships=1800" in reasons
    assert "late_game_extreme_total_ships" in reasons
