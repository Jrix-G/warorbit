from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import v21_pool


def test_empty_state_has_all_pools():
    state = v21_pool.empty_state(now="2026-05-21T00:00:00+00:00")
    assert state["version"] == 1
    assert set(state["pools"]) == {"anchors", "historical", "exploiters", "active", "retired"}


def test_add_and_retire_player_rebuilds_pools():
    state = v21_pool.empty_state()
    state = v21_pool.add_player(state, "a0", "anchor")
    state = v21_pool.add_player(state, "x0", "exploiter")
    assert state["pools"]["anchors"] == ["a0"]
    assert state["pools"]["exploiters"] == ["x0"]

    state = v21_pool.retire_player(state, "x0", "stale")
    assert state["players"]["x0"]["retired"] is True
    assert state["players"]["x0"]["retire_reason"] == "stale"
    assert state["pools"]["exploiters"] == []
    assert state["pools"]["retired"] == ["x0"]


def test_record_match_result_updates_counts():
    state = v21_pool.empty_state()
    state = v21_pool.add_player(state, "p0")
    state = v21_pool.add_player(state, "p1")
    state = v21_pool.record_match_result(state, {"p0": 3.0, "p1": 1.0})

    assert state["players"]["p0"]["games"] == 1
    assert state["players"]["p0"]["wins"] == 1
    assert state["players"]["p1"]["losses"] == 1
    assert state["players"]["p0"]["score"] == 3.0


def test_manager_round_trip_json(tmp_path):
    path = tmp_path / "league_state.json"
    manager = v21_pool.LeagueStateManager(path)
    state = v21_pool.add_player(v21_pool.empty_state(), "p0", "historical")
    manager.save(state)

    loaded = manager.load()
    assert loaded["players"]["p0"]["kind"] == "historical"
    assert json.loads(path.read_text(encoding="utf-8"))["pools"]["historical"] == ["p0"]


def test_pool_cli_smoke_does_not_require_engine(tmp_path, capsys):
    path = tmp_path / "league_state.json"
    rc = v21_pool.main(["--state", str(path), "smoke"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["players"] == 2
    assert path.exists() is False
