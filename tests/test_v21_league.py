from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import v21_league
import v21_pool


def _player(kind: str, mu: float, games: int = 8):
    return {
        "id": "p",
        "kind": kind,
        "rating": {"mu": mu, "sigma": 50.0},
        "games": games,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "score": 0.0,
        "retired": False,
        "retire_reason": "",
        "metadata": {},
    }


def test_classify_player_promotion_and_relegation_are_pure():
    rules = v21_league.PromotionRules(min_games=4, promote_mu=1550.0, relegate_mu=1450.0)
    assert v21_league.classify_player(_player("historical", 1560.0, 4), rules) == "active"
    assert v21_league.classify_player(_player("active", 1440.0, 4), rules) == "historical"
    assert v21_league.classify_player(_player("anchor", 1000.0, 99), rules) == "anchor"


def test_exploiter_promotes_only_after_threshold():
    rules = v21_league.PromotionRules(min_games=4, exploiter_promote_mu=1575.0)
    assert v21_league.classify_player(_player("exploiter", 1574.0, 4), rules) == "exploiter"
    assert v21_league.classify_player(_player("exploiter", 1575.0, 4), rules) == "active"
    assert v21_league.classify_player(_player("exploiter", 1700.0, 3), rules) == "exploiter"


def test_plan_and_apply_transitions_update_pools():
    state = v21_pool.empty_state()
    state = v21_pool.add_player(state, "old", "historical", rating={"mu": 1600.0, "sigma": 50.0})
    state = v21_pool.add_player(state, "bad", "active", rating={"mu": 1200.0, "sigma": 50.0})
    state["players"]["old"]["games"] = 8
    state["players"]["bad"]["games"] = 8

    transitions = v21_league.plan_transitions(state)
    assert {item["player"]: item["to"] for item in transitions} == {"bad": "retired", "old": "active"}

    updated = v21_league.apply_transitions(state, transitions, now="2026-05-21T00:00:00+00:00")
    assert updated["players"]["old"]["kind"] == "active"
    assert updated["players"]["bad"]["retired"] is True
    assert updated["pools"]["active"] == ["old"]
    assert updated["pools"]["retired"] == ["bad"]


def test_select_pairings_prefers_anchor_matches():
    state = v21_pool.empty_state()
    state = v21_pool.add_player(state, "anchor", "anchor")
    state = v21_pool.add_player(state, "candidate", "active", rating={"mu": 1600.0, "sigma": 50.0})
    state = v21_pool.add_player(state, "exploit", "exploiter", rating={"mu": 1700.0, "sigma": 50.0})

    assert v21_league.select_pairings(state, limit=2) == [("exploit", "anchor"), ("candidate", "anchor")]


def test_league_cli_smoke_is_light(tmp_path, capsys):
    path = tmp_path / "league_state.json"
    rc = v21_league.main(["--state", str(path), "smoke"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["summary"]["players"] == 3
    assert out["pairings"]
    assert path.exists() is False
