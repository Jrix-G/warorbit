#!/usr/bin/env python3
"""Smoke tests for the Orbit Wars V8 pipeline."""

from __future__ import annotations

import os

import numpy as np

import bot_v7
import bot_v8
from SimGame import SimGame, passive_policy
from v8_core import LinearV8Model, build_candidate_plans, build_state_features, select_plan


def test_candidate_generation():
    game = SimGame.random_game(seed=7, n_players=2, neutral_pairs=4)
    obs = game.observation(0)
    world = bot_v7._build_world(obs)
    state = build_state_features(world)
    plans = build_candidate_plans(world)
    assert len(plans) >= 2, "expected baseline + variants"
    assert state.shape[0] > 0
    for plan in plans:
        assert isinstance(plan.actions, list)
        assert np.isfinite(plan.features).all()
    print(f"candidate_generation: OK ({len(plans)} plans, {state.shape[0]} state features)")


def test_model_roundtrip():
    model = LinearV8Model.zero()
    path = os.path.abspath(os.path.join("evaluations", "__v8_roundtrip_test.npz"))
    if os.path.exists(path):
        os.remove(path)
    model.score_w[:3] = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    model.value_w[:3] = np.array([0.4, 0.5, -0.6], dtype=np.float32)
    model.save(path)
    restored = LinearV8Model.load(path)
    assert np.allclose(model.score_w, restored.score_w)
    assert np.allclose(model.value_w, restored.value_w)
    os.remove(path)
    print("model_roundtrip: OK")


def test_policy_runs():
    bot_v8.set_model(LinearV8Model.zero())
    game = SimGame.random_game(seed=11, n_players=2, neutral_pairs=4)
    result = game.run([bot_v8.agent, passive_policy], max_steps=40)
    assert "winner" in result
    assert len(result["scores"]) == 2
    print(f"policy_runs: OK winner={result['winner']} scores={result['scores']}")


def test_select_plan():
    game = SimGame.random_game(seed=13, n_players=2, neutral_pairs=4)
    obs = game.observation(0)
    world = bot_v7._build_world(obs)
    model = LinearV8Model.zero()
    plan, plans, state = select_plan(world, model)
    assert plan is not None
    assert len(plans) >= 2
    assert state.shape[0] > 0
    print(f"select_plan: OK chosen={plan.name} plans={len(plans)}")


if __name__ == "__main__":
    test_candidate_generation()
    test_model_roundtrip()
    test_select_plan()
    test_policy_runs()
    print("V8 smoke tests passed")
