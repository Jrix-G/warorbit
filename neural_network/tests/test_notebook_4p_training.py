from pathlib import Path

from neural_network.scripts.run_notebook_4p_training import _prepare_config
from neural_network.scripts.run_90min_6agent_training import _prepare_config as _prepare_population_config
from neural_network.src.model import ModelConfig, NeuralNetworkModel, count_parameters
from neural_network.src.population_4p_training import (
    _composite_score,
    _load_curriculum_state,
    _maybe_advance_curriculum,
    _next_base_checkpoint,
    _should_try_promotion,
    _tier_best_checkpoint_path,
    _tier_pool,
    _training_base_checkpoint,
    DEFAULT_CURRICULUM_TIERS,
)
from neural_network.src.trajectory import safe_plan_shot


def test_prepare_config_forces_four_player():
    cfg = {"checkpoint_dir": "x", "log_dir": "y"}
    out = _prepare_config(cfg, 5.0, 12)
    assert out["curriculum_enabled"] is True
    assert out["four_player_ratio"] == 1.0
    assert out["eval_four_player_ratio"] == 1.0
    assert out["curriculum_early_4p_ratio"] == 1.0
    assert out["curriculum_late_4p_ratio"] == 1.0
    assert out["train_notebook_opponents"] == 3


def test_population_config_uses_six_workers_and_full_notebook_pool():
    cfg = {"checkpoint_dir": "x", "log_dir": "y", "seed": 42}
    out = _prepare_population_config(cfg, 90.0, 6, 16)
    assert out["duration_minutes"] == 90.0
    assert out["workers"] == 6
    assert out["hidden_dim"] == 320
    assert out["notebook_pool_limit"] == 15
    assert out["train_notebook_opponents"] == 3
    assert out["worker_train_steps"] >= 24
    assert out["max_actions_per_turn"] == 4
    assert out["dense_reward_enabled"] is True
    assert out["candidate_eval_episodes"] < out["eval_episodes"]
    assert out["opponent_curriculum_enabled"] is True
    assert out["resume_from_tier_best"] is True
    assert out["tier_checkpoint_dir"].replace("\\", "/").endswith("neural_network/checkpoints/tiers")


def test_population_config_resumes_from_best_and_confirms_promotions():
    cfg = {
        "checkpoint_dir": "x",
        "log_dir": "y",
        "best_checkpoint": "neural_network/checkpoints/best.npz",
        "seed": 42,
    }
    out = _prepare_population_config(cfg, 200.0, 6, 8)
    assert out["resume_checkpoint"] == out["best_checkpoint"]
    assert out["eval_episodes"] == 8
    assert out["promotion_eval_episodes"] == 16
    assert out["candidate_eval_episodes"] == 8
    assert out["promotion_margin"] == 0.02
    assert out["promotion_min_remaining_minutes"] == 12.0


def test_population_promotion_requires_real_improvement():
    assert not _should_try_promotion({"score": 0.375}, best_score=0.375, margin=0.02)
    assert _should_try_promotion({"score": 0.5}, best_score=0.375, margin=0.02)


def test_population_uses_tier_checkpoint_when_available(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    tier_dir = tmp_path / "tiers"
    fallback = checkpoint_dir / "best.npz"
    tier_checkpoint = tier_dir / "heuristic_500.npz"
    tier_checkpoint.parent.mkdir(parents=True)
    tier_checkpoint.write_bytes(b"exists")

    cfg = {"tier_checkpoint_dir": str(tier_dir)}
    assert _tier_best_checkpoint_path(cfg, checkpoint_dir, "heuristic_500") == tier_checkpoint
    assert _training_base_checkpoint(cfg, True, checkpoint_dir, "heuristic_500", fallback) == tier_checkpoint
    assert _training_base_checkpoint(cfg, False, checkpoint_dir, "heuristic_500", fallback) == fallback
    assert _training_base_checkpoint({**cfg, "resume_from_tier_best": False}, True, checkpoint_dir, "heuristic_500", fallback) == fallback
    assert _next_base_checkpoint(cfg, True, tier_checkpoint, fallback, fallback) == tier_checkpoint
    assert _next_base_checkpoint(cfg, False, tier_checkpoint, fallback, fallback) == fallback
    assert _next_base_checkpoint(cfg, False, tier_checkpoint, tier_checkpoint, fallback) == tier_checkpoint


def test_population_tier_checkpoint_path_sanitizes_custom_tier(tmp_path):
    path = _tier_best_checkpoint_path({"tier_checkpoint_dir": str(tmp_path)}, Path("unused"), "mixed 700/notebook")
    assert path == tmp_path / "mixed_700_notebook.npz"


def test_population_composite_score_penalizes_rank_and_noop():
    good = {"winrate": 0.25, "rank_mean": 2.0, "eval_mean": 0.2, "avg_score": 300.0, "eval_do_nothing_rate": 0.2, "eval_avg_ships_sent": 20.0}
    bad = {"winrate": 0.25, "rank_mean": 3.8, "eval_mean": -0.8, "avg_score": 10.0, "eval_do_nothing_rate": 0.9, "eval_avg_ships_sent": 0.0}
    assert _composite_score(good) > _composite_score(bad)


def test_population_composite_score_ignores_stale_cached_score():
    stale = {
        "composite_score": 0.95,
        "score": 0.95,
        "winrate": 0.0,
        "rank_mean": 4.0,
        "eval_mean": -1.0,
        "avg_score": 0.0,
        "eval_do_nothing_rate": 1.0,
        "eval_avg_ships_sent": 0.0,
    }
    assert _composite_score(stale) < 0.0


def test_safe_planner_rejects_sun_crossing_shot():
    game = {
        "turn": 0,
        "angular_velocity": 0.0,
        "initial_planets": [],
        "planets": [
            {"id": 0, "owner": 0, "ships": 50, "x": 18.0, "y": 50.0, "radius": 3.0, "production": 3.0},
            {"id": 1, "owner": 1, "ships": 50, "x": 82.0, "y": 50.0, "radius": 3.0, "production": 3.0},
        ],
    }
    assert safe_plan_shot(game["planets"][0], game["planets"][1], game) is None


def test_population_curriculum_starts_weak_and_advances():
    tiers = [dict(item) for item in DEFAULT_CURRICULUM_TIERS]
    state = _load_curriculum_state("missing_curriculum_state_for_test.json", tiers, {}, resume=True)
    assert tiers[state["tier_index"]]["name"] == "basic_300"
    assert _tier_pool({}, tiers[0]) == ["random", "greedy", "starter"]
    assert any(tier["name"] == "notebook_core4" for tier in tiers)
    assert any(tier["name"] == "notebook_mid8" for tier in tiers)
    state["tier_generation"] = int(tiers[0]["min_generations"])
    state["total_generations"] = 3
    advanced, _reason = _maybe_advance_curriculum(
        state,
        tiers,
        {"winrate": 0.75, "rank_mean": 2.0, "eval_mean": 0.4, "eval_do_nothing_rate": 0.2, "eval_avg_ships_sent": 20.0},
    )
    assert advanced
    assert tiers[state["tier_index"]]["name"] == "heuristic_500"


def test_population_config_caps_duration_at_eight_hours():
    cfg = {"checkpoint_dir": "x", "log_dir": "y", "seed": 42}
    out = _prepare_population_config(cfg, 600.0, 6, 16)
    assert out["duration_minutes"] == 480.0


def test_population_model_parameter_budget_is_near_target():
    input_dim = 11 + 64 * 19 + 128 * 10 + 4 * 8
    model = NeuralNetworkModel(ModelConfig(input_dim=input_dim, hidden_dim=320))
    params = count_parameters(model)
    assert 1_700_000 <= params <= 2_100_000
