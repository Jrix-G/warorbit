import numpy as np

from war_orbit.agents.v9.policy import V9Weights
from war_orbit.config.v9_config import V9Config
from war_orbit.training.curriculum import build_cross_play_specs, build_role_pools
from war_orbit.training.trainer import (
    _is_generalization_failure,
    _partial_reset,
    _selection_score,
    _should_promote,
)


def test_role_pools_are_separated_and_eval_has_required_baselines():
    cfg = V9Config(
        best_checkpoint="evaluations/nonexistent_v9_test_best.npz",
        training_opponents=["random", "noisy_greedy", "starter"],
        eval_opponents=["heldout_random", "heldout_greedy", "bot_v7", "sun_dodge"],
        benchmark_opponents=["structured", "notebook_tactical_heuristic"],
    )
    pools = build_role_pools(cfg)
    assert not (set(pools["train"]) & set(pools["eval"]))
    assert not (set(pools["train"]) & set(pools["benchmark"]))
    assert {"heldout_random", "heldout_greedy", "bot_v7"}.issubset(set(pools["eval"]))


def test_default_v9_protocol_uses_large_benchmark_and_hard_timeout():
    cfg = V9Config()
    pools = build_role_pools(cfg)
    assert cfg.benchmark_games == 128
    assert cfg.min_promotion_benchmark_games == 128
    assert cfg.hard_timeout_minutes == 60.0
    assert "robust" in cfg.checkpoint
    assert len(pools["train"]) >= 12
    assert len(pools["benchmark"]) >= 5


def test_promotion_uses_benchmark_guard_not_internal_eval_only():
    cfg = V9Config(min_improvement=0.01, min_benchmark_score=0.15, max_generalization_gap=0.25)
    eval_summary = {"mean": 1.0}
    collapsed_benchmark = {"mean": 0.0}
    assert _selection_score(eval_summary, collapsed_benchmark) == 0.0
    assert not _should_promote(0.0, -1.0, eval_summary, collapsed_benchmark, cfg)


def test_generalization_failure_detector_trips_on_train_eval_benchmark_split():
    cfg = V9Config(
        overfit_train_threshold=0.80,
        overfit_eval_threshold=0.70,
        benchmark_collapse_threshold=0.12,
        max_generalization_gap=0.30,
    )
    train = {"mean": 0.95}
    eval_summary = {"mean": 0.90}
    benchmark = {"mean": 0.05}
    assert _is_generalization_failure(train, eval_summary, benchmark, cfg)


def test_partial_reset_moves_weights_toward_defaults():
    defaults = V9Weights.defaults().flatten()
    flat = defaults + 5.0
    rng = np.random.default_rng(123)
    reset = _partial_reset(flat, defaults, rng, fraction=1.0)
    assert np.linalg.norm(reset - defaults) < np.linalg.norm(flat - defaults)
    assert not np.allclose(reset, flat)


def test_cross_play_specs_rotate_slots_and_preserve_phase():
    specs = build_cross_play_specs(
        ["heldout_random", "heldout_greedy", "bot_v7"],
        games=6,
        seed=9,
        seed_offset=10,
        max_steps=50,
        four_player_ratio=1.0,
        phase="eval",
    )
    assert {s.phase for s in specs} == {"eval"}
    assert {s.our_index for s in specs} >= {0, 1, 2, 3}
    assert {s.n_players for s in specs} == {2, 4}
