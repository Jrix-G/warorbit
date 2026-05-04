import numpy as np

from war_orbit.agents.v9.policy import V9Weights
from war_orbit.config.v9_config import V9Config
from war_orbit.training.curriculum import build_cross_play_specs, build_role_pools
from war_orbit.training.trainer import (
    _apply_guardian_adjustments,
    _is_generalization_failure,
    _partial_reset,
    _promotion_blockers,
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


def test_promotion_requires_heldout_4p_backbone_and_front_quality():
    cfg = V9Config(
        min_improvement=0.01,
        min_benchmark_score=0.40,
        min_promotion_benchmark_games=8,
        guardian_min_benchmark_4p=0.42,
        guardian_min_benchmark_backbone=0.08,
        guardian_max_benchmark_fronts=2.70,
        guardian_max_generalization_gap=0.18,
    )
    eval_summary = {"mean": 0.55}
    weak_diag = {
        "mean": 0.50,
        "wr_4p": 0.50,
        "n_2p": 2,
        "n_4p": 8,
        "backbone_turn_frac": 0.02,
        "active_front_avg": 1.9,
    }
    strong_diag = dict(weak_diag, backbone_turn_frac=0.10, active_front_avg=2.1)
    assert not _should_promote(0.45, -1.0, eval_summary, weak_diag, cfg)
    assert "bb_low" in _promotion_blockers(0.45, -1.0, eval_summary, weak_diag, cfg)
    assert _should_promote(0.45, -1.0, eval_summary, strong_diag, cfg)


def test_guardian_raises_backbone_pressure_when_heldout_backbone_collapses():
    cfg = V9Config(backbone_penalty_weight=0.10, backbone_bonus_weight=0.08, front_pressure_plan_bias=0.14)
    train = {"mean": 0.55, "backbone_turn_frac": 0.15}
    eval_summary = {"mean": 0.52}
    benchmark = {
        "mean": 0.32,
        "wr_4p": 0.27,
        "n_2p": 4,
        "n_4p": 12,
        "backbone_turn_frac": 0.03,
        "active_front_avg": 1.9,
    }
    event = _apply_guardian_adjustments(cfg, train, eval_summary, benchmark)
    assert event["changed"] == 1.0
    assert cfg.backbone_penalty_weight > 0.10
    assert cfg.backbone_bonus_weight > 0.08
    assert cfg.front_pressure_plan_bias > 0.14


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
    assert {s.n_players for s in specs} == {4}


def test_guardian_enables_strict_focus_after_repeated_low_4p():
    cfg = V9Config()
    train = {"mean": 0.55, "backbone_turn_frac": 0.15}
    eval_summary = {"mean": 0.52}
    benchmark = {
        "mean": 0.32,
        "wr_4p": 0.27,
        "n_2p": 0,
        "n_4p": 12,
        "backbone_turn_frac": 0.10,
        "active_front_avg": 2.0,
    }
    _apply_guardian_adjustments(cfg, train, eval_summary, benchmark)
    event = _apply_guardian_adjustments(cfg, train, eval_summary, benchmark)
    assert event["strict_focus_fix"] == 1.0
    assert cfg.strict_single_target_4p
    assert cfg.disable_snipe_4p
    assert cfg.max_focus_targets_4p == 1
