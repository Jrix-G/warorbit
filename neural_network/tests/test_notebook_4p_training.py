from neural_network.scripts.run_notebook_4p_training import _prepare_config


def test_prepare_config_forces_four_player():
    cfg = {"checkpoint_dir": "x", "log_dir": "y"}
    out = _prepare_config(cfg, 5.0, 12)
    assert out["curriculum_enabled"] is True
    assert out["four_player_ratio"] == 1.0
    assert out["eval_four_player_ratio"] == 1.0
    assert out["curriculum_early_4p_ratio"] == 1.0
    assert out["curriculum_late_4p_ratio"] == 1.0
