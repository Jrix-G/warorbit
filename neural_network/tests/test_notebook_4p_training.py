from neural_network.scripts.run_notebook_4p_training import _prepare_config
from neural_network.scripts.run_90min_6agent_training import _prepare_config as _prepare_population_config
from neural_network.src.model import ModelConfig, NeuralNetworkModel, count_parameters


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
    assert out["hidden_dim"] == 256
    assert out["notebook_pool_limit"] == 15
    assert out["train_notebook_opponents"] == 3


def test_population_config_caps_duration_at_eight_hours():
    cfg = {"checkpoint_dir": "x", "log_dir": "y", "seed": 42}
    out = _prepare_population_config(cfg, 600.0, 6, 16)
    assert out["duration_minutes"] == 480.0


def test_population_model_parameter_budget_is_near_target():
    input_dim = 11 + 64 * 19 + 128 * 10 + 4 * 8
    model = NeuralNetworkModel(ModelConfig(input_dim=input_dim, hidden_dim=256))
    params = count_parameters(model)
    assert 1_200_000 <= params <= 1_500_000
