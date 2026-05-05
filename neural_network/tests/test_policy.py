import numpy as np
import torch
from neural_network.src.policy import build_action_candidates, choose_action, is_valid_action, _candidate_prior
from neural_network.src.model import NeuralNetworkModel, ModelConfig


def sample_game():
    return {
        "my_id": 0,
        "planets": [
            {"id": 0, "owner": 0, "ships": 50, "x": 0.0, "y": 0.0, "production": 3.0},
            {"id": 1, "owner": -1, "ships": 20, "x": 10.0, "y": 0.0, "production": 2.0},
        ],
    }


def test_policy_selects_valid_action():
    game = sample_game()
    model = NeuralNetworkModel(ModelConfig(input_dim=10))
    state = torch.zeros(1, 10)
    candidates = build_action_candidates(game)
    cand_features = torch.tensor(np.stack([c.score_features for c in candidates]), dtype=torch.float32).unsqueeze(0)
    out = model(state, cand_features)
    cand, _ = choose_action(out, game, explore=False)
    assert is_valid_action(cand, game)


def test_policy_filters_sun_blocked_targets():
    game = {
        "my_id": 0,
        "turn": 0,
        "angular_velocity": 0.0,
        "initial_planets": [],
        "planets": [
            {"id": 0, "owner": 0, "ships": 50, "x": 18.0, "y": 50.0, "radius": 3.0, "production": 3.0},
            {"id": 1, "owner": 1, "ships": 20, "x": 82.0, "y": 50.0, "radius": 3.0, "production": 2.0},
        ],
    }
    candidates = build_action_candidates(game)
    assert [c.mission for c in candidates] == ["do_nothing"]


def test_default_policy_ratios_prefer_useful_ship_amounts():
    game = sample_game()
    candidates = build_action_candidates(game)
    amounts = sorted(c.amount for c in candidates if c.mission == "expand")
    assert amounts == [25, 35, 45]


def test_prior_penalizes_tiny_failed_expansions():
    game = sample_game()
    candidates = build_action_candidates(game, send_ratios=[0.02, 0.5], min_expand_attack_ships=1)
    tiny = next(c for c in candidates if c.mission == "expand" and c.amount == 1)
    useful = next(c for c in candidates if c.mission == "expand" and c.amount == 25)
    assert _candidate_prior(useful, game) > _candidate_prior(tiny, game)


def test_min_expand_attack_ships_filters_micro_expansions():
    game = sample_game()
    candidates = build_action_candidates(game, send_ratios=[0.02, 0.5], min_expand_attack_ships=6)
    assert all(c.amount >= 6 for c in candidates if c.mission in {"expand", "attack"})
    assert not any(c.mission == "expand" and c.amount == 1 for c in candidates)
