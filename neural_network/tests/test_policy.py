import numpy as np
import torch
from neural_network.src.policy import build_action_candidates, choose_action, is_valid_action
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
