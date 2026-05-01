import numpy as np
import torch
from neural_network.src.model import NeuralNetworkModel, ModelConfig
from neural_network.src.policy import build_action_candidates, choose_action


def test_policy_does_not_collapse_to_nothing():
    game = {
        "my_id": 0,
        "planets": [
            {"id": 0, "owner": 0, "ships": 50, "x": 0.0, "y": 0.0, "production": 3.0},
            {"id": 1, "owner": -1, "ships": 20, "x": 10.0, "y": 0.0, "production": 2.0},
        ],
    }
    model = NeuralNetworkModel(ModelConfig(input_dim=10))
    state = torch.zeros(1, 10)
    cand = build_action_candidates(game)
    feats = torch.tensor(np.stack([c.score_features for c in cand]), dtype=torch.float32).unsqueeze(0)
    out = model(state, feats)
    real = 0
    for _ in range(100):
        chosen, _ = choose_action(out, game, explore=True)
        if chosen.mission != "do_nothing":
            real += 1
    assert real > 0
