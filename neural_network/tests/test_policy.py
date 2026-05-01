from neural_network.src.policy import build_action_candidates, choose_action, is_valid_action
import numpy as np


def sample_game():
    return {
        "my_id": 0,
        "planets": [
            {"id": 0, "owner": 0, "ships": 50},
            {"id": 1, "owner": -1, "ships": 20},
        ],
    }


def test_policy_selects_valid_action():
    game = sample_game()
    outputs = {"policy_logits": np.array([0.1, 2.0, -1.0], dtype=np.float32)}
    cand = choose_action(outputs, game)
    assert is_valid_action(cand, game)

