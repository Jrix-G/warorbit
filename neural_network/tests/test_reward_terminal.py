from neural_network.src.reward import compute_reward


def test_terminal_reward_for_win():
    prev_state = {"my_id": 0, "planets": [{"owner": 0, "production": 2}], "is_four_player": False}
    next_state = {"my_id": 0, "planets": [{"owner": 0, "production": 2}], "winner": 0, "is_four_player": False}
    reward = compute_reward(prev_state, {"ships": 5}, next_state, terminal=True)
    assert reward > 10
