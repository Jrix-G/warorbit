from neural_network.src.reward import compute_reward


def test_reward_is_numeric():
    prev_state = {"my_id": 0, "planets": [{"owner": 0, "production": 2}], "is_four_player": False}
    next_state = {"my_id": 0, "planets": [{"owner": 0, "production": 2}, {"owner": 0, "production": 3}], "winner": 0, "is_four_player": False}
    reward = compute_reward(prev_state, {"ships": 5}, next_state, terminal=True)
    assert isinstance(reward, float)
    assert reward > 0

