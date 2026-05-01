from neural_network.src.baselines import RandomAgent, GreedyNearestWeakestAgent
from neural_network.src.self_play import make_synthetic_game


def test_baselines_exist():
    game = make_synthetic_game(0)
    assert RandomAgent().act(game)
    assert GreedyNearestWeakestAgent().act(game)
