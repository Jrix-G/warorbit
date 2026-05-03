from SimGame import SimGame

from war_orbit.agents.v9.planner import V9Planner
from war_orbit.agents.v9.policy import V9Agent
from war_orbit.core.game import build_world


def test_v9_planner_generates_strategic_candidates_on_4p_opening():
    game = SimGame.random_game(seed=91, n_players=4, max_steps=40)
    world = build_world(game.observation(0))
    candidates = V9Planner().generate(world)
    names = {c.plan_type for c in candidates}
    assert "aggressive_expansion" in names
    assert "balanced" in names
    assert candidates


def test_v9_agent_returns_valid_action_list():
    game = SimGame.random_game(seed=92, n_players=4, max_steps=40)
    agent = V9Agent()
    moves = agent(game.observation(0), None)
    assert isinstance(moves, list)
    assert all(len(move) == 3 for move in moves)
