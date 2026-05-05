"""Smoke tests for the local simulator folder."""

from __future__ import annotations

import unittest

import engine


class LocalSimulatorTests(unittest.TestCase):
    def test_new_state_and_observation(self):
        state = engine.new_state(seed=123)
        obs = engine.observation(state, 0)
        self.assertEqual(obs["player"], 0)
        self.assertGreaterEqual(len(obs["planets"]), 10)
        self.assertEqual(len(obs["fleets"]), 0)
        self.assertEqual(len(obs["initial_planets"]), len(obs["planets"]))

    def test_v9_bot_returns_legal_shape(self):
        state = engine.new_state(seed=123)
        actions = engine.bot_actions(state)
        self.assertIsInstance(actions, list)
        for action in actions:
            self.assertEqual(len(action), 3)
            self.assertIsInstance(int(action[0]), int)
            self.assertGreater(int(action[2]), 0)

    def test_human_action_and_advance(self):
        state = engine.new_state(seed=321)
        human_planets = [p for p in state.planets if int(p[engine.sim.P_OWNER]) == 0]
        targets = [p for p in state.planets if int(p[engine.sim.P_OWNER]) != 0]
        action = engine.make_human_action(state, int(human_planets[0][engine.sim.P_ID]), int(targets[0][engine.sim.P_ID]), 0.5)
        self.assertIsNotNone(action)
        next_state, result = engine.advance_turn(state, [action])
        self.assertEqual(next_state.step, 1)
        self.assertEqual(result.step, 1)
        self.assertGreaterEqual(len(result.human_actions), 1)


if __name__ == "__main__":
    unittest.main()

