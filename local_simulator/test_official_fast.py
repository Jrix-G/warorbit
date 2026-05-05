from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_ENV_ROOT = ROOT / "github" / "kaggle-environments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(KAGGLE_ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(KAGGLE_ENV_ROOT))

from kaggle_environments import make

from local_simulator.official_fast import (
    OfficialFastGame,
    comparable_state,
    deterministic_expand_agent,
    noop_agent,
)


class OfficialFastGameTests(unittest.TestCase):
    def assert_matches_kaggle(self, agents, *, n_players=2, seed=0, episode_steps=120):
        env = make(
            "orbit_wars",
            configuration={"episodeSteps": episode_steps, "seed": seed},
            debug=False,
        )
        kaggle_steps = env.run(list(agents))

        fast = OfficialFastGame(n_players, seed=seed, episode_steps=episode_steps, use_c_accel=False)
        fast_steps = fast.run(list(agents))

        self.assertEqual(len(fast_steps), len(kaggle_steps))
        self.assertEqual(comparable_state(fast_steps[-1]), comparable_state(kaggle_steps[-1]))

        for idx in (0, 1, len(kaggle_steps) // 2, len(kaggle_steps) - 1):
            self.assertEqual(
                comparable_state(fast_steps[idx]),
                comparable_state(kaggle_steps[idx]),
                f"state mismatch at step index {idx}",
            )

    def test_noop_2p_exact(self):
        for seed in range(4):
            self.assert_matches_kaggle([noop_agent, noop_agent], seed=seed)

    def test_deterministic_expand_2p_exact(self):
        for seed in range(4):
            self.assert_matches_kaggle(
                [deterministic_expand_agent, deterministic_expand_agent],
                seed=seed,
            )

    def test_noop_4p_exact(self):
        self.assert_matches_kaggle(
            [noop_agent, noop_agent, noop_agent, noop_agent],
            n_players=4,
            seed=7,
            episode_steps=100,
        )


if __name__ == "__main__":
    unittest.main()
