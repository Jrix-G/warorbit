from __future__ import annotations

import random
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
from kaggle_environments.envs.orbit_wars import orbit_wars

from local_simulator.c_accel import orbit_wars_accel
from local_simulator.official_fast import (
    OfficialFastGame,
    comparable_state,
    deterministic_expand_agent,
    noop_agent,
)


@unittest.skipIf(orbit_wars_accel.load_c_module() is None, "C accelerator is not built")
class OrbitWarsCAccelTests(unittest.TestCase):
    def test_generate_comet_paths_exact(self):
        original = orbit_wars.generate_comet_paths
        c_module = orbit_wars_accel.load_c_module()
        assert c_module is not None

        for seed in range(40):
            rng = random.Random(seed)
            planets = orbit_wars.generate_planets(rng)
            angular_velocity = rng.uniform(0.025, 0.05)
            for spawn_step in (50, 150, 250):
                py_rng = random.Random(f"orbit_wars-comet-{seed}-{spawn_step}")
                c_rng = random.Random(f"orbit_wars-comet-{seed}-{spawn_step}")
                expected = original(
                    planets,
                    angular_velocity,
                    spawn_step,
                    [],
                    4.0,
                    rng=py_rng,
                )
                actual = c_module.generate_comet_paths(
                    planets,
                    angular_velocity,
                    spawn_step,
                    [],
                    4.0,
                    rng=c_rng,
                )
                self.assertEqual(actual, expected, f"seed={seed} spawn={spawn_step}")

    def assert_c_runner_matches_kaggle(self, agents, *, n_players=2, seed=0, episode_steps=160):
        env = make(
            "orbit_wars",
            configuration={"episodeSteps": episode_steps, "seed": seed},
            debug=False,
        )
        kaggle_steps = env.run(list(agents))

        fast = OfficialFastGame(n_players, seed=seed, episode_steps=episode_steps, use_c_accel=True)
        self.assertTrue(fast.c_accel_enabled)
        fast_steps = fast.run(list(agents))

        self.assertEqual(len(fast_steps), len(kaggle_steps))
        for idx in (0, 1, len(kaggle_steps) // 2, len(kaggle_steps) - 1):
            self.assertEqual(
                comparable_state(fast_steps[idx]),
                comparable_state(kaggle_steps[idx]),
                f"state mismatch at step index {idx}",
            )

    def test_c_runner_noop_exact(self):
        for seed in range(6):
            self.assert_c_runner_matches_kaggle([noop_agent, noop_agent], seed=seed)

    def test_c_runner_expand_exact(self):
        for seed in range(6):
            self.assert_c_runner_matches_kaggle(
                [deterministic_expand_agent, deterministic_expand_agent],
                seed=seed,
            )


if __name__ == "__main__":
    unittest.main()

