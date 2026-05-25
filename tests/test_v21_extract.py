import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import v15_fast_sim as fsim
import v21_extract


def _state():
    planets = np.array(
        [
            [0, 0, 10.0, 10.0, 3.0, 50.0, 2.0],
            [1, -1, 25.0, 10.0, 3.0, 10.0, 3.0],
            [2, 1, 40.0, 10.0, 3.0, 20.0, 2.0],
        ],
        dtype=np.float64,
    )
    return fsim.FastState(
        planets=planets,
        p_init=planets[:, [2, 3]].copy(),
        p_comet=np.zeros(3, dtype=bool),
        fleets=np.zeros((0, 7), dtype=np.float64),
        comets=[],
        step=3,
        angular_velocity=0.0,
        next_fleet_id=0,
        episode_steps=120,
        ship_speed=6.0,
        n_players=2,
    )


def test_sample_from_state_builds_canonical_sample():
    sample = v21_extract.sample_from_state(_state(), 0, [[0, 0.0, 20]], episode_id="ep", source="test", outcome=1.0)
    assert sample["episode_id"] == "ep"
    assert sample["candidates"]
    assert sample["chosen"] in sample["candidates"]
    assert sample["state"]["n_players"] == 2


def test_samples_from_rows_converts_multiple_rows():
    rows = [{"fs": _state(), "player": 0, "action": [[0, 0.0, 20]], "episode_id": "a", "outcome": 1.0}]
    samples = v21_extract.samples_from_rows(rows)
    assert len(samples) == 1
    assert samples[0]["source"] == "rows"
