import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import v15_fast_sim as fsim
import v21_policy_ranker as ranker


def _state():
    planets = np.array(
        [
            [0, 0, 10.0, 10.0, 3.0, 50.0, 2.0],
            [1, -1, 20.0, 10.0, 3.0, 12.0, 3.0],
            [2, 1, 35.0, 10.0, 3.0, 20.0, 2.0],
        ],
        dtype=np.float64,
    )
    return fsim.FastState(
        planets=planets,
        p_init=planets[:, [2, 3]].copy(),
        p_comet=np.zeros(3, dtype=bool),
        fleets=np.zeros((0, 7), dtype=np.float64),
        comets=[],
        step=25,
        angular_velocity=0.0,
        next_fleet_id=0,
        episode_steps=120,
        ship_speed=6.0,
        n_players=2,
    )


def test_ranker_resolves_and_scores_candidates():
    fs = _state()
    shots = [[0, 0.0, 20], [0, 0.1, 10]]
    ranked = ranker.rank_candidates(fs, 0, shots)
    assert len(ranked) == 2
    assert ranked[0].features.shape == (len(ranker.FEATURE_NAMES),)
    assert ranked[0].source_idx == 0
    assert ranked[0].target_idx in {1, 2}


def test_linear_ranker_rejects_bad_weight_shape():
    with np.testing.assert_raises(ValueError):
        ranker.LinearCandidateRanker(np.zeros(3))
