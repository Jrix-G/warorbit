import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import v15_fast_sim as fsim
import v21_search


def test_beam_combos_keeps_unique_sources():
    shots = [[0, 0.0, 5], [1, 0.1, 5], [0, 0.2, 5]]
    combos = v21_search._beam_combos(shots, max_combo=2, beam_width=10)
    assert combos
    for combo in combos:
        assert len({int(s[0]) for s in combo}) == len(combo)


def test_state_from_obs_uses_config_nplayers():
    obs = {
        "player": 0,
        "step": 0,
        "planets": [[0, 0, 10, 10, 3, 50, 2], [1, 3, 20, 10, 3, 50, 2]],
        "initial_planets": [[0, 0, 10, 10, 3, 50, 2], [1, 3, 20, 10, 3, 50, 2]],
        "fleets": [],
        "comets": [],
    }
    fs = v21_search._state_from_obs(obs, {"nPlayers": 4, "episodeSteps": 120})
    assert isinstance(fs, fsim.FastState)
    assert fs.n_players == 4
    assert fs.episode_steps == 120


def test_combo_rank_bonus_prefers_high_ranked_combo():
    bonus = {
        v21_search._shot_bonus_key([0, 0.0, 5]): 1.0,
        v21_search._shot_bonus_key([1, 0.1, 5]): 0.5,
    }
    high = v21_search._combo_rank_bonus([[0, 0.0, 5], [1, 0.1, 5]], bonus)
    low = v21_search._combo_rank_bonus([[2, 0.2, 5]], bonus)
    assert high > low
