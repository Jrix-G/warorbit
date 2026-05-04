import importlib.util
import sys
from pathlib import Path


def _load_submission():
    path = Path(__file__).with_name("submission.py")
    spec = importlib.util.spec_from_file_location("submission_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_under_test"] = module
    spec.loader.exec_module(module)
    return module


def test_add_move_rejects_full_ray_near_sun_and_accepts_safe_target():
    sub = _load_submission()
    obs = {
        "player": 0,
        "step": 10,
        "angular_velocity": 0.0,
        "comets": [],
        "comet_planet_ids": [],
        "fleets": [],
        "planets": [
            [0, 0, 90.0, 90.0, 2.0, 100, 5],
            [1, -1, 10.0, 10.0, 2.0, 5, 1],
            [2, -1, 90.0, 10.0, 2.0, 5, 1],
            [3, 1, 10.0, 90.0, 2.0, 10, 5],
        ],
    }
    world = sub._build_world(obs)
    src = world.planet_by_id[0]

    moves = []
    assert not sub._add_move(moves, world, src, world.planet_by_id[1], 10)
    assert moves == []

    assert sub._add_move(moves, world, src, world.planet_by_id[2], 10)
    clearance = sub._path_sun_clearance(src, moves[0][1], sub.SUN_GUARD_RAY_DISTANCE)
    assert clearance >= sub.SUN_R + sub.SUN_SHOT_MARGIN
