import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import v21_policy_ranker
import v21_train_ranker


def _samples(n=16):
    dim = len(v21_policy_ranker.FEATURE_NAMES)
    good = [1.0] + [0.0] * (dim - 1)
    bad = [0.0] * dim
    out = []
    for i in range(n):
        chosen = {"shot": [0, 0.0, 10], "features": good}
        other = {"shot": [0, 1.0, 2], "features": bad}
        out.append(
            {
                "state": {"i": i},
                "candidates": [other, chosen],
                "chosen": chosen,
                "outcome": 1.0,
                "esc": 0.0,
                "episode_id": f"ep-{i}",
                "player": 0,
                "n_players": 2,
                "source": "test",
            }
        )
    return out


def test_train_ranker_learns_synthetic_signal():
    ranker, metrics = v21_train_ranker.train_ranker(_samples(), epochs=60, lr=0.3)
    assert metrics["top1"] == 1.0
    assert ranker.w[0] > 0.0


def test_save_and_load_ranker(tmp_path):
    ranker, _ = v21_train_ranker.train_ranker(_samples(), epochs=10)
    path = tmp_path / "ranker.npz"
    v21_train_ranker.save_ranker(path, ranker, metadata={"x": 1})
    loaded = v21_policy_ranker.LinearCandidateRanker.load(path)
    assert np.allclose(loaded.w, ranker.w, atol=1e-6)


def test_train_ranker_accepts_soft_targets():
    dim = len(v21_policy_ranker.FEATURE_NAMES)
    good = {"shot": [0, 0.0, 10], "features": [1.0] + [0.0] * (dim - 1), "target_weight": 0.8}
    weak = {"shot": [0, 1.0, 2], "features": [0.0] * dim, "target_weight": 0.2}
    samples = [
        {
            "state": {"i": i},
            "candidates": [weak, good],
            "chosen": good,
            "outcome": 1.0,
            "esc": 0.0,
            "episode_id": f"soft-{i}",
            "player": 0,
            "n_players": 2,
            "source": "test",
        }
        for i in range(8)
    ]
    ranker, metrics = v21_train_ranker.train_ranker(samples, epochs=40, lr=0.25)
    assert metrics["top1"] == 1.0
    assert ranker.w[0] > 0.0
