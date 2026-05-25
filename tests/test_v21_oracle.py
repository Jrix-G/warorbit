import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import v21_oracle


def test_oracle_sample_labels_best_candidate():
    fs = v21_oracle._smoke_state()
    sample = v21_oracle.oracle_sample_from_state(fs, 0, episode_id="ep", horizon=4, top_k=4)
    advantages = [candidate["oracle_advantage"] for candidate in sample["candidates"]]
    assert sample["chosen"]["oracle_advantage"] == max(advantages)
    assert sample["chosen"] in sample["candidates"]


def test_oracle_rejects_bad_top_k():
    fs = v21_oracle._smoke_state()
    try:
        v21_oracle.oracle_sample_from_state(fs, 0, episode_id="ep", top_k=0)
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_combo_oracle_adds_soft_targets():
    fs = v21_oracle._smoke_state()
    sample = v21_oracle.combo_oracle_sample_from_state(fs, 0, episode_id="ep", horizon=4, top_k=4, beam_width=8)
    weights = [candidate.get("target_weight", 0.0) for candidate in sample["candidates"]]
    assert sum(weights) > 0.0
    assert sample["chosen"]["target_weight"] == max(weights)
    assert any(candidate.get("in_best_combo") for candidate in sample["candidates"])
