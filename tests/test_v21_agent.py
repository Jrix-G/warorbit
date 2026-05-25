import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import v21_agent
import v21_oracle


def test_4p_guard_enabled_by_default(monkeypatch):
    monkeypatch.delenv("V21_DISABLE_4P_GUARD", raising=False)
    monkeypatch.delenv("V21_FORCE_LEARNED", raising=False)
    assert v21_agent._use_4p_guard({"n_players": 4, "planets": []}, None)


def test_4p_guard_can_be_disabled(monkeypatch):
    monkeypatch.setenv("V21_FORCE_LEARNED", "1")
    assert not v21_agent._use_4p_guard({"n_players": 4, "planets": []}, None)


def test_normalise_move_drops_invalid_shots():
    move = [[1, 0.5, 4], ["bad"], [2, 1.0, 0], [3, "2.0", "5"]]
    assert v21_agent._normalise_move(move) == [[1, 0.5, 4], [3, 2.0, 5]]


def test_move_key_ignores_order():
    left = [[2, 0.2, 5], [1, 0.1, 3]]
    right = [[1, 0.1, 3], [2, 0.2, 5]]
    assert v21_agent._move_key(left) == v21_agent._move_key(right)


def test_det_eval_returns_finite_score():
    fs = v21_oracle._smoke_state()
    score = v21_agent._eval_combo_det(fs, 0, [], 3)
    assert 0.0 <= score <= 1.0
