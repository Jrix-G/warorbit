from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import v21_metrics as metrics


def test_wilson_ci_handles_empty_and_validates_counts() -> None:
    empty = metrics.wilson_ci(0, 0)
    assert empty.rate == 0.0
    assert empty.low == 0.0
    assert empty.high == 0.0

    ci = metrics.wilson_ci(7, 10)
    assert ci.rate == 0.7
    assert 0.0 <= ci.low < ci.rate < ci.high <= 1.0

    with pytest.raises(ValueError):
        metrics.wilson_ci(11, 10)


def test_tie_aware_score_and_mean_rank_split_ties() -> None:
    scores = [12, 12, 5, 0]
    assert metrics.tie_aware_score(scores, 0) == 0.5
    assert metrics.tie_aware_score(scores, 1) == 0.5
    assert metrics.tie_aware_score(scores, 2) == 0.0
    assert metrics.mean_rank(scores, 0) == 1.5
    assert metrics.mean_rank(scores, 2) == 3.0


def test_game_row_and_summary_include_bootstrap_and_seat_bias() -> None:
    rows = [
        metrics.game_row([10, 5], 0, seat=0),
        metrics.game_row([3, 8], 0, seat=1),
        metrics.game_row([7, 7], 0, seat=0),
    ]

    summary = metrics.summarize_games(rows, bootstrap_iters=200)
    assert summary.total == 3
    assert summary.wins == 1
    assert summary.winrate.wins == 1
    assert summary.tie_score_mean == pytest.approx(0.5)
    assert summary.mean_rank == pytest.approx((1.0 + 2.0 + 1.5) / 3.0)
    assert summary.avg_margin == pytest.approx((5.0 - 5.0 + 0.0) / 3.0)
    assert [seat.seat for seat in summary.seats] == [0, 1]
    assert summary.seats[0].mean == pytest.approx(0.75)
    assert summary.seats[1].mean == pytest.approx(0.0)


def test_bootstrap_mean_ci_is_deterministic() -> None:
    first = metrics.bootstrap_mean_ci([0.0, 1.0, 1.0, 0.0], iters=300, seed=7)
    second = metrics.bootstrap_mean_ci([0.0, 1.0, 1.0, 0.0], iters=300, seed=7)
    assert first == second
    assert first.mean == 0.5
    assert first.low <= first.mean <= first.high
