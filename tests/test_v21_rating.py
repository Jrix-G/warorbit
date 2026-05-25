from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import v21_rating as rating


def test_2p_ratings_are_separate_from_4p_ratings() -> None:
    table = rating.V21Rating()
    table.record_game(["a", "b"], [10, 2])
    table.record_game(["a", "b", "c", "d"], [0, 4, 3, 2])

    assert table.rating("a", "2p").mu > table.rating("b", "2p").mu
    assert table.rating("a", "4p").mu < table.rating("b", "4p").mu
    assert table.games("a", "2p") == 1
    assert table.games("a", "4p") == 1


def test_4p_update_is_pairwise_and_tie_aware() -> None:
    table = rating.V21Rating()
    table.record_game(["winner", "tie_a", "tie_b", "last"], [9, 4, 4, 0])

    board = {row.name: row for row in table.leaderboard("4p")}
    assert board["winner"].mu > board["tie_a"].mu
    assert board["tie_a"].mu == pytest.approx(board["tie_b"].mu)
    assert board["tie_a"].mu > board["last"].mu
    assert board["winner"].games == 1


def test_snapshot_round_trips() -> None:
    table = rating.V21Rating()
    table.record_game(["a", "b"], [1, 0])
    snap = table.snapshot()

    loaded = rating.V21Rating()
    loaded.load_snapshot(snap)
    assert loaded.rating("a", "2p") == table.rating("a", "2p")
    assert loaded.games("a", "2p") == 1


def test_rate_games_and_validation() -> None:
    table = rating.rate_games([
        {"mode": "2p", "names": ["a", "b"], "scores": [1, 0]},
        {"n_players": 4, "names": ["a", "b", "c", "d"], "scores": [1, 2, 3, 4]},
    ])
    assert len(table.leaderboard("2p")) == 2
    assert len(table.leaderboard("4p")) == 4

    with pytest.raises(ValueError):
        table.record_game(["a", "b", "c"], [1, 2, 3])

    with pytest.raises(ValueError):
        table.record_pair("a", "b", 0.25, mode="2p")
