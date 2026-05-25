from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, MutableMapping, Sequence

from training.trueskill_lite import MU0, SIGMA0, conservative_skill, update_match


@dataclass(frozen=True)
class Rating:
    mu: float = MU0
    sigma: float = SIGMA0

    @property
    def conservative(self) -> float:
        return conservative_skill((self.mu, self.sigma))

    def as_tuple(self) -> tuple[float, float]:
        return self.mu, self.sigma


@dataclass(frozen=True)
class LeaderboardRow:
    name: str
    mode: str
    mu: float
    sigma: float
    conservative: float
    games: int


def pairwise_result(scores: Sequence[float], left: int, right: int) -> float:
    vals = _scores(scores)
    li = _index(left, len(vals))
    ri = _index(right, len(vals))
    if vals[li] > vals[ri]:
        return 1.0
    if vals[li] < vals[ri]:
        return 0.0
    return 0.5


class V21Rating:
    """Separate 2p and 4p rating tables with pairwise 4p updates."""

    def __init__(self) -> None:
        self._ratings: dict[str, dict[str, Rating]] = {"2p": {}, "4p": {}}
        self._games: dict[str, dict[str, int]] = {"2p": {}, "4p": {}}

    def rating(self, name: str, mode: str | int) -> Rating:
        table = self._table(mode)
        key = _name(name)
        if key not in table:
            table[key] = Rating()
        return table[key]

    def games(self, name: str, mode: str | int) -> int:
        return self._games[_mode(mode)].get(_name(name), 0)

    def record_game(self, names: Sequence[str], scores: Sequence[float], *, mode: str | int | None = None) -> None:
        names_t = tuple(_name(name) for name in names)
        scores_t = _scores(scores)
        if len(names_t) != len(scores_t):
            raise ValueError("names and scores must have the same length")
        if len(names_t) not in (2, 4):
            raise ValueError("only 2p and 4p games are supported")
        mode_key = _mode(mode if mode is not None else len(names_t))
        expected = 2 if mode_key == "2p" else 4
        if len(names_t) != expected:
            raise ValueError(f"{mode_key} mode requires exactly {expected} players")

        if mode_key == "2p":
            self.record_pair(names_t[0], names_t[1], pairwise_result(scores_t, 0, 1), mode=mode_key)
        else:
            self._record_pairwise_batch(names_t, scores_t, mode_key)
        for name in names_t:
            self._games[mode_key][name] = self._games[mode_key].get(name, 0) + 1

    def record_pair(self, left: str, right: str, score_left: float, *, mode: str | int) -> None:
        mode_key = _mode(mode)
        score = _score_left(score_left)
        left_key = _name(left)
        right_key = _name(right)
        if left_key == right_key:
            raise ValueError("left and right names must differ")
        table = self._table(mode_key)
        left_rating = table.get(left_key, Rating())
        right_rating = table.get(right_key, Rating())
        new_left, new_right = update_match(left_rating.as_tuple(), right_rating.as_tuple(), score)
        table[left_key] = Rating(float(new_left[0]), float(new_left[1]))
        table[right_key] = Rating(float(new_right[0]), float(new_right[1]))

    def leaderboard(self, mode: str | int) -> tuple[LeaderboardRow, ...]:
        mode_key = _mode(mode)
        rows = [
            LeaderboardRow(
                name=name,
                mode=mode_key,
                mu=rating.mu,
                sigma=rating.sigma,
                conservative=rating.conservative,
                games=self._games[mode_key].get(name, 0),
            )
            for name, rating in self._ratings[mode_key].items()
        ]
        rows.sort(key=lambda row: (row.conservative, row.mu, row.name), reverse=True)
        return tuple(rows)

    def snapshot(self) -> dict[str, dict[str, dict[str, float | int]]]:
        out: dict[str, dict[str, dict[str, float | int]]] = {}
        for mode_key, table in self._ratings.items():
            out[mode_key] = {}
            for name, rating in table.items():
                out[mode_key][name] = {
                    "mu": rating.mu,
                    "sigma": rating.sigma,
                    "conservative": rating.conservative,
                    "games": self._games[mode_key].get(name, 0),
                }
        return out

    def load_snapshot(self, data: Mapping[str, Mapping[str, Mapping[str, object]]]) -> None:
        ratings: dict[str, dict[str, Rating]] = {"2p": {}, "4p": {}}
        games: dict[str, dict[str, int]] = {"2p": {}, "4p": {}}
        for mode_raw, table_raw in data.items():
            mode_key = _mode(mode_raw)
            for name_raw, row in table_raw.items():
                name = _name(name_raw)
                ratings[mode_key][name] = Rating(_float(row.get("mu", MU0), "mu"), _float(row.get("sigma", SIGMA0), "sigma"))
                games[mode_key][name] = int(row.get("games", 0) or 0)
        self._ratings = ratings
        self._games = games

    def _table(self, mode: str | int) -> MutableMapping[str, Rating]:
        return self._ratings[_mode(mode)]

    def _record_pairwise_batch(self, names: Sequence[str], scores: Sequence[float], mode_key: str) -> None:
        table = self._table(mode_key)
        base = {name: table.get(name, Rating()) for name in names}
        delta_mu = {name: 0.0 for name in names}
        delta_sigma = {name: 0.0 for name in names}

        for left, right in combinations(range(len(names)), 2):
            left_name = names[left]
            right_name = names[right]
            score_left = pairwise_result(scores, left, right)
            left_rating = base[left_name]
            right_rating = base[right_name]
            new_left, new_right = update_match(left_rating.as_tuple(), right_rating.as_tuple(), score_left)
            delta_mu[left_name] += float(new_left[0]) - left_rating.mu
            delta_sigma[left_name] += float(new_left[1]) - left_rating.sigma
            delta_mu[right_name] += float(new_right[0]) - right_rating.mu
            delta_sigma[right_name] += float(new_right[1]) - right_rating.sigma

        for name in names:
            start = base[name]
            table[name] = Rating(start.mu + delta_mu[name], max(1e-9, start.sigma + delta_sigma[name]))


def rate_games(rows: Iterable[Mapping[str, object]], *, name_key: str = "names", score_key: str = "scores") -> V21Rating:
    ratings = V21Rating()
    for row in rows:
        if name_key not in row or score_key not in row:
            raise ValueError(f"row must contain {name_key!r} and {score_key!r}")
        mode = row.get("mode", row.get("n_players"))
        ratings.record_game(_sequence(row[name_key], name_key), _sequence(row[score_key], score_key), mode=mode)
    return ratings


def _mode(mode: str | int | None) -> str:
    if mode is None:
        raise ValueError("mode is required")
    if isinstance(mode, str):
        raw = mode.strip().lower()
        if raw in {"2", "2p"}:
            return "2p"
        if raw in {"4", "4p"}:
            return "4p"
    if int(mode) == 2:
        return "2p"
    if int(mode) == 4:
        return "4p"
    raise ValueError("mode must be 2p or 4p")


def _name(name: object) -> str:
    out = str(name).strip()
    if not out:
        raise ValueError("agent name must be non-empty")
    return out


def _scores(scores: Sequence[float]) -> tuple[float, ...]:
    vals = tuple(_float(value, "score") for value in scores)
    if len(vals) < 2:
        raise ValueError("at least two scores are required")
    return vals


def _score_left(value: float) -> float:
    score = _float(value, "score_left")
    if score not in (0.0, 0.5, 1.0):
        raise ValueError("score_left must be 0.0, 0.5, or 1.0")
    return score


def _float(value: object, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if out != out or out in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return out


def _index(index: int, size: int) -> int:
    idx = int(index)
    if idx < 0 or idx >= size:
        raise ValueError("player index out of range")
    return idx


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return value
