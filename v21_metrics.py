from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Mapping, Sequence


_EPS = 1e-12


@dataclass(frozen=True)
class WilsonInterval:
    wins: int
    total: int
    rate: float
    low: float
    high: float


@dataclass(frozen=True)
class BootstrapInterval:
    n: int
    mean: float
    low: float
    high: float


@dataclass(frozen=True)
class SeatSummary:
    seat: int
    total: int
    mean: float
    bias: float
    wins: int
    wilson: WilsonInterval


@dataclass(frozen=True)
class GameSummary:
    total: int
    wins: int
    winrate: WilsonInterval
    tie_score_mean: float
    tie_score_ci: BootstrapInterval
    mean_rank: float
    mean_rank_ci: BootstrapInterval
    avg_margin: float
    seats: tuple[SeatSummary, ...]


def wilson_ci(wins: int, total: int, z: float = 1.96) -> WilsonInterval:
    """Return a Wilson score interval for a binomial win count."""
    wins_i = int(wins)
    total_i = int(total)
    z_f = _finite_float(z, "z")
    if total_i < 0:
        raise ValueError("total must be non-negative")
    if wins_i < 0 or wins_i > total_i:
        raise ValueError("wins must satisfy 0 <= wins <= total")
    if total_i == 0:
        return WilsonInterval(wins_i, total_i, 0.0, 0.0, 0.0)
    p = wins_i / total_i
    z2 = z_f * z_f
    denom = 1.0 + z2 / total_i
    centre = (p + z2 / (2.0 * total_i)) / denom
    margin = z_f * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total_i)) / total_i) / denom
    return WilsonInterval(wins_i, total_i, p, max(0.0, centre - margin), min(1.0, centre + margin))


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    iters: int = 2000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> BootstrapInterval:
    """Return a deterministic bootstrap confidence interval for the mean."""
    data = [_finite_float(v, "value") for v in values]
    n = len(data)
    if n == 0:
        return BootstrapInterval(0, 0.0, 0.0, 0.0)
    if iters <= 0:
        raise ValueError("iters must be positive")
    conf = _finite_float(confidence, "confidence")
    if not 0.0 < conf < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    mean = sum(data) / n
    if n == 1:
        return BootstrapInterval(n, mean, mean, mean)

    rng = random.Random(int(seed))
    means = []
    for _ in range(int(iters)):
        total = 0.0
        for _j in range(n):
            total += data[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    alpha = (1.0 - conf) / 2.0
    lo = _quantile_sorted(means, alpha)
    hi = _quantile_sorted(means, 1.0 - alpha)
    return BootstrapInterval(n, mean, lo, hi)


def tie_aware_score(scores: Sequence[float], player: int) -> float:
    """Top-place score with ties split equally among tied winners."""
    vals = _score_values(scores)
    idx = _player_index(player, len(vals))
    best = max(vals)
    if best <= 0.0 or vals[idx] < best:
        return 0.0
    tied = sum(1 for value in vals if abs(value - best) <= _EPS)
    return 1.0 / max(1, tied)


def mean_rank(scores: Sequence[float], player: int) -> float:
    """One-based descending rank, averaging tied rank positions."""
    vals = _score_values(scores)
    idx = _player_index(player, len(vals))
    mine = vals[idx]
    better = sum(1 for value in vals if value > mine + _EPS)
    tied = sum(1 for value in vals if abs(value - mine) <= _EPS)
    first = better + 1
    last = better + tied
    return 0.5 * (first + last)


def score_margin(scores: Sequence[float], player: int) -> float:
    vals = _score_values(scores)
    idx = _player_index(player, len(vals))
    others = [value for pos, value in enumerate(vals) if pos != idx]
    return vals[idx] - max(others, default=0.0)


def seat_bias(
    rows: Iterable[Mapping[str, object]],
    *,
    seat_key: str = "our_seat",
    value_key: str = "tie_score",
    win_key: str = "win",
) -> tuple[SeatSummary, ...]:
    materialized = list(rows)
    if not materialized:
        return ()
    values = [_row_float(row, value_key, default=_row_float(row, win_key, default=0.0)) for row in materialized]
    overall = sum(values) / len(values)
    by_seat: dict[int, list[Mapping[str, object]]] = {}
    for row in materialized:
        if seat_key not in row:
            continue
        seat = int(row[seat_key])
        by_seat.setdefault(seat, []).append(row)

    summaries = []
    for seat in sorted(by_seat):
        seat_rows = by_seat[seat]
        seat_values = [
            _row_float(row, value_key, default=_row_float(row, win_key, default=0.0))
            for row in seat_rows
        ]
        total = len(seat_values)
        mean = sum(seat_values) / total if total else 0.0
        wins = sum(1 for row in seat_rows if _row_float(row, win_key, default=0.0) >= 1.0)
        summaries.append(SeatSummary(seat, total, mean, mean - overall, wins, wilson_ci(wins, total)))
    return tuple(summaries)


def game_row(scores: Sequence[float], player: int, *, seat: int | None = None, n_players: int | None = None) -> dict[str, float | int]:
    """Build a normalized metrics row from raw final scores."""
    vals = _score_values(scores)
    idx = _player_index(player, len(vals))
    row: dict[str, float | int] = {
        "n_players": int(n_players if n_players is not None else len(vals)),
        "player": idx,
        "win": 1 if tie_aware_score(vals, idx) == 1.0 else 0,
        "tie_score": tie_aware_score(vals, idx),
        "rank": mean_rank(vals, idx),
        "score_margin": score_margin(vals, idx),
    }
    if seat is not None:
        row["our_seat"] = int(seat)
    return row


def summarize_games(rows: Iterable[Mapping[str, object]], *, bootstrap_iters: int = 2000) -> GameSummary:
    materialized = list(rows)
    total = len(materialized)
    wins = sum(1 for row in materialized if _row_float(row, "win", default=0.0) >= 1.0)
    tie_scores = [_row_float(row, "tie_score", default=_row_float(row, "win", default=0.0)) for row in materialized]
    ranks = [_row_float(row, "rank", default=0.0) for row in materialized]
    margins = [_row_float(row, "score_margin", default=0.0) for row in materialized]
    return GameSummary(
        total=total,
        wins=wins,
        winrate=wilson_ci(wins, total),
        tie_score_mean=sum(tie_scores) / total if total else 0.0,
        tie_score_ci=bootstrap_mean_ci(tie_scores, iters=bootstrap_iters),
        mean_rank=sum(ranks) / total if total else 0.0,
        mean_rank_ci=bootstrap_mean_ci(ranks, iters=bootstrap_iters),
        avg_margin=sum(margins) / total if total else 0.0,
        seats=seat_bias(materialized),
    )


def _finite_float(value: object, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _score_values(scores: Sequence[float]) -> tuple[float, ...]:
    vals = tuple(_finite_float(value, "score") for value in scores)
    if len(vals) < 2:
        raise ValueError("at least two scores are required")
    return vals


def _player_index(player: int, n_players: int) -> int:
    idx = int(player)
    if idx < 0 or idx >= n_players:
        raise ValueError("player index out of range")
    return idx


def _row_float(row: Mapping[str, object], key: str, *, default: float) -> float:
    if key not in row or row[key] is None:
        return float(default)
    return _finite_float(row[key], key)


def _quantile_sorted(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    pos = min(max(q, 0.0), 1.0) * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    frac = pos - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)
