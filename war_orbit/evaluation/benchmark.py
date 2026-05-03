"""Benchmark utilities for V9."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from ..agents.v9.policy import V9Agent, V9Weights
from ..config.v9_config import V9Config
from ..training.curriculum import build_cross_play_specs
from ..training.self_play import evaluate_weights, summarise_results


V9_4P_LOG_TARGETS = {
    "xfer": 0.30,
    "bb": 0.15,
    "lock": 0.90,
    "fronts": 2.00,
}


@dataclass
class BenchmarkReport:
    summary: Dict[str, float]
    by_opponent: Dict[str, Dict[str, float]] = field(default_factory=dict)


def benchmark_v9(
    weights: V9Weights,
    config: V9Config,
    opponents: Sequence[str],
    *,
    games: int,
    max_steps: int,
    four_player_ratio: float,
    seed_offset: int = 70000,
) -> BenchmarkReport:
    specs = build_cross_play_specs(
        opponents,
        games=games,
        seed=config.seed,
        seed_offset=seed_offset,
        max_steps=max_steps,
        four_player_ratio=four_player_ratio,
        phase="benchmark",
    )
    pool = None
    if int(getattr(config, "workers", 1)) > 1:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=int(config.workers))
        print(f"V9 benchmark worker pool started workers={config.workers}", flush=True)
    try:
        results = evaluate_weights(
            weights,
            config,
            specs,
            progress_label="V9 benchmark",
            progress_every=max(1, int(getattr(config, "benchmark_progress_every", 1))),
            pool=pool,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    summary = summarise_results(results)
    by_opp: Dict[str, List[dict]] = {}
    for result in results:
        label = ",".join(result.get("opponents", []))
        by_opp.setdefault(label, []).append(result)
    return BenchmarkReport(summary=summary, by_opponent={k: summarise_results(v) for k, v in by_opp.items()})


def _format_four_p_diag(summary: Dict[str, float]) -> str:
    xfer = float(summary.get("transfer_move_frac", 0.0))
    bb = float(summary.get("backbone_turn_frac", 0.0))
    lock = float(summary.get("front_lock_turn_frac", 0.0))
    fronts = float(summary.get("active_front_avg", 0.0))
    ok = (
        xfer >= V9_4P_LOG_TARGETS["xfer"]
        and bb >= V9_4P_LOG_TARGETS["bb"]
        and lock >= V9_4P_LOG_TARGETS["lock"]
        and fronts <= V9_4P_LOG_TARGETS["fronts"]
    )
    return (
        f"4pdiag={'OK' if ok else 'WARN'} "
        f"xfer={xfer:.2f}/{V9_4P_LOG_TARGETS['xfer']:.2f}+ "
        f"bb={bb:.2f}/{V9_4P_LOG_TARGETS['bb']:.2f}+ "
        f"lock={lock:.2f}/{V9_4P_LOG_TARGETS['lock']:.2f}+ "
        f"fronts={fronts:.1f}/{V9_4P_LOG_TARGETS['fronts']:.1f}-"
    )


def print_report(report: BenchmarkReport) -> None:
    s = report.summary
    print(
        f"V9 eval mean={s['mean']:.3f} 2p={s['wr_2p']:.3f}/{s['n_2p']} "
        f"4p={s['wr_4p']:.3f}/{s['n_4p']} "
        f"{_format_four_p_diag(s)}",
        flush=True,
    )
    for label, stats in sorted(report.by_opponent.items()):
        print(
            f"  {label}: mean={stats['mean']:.3f} 2p={stats['wr_2p']:.3f}/{stats['n_2p']} "
            f"4p={stats['wr_4p']:.3f}/{stats['n_4p']} "
            f"{_format_four_p_diag(stats)}",
            flush=True,
        )
