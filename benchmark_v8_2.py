#!/usr/bin/env python3
"""Reproducible local benchmark for bot_v8_2.

SimGame-based — fast smoke gate before Kaggle. Reports:

* per-opponent / per-mode WR
* aggregate 2p WR, 4p WR, global WR
* time per game (median, p95)
* plan-choice histogram (proxy for ranker behavior)
"""

from __future__ import annotations

import argparse
import io
import statistics
import sys
import time
import multiprocessing as mp
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Worker process: imports done once, plan logging done in-process.
# ---------------------------------------------------------------------------

_WORKER_BOT = None
_WORKER_RUN = None
_WORKER_ZOO = None
_WORKER_V7 = None
_PLAN_COUNTER: Counter = Counter()
_CANDIDATE_COUNTER: Counter = Counter()


def _silent_imports():
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    try:
        import bot_v7  # noqa: F401
        import bot_v8_2  # noqa: F401
        from SimGame import run_match  # noqa: F401
        from opponents import ZOO  # noqa: F401
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return bot_v8_2, bot_v7, run_match, ZOO


def _worker_init():
    global _WORKER_BOT, _WORKER_RUN, _WORKER_ZOO, _WORKER_V7, _PLAN_COUNTER, _CANDIDATE_COUNTER
    _WORKER_BOT, _WORKER_V7, _WORKER_RUN, _WORKER_ZOO = _silent_imports()
    _PLAN_COUNTER = Counter()
    _CANDIDATE_COUNTER = Counter()

    def _log(d):
        names = d.get("candidate_names") or []
        chosen = int(d.get("chosen", 0))
        _CANDIDATE_COUNTER.update(names)
        if 0 <= chosen < len(names):
            _PLAN_COUNTER[names[chosen]] += 1

    _WORKER_BOT.set_candidate_log_callback(_log)


def _agent_for(name: str):
    if name == "bot_v7":
        return _WORKER_V7.agent
    if name == "bot_v8_2":
        return _WORKER_BOT.agent
    return _WORKER_ZOO[name]


def _worker_play(task):
    global _PLAN_COUNTER, _CANDIDATE_COUNTER
    label, opp_names, our_index, seed, max_steps = task
    if _WORKER_BOT is None:
        _worker_init()
    _PLAN_COUNTER = Counter()
    _CANDIDATE_COUNTER = Counter()
    agents = []
    opp_iter = iter(opp_names)
    for slot in range(len(opp_names) + 1):
        if slot == our_index:
            agents.append(_WORKER_BOT.agent)
        else:
            agents.append(_agent_for(next(opp_iter)))
    started = time.perf_counter()
    result = _WORKER_RUN(agents, seed=seed, n_players=len(agents), max_steps=max_steps)
    elapsed = time.perf_counter() - started
    winner = int(result.get("winner", -1))
    if winner == our_index:
        outcome = 1
    elif winner < 0:
        outcome = 0
    else:
        outcome = -1
    plan_snapshot = dict(_PLAN_COUNTER)
    candidate_snapshot = dict(_CANDIDATE_COUNTER)
    return label, outcome, elapsed, int(result.get("steps", 0)), plan_snapshot, candidate_snapshot


# ---------------------------------------------------------------------------
# Stats helpers.
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    seconds: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def wr(self) -> float:
        return self.wins / max(1, self.games)


def _rotating_4p_names(anchor: str, pool: Sequence[str], game_idx: int) -> List[str]:
    """Pick three opponents — anchor + 2 rotating buddies from the pool."""
    others = [name for name in pool if name != anchor]
    if not others:
        others = [anchor]
    pick = []
    for k in range(2):
        pick.append(others[(game_idx + k) % len(others)])
    return [anchor] + pick


def build_tasks(opponents: Sequence[str], games_per_opp: int, mode: str,
                seed_offset: int, max_steps: int):
    tasks = []
    pool = list(opponents) or ["bot_v7"]
    for opp in opponents:
        for i in range(games_per_opp):
            use_4p = mode == "4p" or (mode == "mixed" and i % 2 == 1)
            seed = seed_offset + len(tasks)
            if use_4p:
                opp_names = _rotating_4p_names(opp, pool, i + 1)
                our_index = i % 4
                label = f"4p:{opp}"
            else:
                opp_names = [opp]
                our_index = i % 2
                label = f"2p:{opp}"
            tasks.append((label, opp_names, our_index, seed, max_steps))
    return tasks


def _aggregate(per_label: Dict[str, Stats], total: Stats) -> Tuple[Stats, Stats]:
    s2 = Stats()
    s4 = Stats()
    for label, stats in per_label.items():
        target = s2 if label.startswith("2p:") else s4
        target.wins += stats.wins
        target.losses += stats.losses
        target.draws += stats.draws
        target.seconds.extend(stats.seconds)
        target.steps.extend(stats.steps)
    return s2, s4


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    k = max(0, min(len(v) - 1, int(round(p * (len(v) - 1)))))
    return v[k]


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

DEFAULT_OPPONENTS = [
    "notebook_orbitbotnext",
    "notebook_distance_prioritized",
    "notebook_physics_accurate",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark bot_v8_2 against local opponents.")
    parser.add_argument("--games-per-opp", type=int, default=4)
    parser.add_argument("--mode", choices=("2p", "4p", "mixed"), default="mixed")
    parser.add_argument("--opponents", nargs="*", default=DEFAULT_OPPONENTS)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--seed-offset", type=int, default=8200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pool-limit", type=int, default=0,
                        help="Use first N training_pool opponents instead of --opponents.")
    parser.add_argument("--no-plan-histogram", action="store_true")
    args = parser.parse_args()

    # Lazy imports so worker initializer mirrors main process imports.
    bot_v8_2_main, bot_v7_main, run_match, ZOO_main = _silent_imports()
    from opponents import training_pool

    opponents = training_pool(args.pool_limit) if args.pool_limit > 0 else list(args.opponents)
    missing = [name for name in opponents if name not in ZOO_main and name not in {"bot_v7", "bot_v8_2"}]
    if missing:
        raise SystemExit(f"Unknown opponents: {missing}")

    tasks = build_tasks(opponents, args.games_per_opp, args.mode, args.seed_offset, args.max_steps)
    print(
        f"V8.2 benchmark | mode={args.mode} games={len(tasks)} "
        f"opponents={len(opponents)} max_steps={args.max_steps} workers={args.workers}",
        flush=True,
    )

    if args.workers <= 1:
        _worker_init()
        outcomes = [_worker_play(task) for task in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers, initializer=_worker_init) as pool:
            outcomes = list(pool.map(_worker_play, tasks))

    per_label: Dict[str, Stats] = defaultdict(Stats)
    total = Stats()
    plan_total: Counter = Counter()
    candidate_total: Counter = Counter()
    for label, outcome, elapsed, steps, plan_snapshot, candidate_snapshot in outcomes:
        stats = per_label[label]
        stats.seconds.append(elapsed)
        stats.steps.append(steps)
        total.seconds.append(elapsed)
        total.steps.append(steps)
        if outcome > 0:
            stats.wins += 1
            total.wins += 1
        elif outcome < 0:
            stats.losses += 1
            total.losses += 1
        else:
            stats.draws += 1
            total.draws += 1
        plan_total.update(plan_snapshot)
        candidate_total.update(candidate_snapshot)

    for label in sorted(per_label):
        s = per_label[label]
        med = statistics.median(s.seconds) if s.seconds else 0.0
        print(f"- {label:38s} W/L/D={s.wins:3d}/{s.losses:3d}/{s.draws:3d} "
              f"WR={s.wr * 100:5.1f}%  median={med:5.1f}s")

    s2, s4 = _aggregate(per_label, total)
    p95 = _percentile(total.seconds, 0.95)
    med = statistics.median(total.seconds) if total.seconds else 0.0
    print(
        f"-- 2p: {s2.wins}/{s2.losses}/{s2.draws} WR={s2.wr * 100:5.1f}%  "
        f"4p: {s4.wins}/{s4.losses}/{s4.draws} WR={s4.wr * 100:5.1f}%"
    )
    print(
        f"Global W/L/D={total.wins}/{total.losses}/{total.draws} "
        f"WR={total.wr * 100:.1f}%  median={med:.1f}s p95={p95:.1f}s"
    )

    if not args.no_plan_histogram and plan_total:
        # The histogram is the union of per-worker plan choices over the games we
        # actually played. Useful to confirm the ranker isn't collapsing to a
        # single plan even when zero-init.
        total_picks = sum(plan_total.values())
        print("Plan-choice histogram:")
        for plan, count in sorted(plan_total.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * count / max(1, total_picks)
            print(f"  {plan:24s} {count:6d}  {pct:5.1f}%")
    if not args.no_plan_histogram and candidate_total:
        total_seen = sum(candidate_total.values())
        print("Candidate availability histogram:")
        for plan, count in sorted(candidate_total.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * count / max(1, total_seen)
            print(f"  {plan:24s} {count:6d}  {pct:5.1f}%")


if __name__ == "__main__":
    main()
