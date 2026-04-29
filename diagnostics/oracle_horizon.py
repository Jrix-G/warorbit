#!/usr/bin/env python3
"""Measure whether the V8 short-rollout oracle carries the right ranking signal.

Hypothesis under test
---------------------
`step_policy_episode(steps=H)` is used at training time to label which of the
six candidate plans is "best" at a given state. The training loop uses H=2..4.
If that horizon is too short for fleets to actually arrive, the oracle ranks
candidates on production noise and ship conservation rather than on plan
quality, which would explain the V8.1 plateau (rising acc, flat win rate).

Protocol
--------
1. Generate N decision states by running bot_v7 vs a reference opponent and
   snapshotting at a spread of turns (early/mid/late).
2. At each state, build the 6 V8 candidate plans.
3. Score every (state, plan) pair at multiple horizons H using the exact same
   `step_policy_episode` that training uses.
4. Report, per H:
   - Spearman rank correlation against the longest horizon (the reference).
   - Top-1 agreement with the longest horizon.
   - The "passivity bias": Spearman between oracle value and ship spend.
   - The fraction of states where all candidates score within +/- epsilon
     (label noise floor).

Verdict
-------
A diagnostic line at the end translates the numbers into a recommendation
("oracle horizon H>=K is required" or "oracle is fine, look elsewhere").

Usage
-----
    python -m diagnostics.oracle_horizon --n-states 200 --workers 8
    python -m diagnostics.oracle_horizon --quick   # 40 states, single process
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

# Make `python -m diagnostics.oracle_horizon` and direct invocation both work.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import bot_v7  # noqa: E402
import bot_v8  # noqa: E402
from SimGame import SimGame, FastState  # noqa: E402
from v8_core import (  # noqa: E402
    LinearV8Model,
    build_candidate_plans,
    step_policy_episode,
)


DEFAULT_HORIZONS = (2, 5, 15, 40, 100)
DEFAULT_OPPONENT = "notebook_physics_accurate"
SAMPLE_TURNS = (30, 60, 100, 150, 220)  # well-spread decision points


# ----------------------------- data structures ----------------------------- #


@dataclass
class StateSnapshot:
    """A frozen game state and its decision context.

    We pickle FastState dicts (not SimGame objects) for cheap multiprocessing
    transfer; SimGame is reconstructed inside workers.
    """

    seed: int
    turn: int
    state_blob: dict   # FastState fields, primitives + lists
    n_players: int
    opponent_name: str


@dataclass
class StateResult:
    seed: int
    turn: int
    n_plans: int
    total_ships_per_plan: List[float]
    values_by_horizon: dict          # {H: [v_plan_0, ..., v_plan_k]}
    elapsed_s: float


# --------------------------- opponent / policies --------------------------- #


def _load_opponent(name: str) -> Callable:
    from opponents import ZOO

    if name not in ZOO:
        raise SystemExit(
            f"unknown opponent {name!r}; available: {sorted(ZOO.keys())}"
        )
    return ZOO[name]


def _make_obs_opponent_policy(agent: Callable) -> Callable:
    def policy(game: SimGame, player: int) -> list:
        obs = game.observation(player)
        try:
            move = agent(obs, None)
        except TypeError:
            move = agent(obs)
        return move if isinstance(move, list) else []
    return policy


def _make_rollout_policy(train_player: int, opponent_policy: Callable) -> Callable:
    """Same shape as train_v8._make_rollout_policy; kept local to avoid coupling."""
    def policy(game: SimGame, player: int) -> list:
        if player == train_player:
            return bot_v8.state_policy(game, player)
        return opponent_policy(game, player)
    return policy


# ----------------------------- state harvesting ---------------------------- #


def _state_to_blob(state: FastState) -> dict:
    return {
        "planets": state.planets.tolist(),
        "fleets": state.fleets.tolist(),
        "initial_planets": state.initial_planets.tolist(),
        "angular_velocity": float(state.angular_velocity),
        "step": int(state.step),
        "next_fleet_id": int(state.next_fleet_id),
        "max_steps": int(state.max_steps),
    }


def _blob_to_state(blob: dict) -> FastState:
    return FastState(
        planets=np.asarray(blob["planets"], dtype=np.float32),
        fleets=np.asarray(blob["fleets"], dtype=np.float32) if blob["fleets"] else np.zeros((0, 7), dtype=np.float32),
        initial_planets=np.asarray(blob["initial_planets"], dtype=np.float32),
        angular_velocity=float(blob["angular_velocity"]),
        step=int(blob["step"]),
        next_fleet_id=int(blob["next_fleet_id"]),
        max_steps=int(blob["max_steps"]),
    )


def harvest_states(
    n_states: int,
    opponent_name: str,
    seed_start: int = 0,
    sample_turns: Sequence[int] = SAMPLE_TURNS,
    n_players: int = 2,
    neutral_pairs: int = 8,
) -> List[StateSnapshot]:
    """Run V7 vs opponent and snapshot the state at the configured turns.

    One game yields up to len(sample_turns) snapshots, alternating which player
    is "us" so the harvested distribution is symmetric.
    """
    opponent = _load_opponent(opponent_name)
    opp_policy = _make_obs_opponent_policy(opponent)

    snapshots: List[StateSnapshot] = []
    seed = seed_start
    games_per_loop = max(1, math.ceil(n_states / max(1, len(sample_turns))))
    target_turns = sorted(set(int(t) for t in sample_turns))

    while len(snapshots) < n_states:
        for game_index in range(games_per_loop):
            if len(snapshots) >= n_states:
                break
            our_player = (seed + game_index) % n_players
            game = SimGame.random_game(
                seed=seed + game_index,
                n_players=n_players,
                neutral_pairs=neutral_pairs,
            )
            remaining_targets = list(target_turns)
            while not game.is_terminal() and remaining_targets:
                if game.state.step >= remaining_targets[0]:
                    snapshots.append(StateSnapshot(
                        seed=seed + game_index,
                        turn=int(game.state.step),
                        state_blob=_state_to_blob(game.state),
                        n_players=n_players,
                        opponent_name=opponent_name,
                    ))
                    remaining_targets.pop(0)
                    if len(snapshots) >= n_states:
                        break
                actions = {}
                for player in range(n_players):
                    obs = game.observation(player)
                    if player == our_player:
                        try:
                            move = bot_v7.agent(obs, None)
                        except Exception:
                            move = []
                    else:
                        move = opp_policy(game, player)
                    actions[player] = move if isinstance(move, list) else []
                game.step(actions)
        seed += games_per_loop

    return snapshots[:n_states]


# ------------------------------ scoring core ------------------------------- #


def _evaluate_state(args: Tuple[StateSnapshot, Tuple[int, ...]]) -> StateResult:
    """Worker: score every (plan, horizon) for one snapshot.

    Runs in a child process. Imports are at module top so the pool inherits
    them; we only rebuild lightweight objects here.
    """
    snap, horizons = args
    t0 = time.perf_counter()

    # The worker uses a clean zero-weight model; the diagnostic explicitly
    # measures the *oracle*, not whatever weights happen to be cached.
    bot_v8.set_model(LinearV8Model.zero())
    bot_v7.set_scorer(None)

    state = _blob_to_state(snap.state_blob)
    game = SimGame(state, n_players=snap.n_players)
    obs = game.observation(0)  # arbitrary; we just need a world for plans
    world = bot_v7._build_world(obs)
    plans = build_candidate_plans(world)

    opponent = _load_opponent(snap.opponent_name)
    opp_policy = _make_obs_opponent_policy(opponent)
    rollout_policy = _make_rollout_policy(train_player=0, opponent_policy=opp_policy)

    total_ships = []
    for plan in plans:
        s = 0.0
        for action in plan.actions or []:
            if action and len(action) == 3:
                s += float(action[2])
        total_ships.append(s)

    values_by_horizon: dict = {}
    for H in horizons:
        vals = []
        for plan in plans:
            v = step_policy_episode(
                game,
                our_player=0,
                our_actions=plan.actions or [],
                opponent_action_fn=rollout_policy,
                steps=int(H),
            )
            vals.append(float(v))
        values_by_horizon[int(H)] = vals

    return StateResult(
        seed=snap.seed,
        turn=snap.turn,
        n_plans=len(plans),
        total_ships_per_plan=total_ships,
        values_by_horizon=values_by_horizon,
        elapsed_s=time.perf_counter() - t0,
    )


# ------------------------------ statistics --------------------------------- #


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rho without scipy. Returns NaN on degenerate input."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.size != b.size:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")  # constant input -> rank correlation undefined
    ra = _ranks(a)
    rb = _ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float(np.dot(ra, ra)) * float(np.dot(rb, rb)))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(ra, rb) / denom)


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average-rank for ties (matches scipy.stats.rankdata default)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    n = len(x)
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def _safe_nanmean(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if not (isinstance(v, float) and math.isnan(v))], dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


# ------------------------------ aggregation -------------------------------- #


def _maybe_margin(values: List[float], oracle_mode: str, baseline_idx: int) -> List[float]:
    """In margin mode, subtract the baseline plan's value (control variate).

    The argmax is invariant under this transform, but the metric we report
    later (passivity bias, label-noise span) does change because constants
    common to all plans are removed before measurement.
    """
    if oracle_mode == "absolute" or not values or baseline_idx >= len(values):
        return list(values)
    base = float(values[baseline_idx])
    return [float(v) - base for v in values]


def aggregate(
    results: List[StateResult],
    horizons: Sequence[int],
    tie_eps: float,
    oracle_mode: str = "absolute",
    baseline_idx: int = 0,
) -> dict:
    H_ref = max(horizons)
    rows = []
    for H in horizons:
        if H == H_ref:
            continue
        per_state_rho = []
        per_state_top1 = []
        for r in results:
            v_short = _maybe_margin(r.values_by_horizon.get(H, []), oracle_mode, baseline_idx)
            v_ref = _maybe_margin(r.values_by_horizon.get(H_ref, []), oracle_mode, baseline_idx)
            if len(v_short) < 2 or len(v_ref) != len(v_short):
                continue
            per_state_rho.append(_spearman(v_short, v_ref))
            per_state_top1.append(1.0 if int(np.argmax(v_short)) == int(np.argmax(v_ref)) else 0.0)
        rows.append({
            "horizon": int(H),
            "spearman_vs_ref": _safe_nanmean(per_state_rho),
            "top1_vs_ref": _safe_nanmean(per_state_top1),
            "n_states": len(per_state_rho),
        })

    # Passivity bias: per-state Spearman(value@H, -total_ships).
    # Positive rho => fewer ships sent => higher oracle value (i.e. the bias).
    bias_rows = []
    for H in horizons:
        per_state_bias = []
        for r in results:
            v = _maybe_margin(r.values_by_horizon.get(H, []), oracle_mode, baseline_idx)
            if len(v) < 2 or len(r.total_ships_per_plan) != len(v):
                continue
            neg_ships = [-x for x in r.total_ships_per_plan]
            per_state_bias.append(_spearman(v, neg_ships))
        bias_rows.append({
            "horizon": int(H),
            "passivity_rho": _safe_nanmean(per_state_bias),
            "n_states": len(per_state_bias),
        })

    # Label-noise floor: fraction of states where max(values) - min(values) < tie_eps.
    noise_rows = []
    for H in horizons:
        flats = []
        spans = []
        for r in results:
            v = _maybe_margin(r.values_by_horizon.get(H, []), oracle_mode, baseline_idx)
            if not v:
                continue
            span = float(max(v) - min(v))
            spans.append(span)
            flats.append(1.0 if span < tie_eps else 0.0)
        noise_rows.append({
            "horizon": int(H),
            "frac_flat": _safe_nanmean(flats),
            "mean_span": _safe_nanmean(spans),
            "n_states": len(flats),
        })

    return {
        "reference_horizon": int(H_ref),
        "oracle_mode": oracle_mode,
        "ranking_agreement": rows,
        "passivity_bias": bias_rows,
        "label_noise": noise_rows,
        "n_states": len(results),
    }


# ------------------------------ presentation ------------------------------- #


def _print_table(title: str, rows: List[dict], columns: List[Tuple[str, str, str]]) -> None:
    """columns is a list of (key, header, fmt)."""
    print(f"\n{title}")
    print("-" * len(title))
    headers = "  ".join(f"{h:>14s}" for _, h, _ in columns)
    print(headers)
    for row in rows:
        cells = []
        for key, _, fmt in columns:
            v = row.get(key)
            if v is None:
                cells.append(f"{'-':>14s}")
            elif isinstance(v, float) and math.isnan(v):
                cells.append(f"{'nan':>14s}")
            else:
                cells.append(format(v, fmt).rjust(14))
        print("  ".join(cells))


def render_report(report: dict, tie_eps: float) -> None:
    H_ref = report["reference_horizon"]
    mode = report.get("oracle_mode", "absolute")
    print(f"\n=== Oracle horizon diagnostic [{mode}] "
          f"(reference H={H_ref}, n_states={report['n_states']}) ===")

    _print_table(
        f"Ranking agreement vs H={H_ref}",
        report["ranking_agreement"],
        [
            ("horizon", "horizon", "d"),
            ("spearman_vs_ref", "spearman_rho", ".3f"),
            ("top1_vs_ref", "top1_match", ".3f"),
            ("n_states", "n", "d"),
        ],
    )

    _print_table(
        "Passivity bias  (rho>0 means: oracle prefers plans that spend fewer ships)",
        report["passivity_bias"],
        [
            ("horizon", "horizon", "d"),
            ("passivity_rho", "passivity_rho", ".3f"),
            ("n_states", "n", "d"),
        ],
    )

    _print_table(
        f"Label-noise floor  (frac_flat = states where value-span < {tie_eps:.3f})",
        report["label_noise"],
        [
            ("horizon", "horizon", "d"),
            ("frac_flat", "frac_flat", ".3f"),
            ("mean_span", "mean_span", ".4f"),
            ("n_states", "n", "d"),
        ],
    )

    print()
    _print_verdict(report)


def _pick(rows: List[dict], horizon: int, key: str) -> Optional[float]:
    for r in rows:
        if r["horizon"] == horizon:
            v = r.get(key)
            if isinstance(v, float) and math.isnan(v):
                return None
            return v
    return None


def _pick_short(rows: List[dict], key: str) -> Optional[float]:
    short = sorted(r["horizon"] for r in rows)[:1]
    if not short:
        return None
    return _pick(rows, short[0], key)


def _verdict_lines(report: dict) -> List[str]:
    lines: List[str] = []
    rank_rows = report["ranking_agreement"]
    bias_rows = report["passivity_bias"]
    noise_rows = report["label_noise"]
    H_ref = report["reference_horizon"]

    rho_short = _pick_short(rank_rows, "spearman_vs_ref")
    if rho_short is None:
        lines.append("Inconclusive: ranking agreement could not be computed.")
    elif rho_short < 0.30:
        lines.append(
            f"FINDING #1 confirmed: rank correlation between shortest H and reference H={H_ref} "
            f"is {rho_short:.2f} (<0.30). The training oracle does not preserve the long-horizon "
            "ranking. Increase --rollout-steps in train_v8.py."
        )
    elif rho_short < 0.60:
        lines.append(
            f"Partial confirmation: short-horizon rho={rho_short:.2f}. The oracle is noisy but "
            "carries signal; expect slow learning rather than systematic miscoding."
        )
    else:
        lines.append(
            f"Oracle horizon looks acceptable: short-horizon rho={rho_short:.2f}. "
            "The plateau probably comes from elsewhere (validation, features, label aggregation)."
        )

    # Recommend the smallest horizon with rho >= 0.6 vs reference.
    candidate = None
    for r in sorted(rank_rows, key=lambda x: x["horizon"]):
        rho = r.get("spearman_vs_ref")
        if isinstance(rho, float) and not math.isnan(rho) and rho >= 0.60:
            candidate = r["horizon"]
            break
    if candidate is not None:
        lines.append(f"Recommended minimum training horizon: H>={candidate}.")
    else:
        lines.append(
            f"No horizon below H={H_ref} reaches rho>=0.60. Consider H>={H_ref} "
            "or a different oracle (e.g. margin vs V7-baseline plan)."
        )

    bias_short = _pick_short(bias_rows, "passivity_rho")
    if bias_short is not None and bias_short > 0.30:
        lines.append(
            f"Passivity bias is real at the shortest H: rho={bias_short:.2f}. The oracle "
            "systematically prefers plans that spend fewer ships, which trains the model to be "
            "passive. Either lengthen H or score margin relative to a fixed baseline plan."
        )

    flat_short = _pick_short(noise_rows, "frac_flat")
    if flat_short is not None and flat_short > 0.50:
        lines.append(
            f"Label-noise floor is high at the shortest H: {flat_short:.0%} of states have all "
            "candidates within the tie threshold. argmax over near-ties is essentially random."
        )

    return lines


def _print_verdict(report: dict) -> None:
    print("Verdict")
    print("-------")
    for line in _verdict_lines(report):
        print(f"- {line}")


# --------------------------------- main ------------------------------------ #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-states", type=int, default=200)
    p.add_argument("--horizons", type=str, default=",".join(str(h) for h in DEFAULT_HORIZONS),
                   help="comma-separated rollout horizons; the largest is the reference")
    p.add_argument("--opponent", type=str, default=DEFAULT_OPPONENT)
    p.add_argument("--seed-start", type=int, default=20240501)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--tie-eps", type=float, default=0.02,
                   help="value-span below this counts as a 'flat' state for label-noise stat")
    p.add_argument("--out-json", type=str, default=None,
                   help="optional path to dump the full report as JSON")
    p.add_argument("--oracle", choices=("absolute", "margin", "both"), default="both",
                   help="absolute: raw rollout value; margin: value - V7-baseline value; "
                        "both: render both reports from the same data")
    p.add_argument("--quick", action="store_true", help="40 states, single process; smoke run")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.n_states = 40
        args.workers = 1

    horizons = tuple(sorted({int(x) for x in args.horizons.split(",") if x.strip()}))
    if len(horizons) < 2:
        raise SystemExit("need at least two horizons to compare")

    print(f"Harvesting {args.n_states} decision states from bot_v7 vs {args.opponent}...", flush=True)
    t_harvest = time.perf_counter()
    snapshots = harvest_states(
        n_states=args.n_states,
        opponent_name=args.opponent,
        seed_start=args.seed_start,
    )
    print(f"  harvested {len(snapshots)} states in {time.perf_counter() - t_harvest:.1f}s")

    print(f"Scoring {len(snapshots)} states x {len(horizons)} horizons "
          f"(workers={args.workers})...", flush=True)
    work = [(snap, horizons) for snap in snapshots]
    t_score = time.perf_counter()

    if args.workers <= 1:
        results = [_evaluate_state(item) for item in work]
    else:
        # `spawn` is the Windows default; the explicit context keeps behavior
        # identical across platforms and avoids fork-related surprises.
        from multiprocessing import get_context
        ctx = get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            results = pool.map(_evaluate_state, work, chunksize=1)
    elapsed_score = time.perf_counter() - t_score
    print(f"  scored in {elapsed_score:.1f}s "
          f"({len(results) * sum(horizons) / max(elapsed_score, 1e-9):.1f} plan-steps/s aggregate)")

    modes = ("absolute", "margin") if args.oracle == "both" else (args.oracle,)
    reports = {}
    for mode in modes:
        rep = aggregate(results, horizons, tie_eps=args.tie_eps, oracle_mode=mode, baseline_idx=0)
        render_report(rep, tie_eps=args.tie_eps)
        reports[mode] = rep

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        payload = {
            "config": {
                "n_states": args.n_states,
                "horizons": list(horizons),
                "opponent": args.opponent,
                "seed_start": args.seed_start,
                "tie_eps": args.tie_eps,
                "oracle": args.oracle,
            },
            "reports": reports,
            "states": [asdict(r) for r in results],
        }
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nWrote {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
