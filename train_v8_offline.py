#!/usr/bin/env python3
"""Offline training for the Orbit Wars V8 plan ranker.

This version does not label fresh states with the current model. Instead it:
- harvests a fixed dataset from V7 vs fixed opponents,
- labels each snapshot with a fixed rollout teacher,
- trains the ranker only on that frozen dataset.

The goal is to remove the moving-proxy / self-induced distribution shift that
made the online DAgger loop unstable.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import bot_v7
import bot_v8
from SimGame import F_OWNER, F_SHIPS, P_OWNER, P_PROD, P_SHIPS, SimGame, run_match
from v8_core import (
    LinearV8Model,
    TrainingExample,
    build_candidate_plans,
    build_training_example,
)


DEFAULT_OPPONENTS = [
    "notebook_orbitbotnext",
    "notebook_distance_prioritized",
    "notebook_physics_accurate",
    # notebook_tactical_heuristic disabled: NameError on Planet at line 309
]

DEFAULT_SAMPLE_TURNS = (20, 45, 75, 115, 165, 230, 310)

# Plans used by the variance pre-filter. Picked to be qualitatively different
# from each other so that "all four agree" is a strong signal that the snapshot
# is decided.
VARIANCE_PROBE_STYLES = ("v7", "attack", "defense", "hold")
VARIANCE_PROBE_HORIZON = 220  # turns to roll out beyond snapshot before judging


@dataclass
class OfflineSnapshot:
    seed: int
    turn: int
    state_blob: dict
    n_players: int
    our_player: int
    opponent_name: str


def _call_obs_agent(agent: Callable, obs: dict):
    try:
        return agent(obs, None)
    except TypeError:
        pass
    except Exception:
        return []
    try:
        return agent(obs)
    except Exception:
        return []


def _load_opponent(name: str) -> Callable:
    try:
        from opponents import ZOO
        return ZOO[name]
    except Exception:
        def passive(obs, config=None):
            return []

        return passive


def _make_opponent_policy(agent: Callable) -> Callable:
    def policy(game, player: int):
        obs = game.observation(player)
        return _call_obs_agent(agent, obs)

    return policy


def _make_rollout_policy(train_player: int, opponent_policy: Callable) -> Callable:
    """Fixed teacher: V7 continuation on our side, fixed opponent on theirs."""

    def policy(game, player: int):
        obs = game.observation(player)
        if player == train_player:
            try:
                move = bot_v7.agent(obs, None)
            except Exception:
                move = []
            return move if isinstance(move, list) else []
        return opponent_policy(game, player)

    return policy


def _state_to_blob(state) -> dict:
    return {
        "planets": state.planets.tolist(),
        "fleets": state.fleets.tolist(),
        "initial_planets": state.initial_planets.tolist(),
        "angular_velocity": float(state.angular_velocity),
        "step": int(state.step),
        "next_fleet_id": int(state.next_fleet_id),
        "max_steps": int(state.max_steps),
    }


def _blob_to_state(blob: dict):
    from SimGame import FastState

    fleets = np.asarray(blob["fleets"], dtype=np.float32)
    if fleets.size == 0:
        fleets = np.zeros((0, 7), dtype=np.float32)
    return FastState(
        planets=np.asarray(blob["planets"], dtype=np.float32),
        fleets=fleets,
        initial_planets=np.asarray(blob["initial_planets"], dtype=np.float32),
        angular_velocity=float(blob["angular_velocity"]),
        step=int(blob["step"]),
        next_fleet_id=int(blob["next_fleet_id"]),
        max_steps=int(blob["max_steps"]),
    )


def _default_state_path(out_path: str) -> str:
    root, _ = os.path.splitext(out_path)
    return f"{root}_offline_state.pkl"


def _default_dataset_path(out_path: str) -> str:
    root, _ = os.path.splitext(out_path)
    return f"{root}_offline_dataset.pkl"


def _save_pickle(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: str) -> Optional[object]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _harvest_snapshots(
    n_states: int,
    sample_turns: Sequence[int],
    opponents: Sequence[str],
    seed_start: int = 0,
    neutral_pairs: int = 8,
) -> List[OfflineSnapshot]:
    snapshots: List[OfflineSnapshot] = []
    seed = seed_start
    target_turns = sorted(set(int(t) for t in sample_turns))

    while len(snapshots) < n_states:
        opp_name = opponents[(seed - seed_start) % max(1, len(opponents))]
        opponent = _load_opponent(opp_name)
        opp_policy = _make_opponent_policy(opponent)
        game = SimGame.random_game(seed=seed, n_players=2, neutral_pairs=neutral_pairs)
        our_player = seed % game.n_players
        remaining_targets = list(target_turns)
        game_t0 = time.time()
        snaps_before = len(snapshots)
        print(
            f"    harvest game seed={seed} opp={opp_name[:24]:24s} "
            f"({len(snapshots)}/{n_states} snapshots so far)",
            flush=True,
        )

        while not game.is_terminal() and remaining_targets:
            if game.state.step >= remaining_targets[0]:
                snapshots.append(
                    OfflineSnapshot(
                        seed=seed,
                        turn=int(game.state.step),
                        state_blob=_state_to_blob(game.state),
                        n_players=game.n_players,
                        our_player=our_player,
                        opponent_name=opp_name,
                    )
                )
                remaining_targets.pop(0)
                if len(snapshots) >= n_states:
                    break

            actions = {}
            for player in range(game.n_players):
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

        print(
            f"      done seed={seed} (+{len(snapshots)-snaps_before} snapshots, "
            f"{time.time()-game_t0:4.1f}s, end_step={game.state.step})",
            flush=True,
        )
        seed += 1

    return snapshots[:n_states]


def _structural_score(game: SimGame, player: int) -> float:
    """Score the durable conversion state highlighted by the top-1 report."""

    planets = game.state.planets
    fleets = game.state.fleets
    n_players = max(1, game.n_players)

    my_planets = my_prod = my_ships = my_fleet_ships = 0.0
    enemy_planets = enemy_prod = enemy_ships = enemy_fleet_ships = 0.0
    if len(planets):
        my_mask = planets[:, P_OWNER] == player
        enemy_mask = (planets[:, P_OWNER] >= 0) & (planets[:, P_OWNER] != player)
        my_planets = float(np.sum(my_mask))
        enemy_planets = float(np.sum(enemy_mask))
        my_prod = float(np.sum(planets[my_mask, P_PROD])) if np.any(my_mask) else 0.0
        enemy_prod = float(np.sum(planets[enemy_mask, P_PROD])) if np.any(enemy_mask) else 0.0
        my_ships = float(np.sum(planets[my_mask, P_SHIPS])) if np.any(my_mask) else 0.0
        enemy_ships = float(np.sum(planets[enemy_mask, P_SHIPS])) if np.any(enemy_mask) else 0.0
    if len(fleets):
        my_fleet_mask = fleets[:, F_OWNER] == player
        enemy_fleet_mask = (fleets[:, F_OWNER] >= 0) & (fleets[:, F_OWNER] != player)
        my_fleet_ships = float(np.sum(fleets[my_fleet_mask, F_SHIPS])) if np.any(my_fleet_mask) else 0.0
        enemy_fleet_ships = float(np.sum(fleets[enemy_fleet_mask, F_SHIPS])) if np.any(enemy_fleet_mask) else 0.0

    total_planets = max(1.0, my_planets + enemy_planets)
    total_prod = max(1.0, my_prod + enemy_prod)
    total_ships = max(1.0, my_ships + my_fleet_ships + enemy_ships + enemy_fleet_ships)

    planet_dom = (my_planets - enemy_planets / max(1, n_players - 1)) / total_planets
    prod_dom = (my_prod - enemy_prod / max(1, n_players - 1)) / total_prod
    ship_dom = ((my_ships + my_fleet_ships) - (enemy_ships + enemy_fleet_ships) / max(1, n_players - 1)) / total_ships

    # The report's repeated failure mode is missing the conversion threshold:
    # wins build a durable production backbone; losses plateau below it.
    conversion = math.tanh((my_planets - 10.0) / 5.0) if n_players == 2 else math.tanh((my_planets - 14.0) / 6.0)
    prod_lock = math.tanh((my_prod - 20.0) / 12.0) if n_players == 2 else math.tanh((my_prod - 28.0) / 15.0)
    fleet_drag = -0.05 * math.tanh(my_fleet_ships / max(1.0, my_ships + 1.0))
    winner = game.winner()
    win_term = 1.0 if winner == player else (-1.0 if winner >= 0 else 0.0)

    return float(
        0.30 * prod_dom
        + 0.25 * planet_dom
        + 0.15 * ship_dom
        + 0.15 * conversion
        + 0.10 * prod_lock
        + 0.05 * win_term
        + fleet_drag
    )


def _candidate_teacher_value(
    game: SimGame,
    train_player: int,
    our_actions: List[List[float]],
    rollout_policy: Callable,
    horizon: int,
) -> float:
    sim = SimGame(game.state.copy(), n_players=game.n_players)
    if sim.is_terminal():
        return _structural_score(sim, train_player)

    actions_by_player = {}
    for player in range(sim.n_players):
        actions_by_player[player] = our_actions if player == train_player else rollout_policy(sim, player)
    sim.step(actions_by_player)

    for _ in range(max(0, int(horizon) - 1)):
        if sim.is_terminal():
            break
        actions_by_player = {}
        for player in range(sim.n_players):
            actions_by_player[player] = rollout_policy(sim, player)
        sim.step(actions_by_player)
    return _structural_score(sim, train_player)


def _outcome_winner_after(
    game: SimGame,
    train_player: int,
    our_actions: List[List[float]],
    rollout_policy: Callable,
    horizon: int,
) -> int:
    """Run a short rollout and return the winner (-1 if no clear winner)."""
    sim = SimGame(game.state.copy(), n_players=game.n_players)
    if sim.is_terminal():
        return sim.winner()
    actions_by_player = {}
    for player in range(sim.n_players):
        actions_by_player[player] = our_actions if player == train_player else rollout_policy(sim, player)
    sim.step(actions_by_player)
    for _ in range(max(0, int(horizon) - 1)):
        if sim.is_terminal():
            break
        actions_by_player = {p: rollout_policy(sim, p) for p in range(sim.n_players)}
        sim.step(actions_by_player)
    return sim.winner()


def _snapshot_outcome_variance(
    game: SimGame,
    train_player: int,
    rollout_policy: Callable,
    candidate_plans: Sequence,
    horizon: int,
) -> Tuple[bool, int]:
    """Probe if the snapshot's outcome depends on plan choice.

    Returns (is_decisive, distinct_winners). A snapshot is decisive iff the
    short rollout produces at least two different winners across the probe
    plans. We sample VARIANCE_PROBE_STYLES; missing styles fall back to the
    first candidate.
    """
    by_style = {p.name: p for p in candidate_plans}
    fallback = candidate_plans[0]
    winners = set()
    for style in VARIANCE_PROBE_STYLES:
        plan = by_style.get(style, fallback)
        winner = _outcome_winner_after(
            game,
            train_player,
            plan.actions or [],
            rollout_policy,
            horizon=horizon,
        )
        winners.add(int(winner))
    return (len(winners) >= 2, len(winners))


def _label_snapshot(
    snapshot: OfflineSnapshot,
    oracle_horizon: int,
    min_gap: float,
    variance_check: bool = True,
    variance_horizon: int = VARIANCE_PROBE_HORIZON,
) -> Optional[Tuple[TrainingExample, float]]:
    state = _blob_to_state(snapshot.state_blob)
    game = SimGame(state, n_players=snapshot.n_players)
    obs = game.observation(snapshot.our_player)
    world = bot_v7._build_world(obs)
    candidate_plans = build_candidate_plans(world)
    if not candidate_plans:
        raise RuntimeError("no candidate plans generated for offline snapshot")

    opponent = _load_opponent(snapshot.opponent_name)
    opp_policy = _make_opponent_policy(opponent)
    rollout_policy = _make_rollout_policy(snapshot.our_player, opp_policy)

    if variance_check:
        decisive, _n_winners = _snapshot_outcome_variance(
            game,
            snapshot.our_player,
            rollout_policy,
            candidate_plans,
            horizon=int(variance_horizon),
        )
        if not decisive:
            return None

    values = [
        float(
            _candidate_teacher_value(
                game,
                snapshot.our_player,
                plan.actions or [],
                rollout_policy,
                horizon=max(1, int(oracle_horizon)),
            )
        )
        for plan in candidate_plans
    ]
    baseline = float(values[0]) if values else 0.0
    margins = [float(v) - baseline for v in values]
    oracle_idx = int(np.argmax(margins)) if margins else 0
    sorted_margins = sorted(margins, reverse=True)
    gap = float(sorted_margins[0] - sorted_margins[1]) if len(sorted_margins) > 1 else 0.0
    if gap < float(min_gap):
        return None

    example = build_training_example(world, candidate_plans, oracle_idx, float(margins[oracle_idx]))
    return example, gap


def _load_or_build_dataset(
    dataset_path: str,
    refresh: bool,
    n_states: int,
    target_examples: int,
    max_snapshots: int,
    sample_turns: Sequence[int],
    oracle_horizon: int,
    min_gap: float,
    seed_start: int,
    variance_check: bool = True,
    variance_horizon: int = VARIANCE_PROBE_HORIZON,
) -> List[TrainingExample]:
    if not refresh:
        cached = _load_pickle(dataset_path)
        if isinstance(cached, dict) and cached.get("version") == 1 and isinstance(cached.get("examples"), list):
            examples = [ex for ex in cached["examples"] if isinstance(ex, TrainingExample)]
            if examples:
                print(f"Loaded offline dataset from {dataset_path} ({len(examples)} examples)")
                return examples

    examples: List[TrainingExample] = []
    gaps: List[float] = []
    rejected = 0
    harvested_total = 0
    seed_cursor = seed_start
    print("Collecting offline snapshots...", flush=True)
    if variance_check:
        print(
            f"  variance pre-filter: ON (probe={list(VARIANCE_PROBE_STYLES)} horizon={variance_horizon})",
            flush=True,
        )
    while len(examples) < target_examples and harvested_total < max_snapshots:
        batch_target = min(max(8, n_states), max_snapshots - harvested_total)
        snapshots = _harvest_snapshots(
            n_states=batch_target,
            sample_turns=sample_turns,
            opponents=DEFAULT_OPPONENTS,
            seed_start=seed_cursor,
        )
        if not snapshots:
            break
        harvested_total += len(snapshots)
        seed_cursor += max(1, int(math.ceil(len(snapshots) / max(1, len(sample_turns)))))
        batch_t0 = time.time()
        for snap_i, snapshot in enumerate(snapshots):
            snap_t0 = time.time()
            labeled = _label_snapshot(
                snapshot,
                oracle_horizon=oracle_horizon,
                min_gap=min_gap,
                variance_check=variance_check,
                variance_horizon=variance_horizon,
            )
            took = time.time() - snap_t0
            status = "ACC" if labeled is not None else "REJ"
            print(
                f"    snap {snap_i+1:3d}/{len(snapshots):3d} turn={snapshot.turn:3d} "
                f"opp={snapshot.opponent_name[:24]:24s} {status} "
                f"({took:4.1f}s) examples={len(examples):3d}",
                flush=True,
            )
            if labeled is None:
                rejected += 1
                continue
            example, gap = labeled
            gaps.append(gap)
            examples.append(example)
            if len(examples) >= target_examples:
                break
        print(
            f"  harvested={harvested_total:4d} accepted={len(examples):4d} "
            f"rejected={rejected:4d} last_gap={(gaps[-1] if gaps else 0.0):.4f}",
            flush=True,
        )

    payload = {
        "version": 1,
        "config": {
            "n_states": n_states,
            "target_examples": target_examples,
            "max_snapshots": max_snapshots,
            "sample_turns": tuple(int(x) for x in sample_turns),
            "oracle_horizon": int(oracle_horizon),
            "min_gap": float(min_gap),
            "seed_start": int(seed_start),
        },
        "examples": examples,
    }
    _save_pickle(dataset_path, payload)
    print(f"Saved offline dataset to {dataset_path}")

    if gaps:
        print(
            f"  gap stats: mean={float(np.mean(gaps)):.4f} "
            f"median={float(np.median(gaps)):.4f} min={float(np.min(gaps)):.4f}"
        )
    min_required = min(int(target_examples), max(16, int(target_examples) // 4))
    if len(examples) < min_required:
        raise SystemExit(
            f"offline dataset too weak: accepted {len(examples)} examples from {harvested_total} snapshots; "
            f"lower --min-gap or increase --max-snapshots"
        )
    return examples


def _load_opponents_for_benchmark(holdout_opponent: str) -> List[str]:
    bench_opponents = list(DEFAULT_OPPONENTS)
    if holdout_opponent and holdout_opponent not in bench_opponents:
        try:
            _load_opponent(holdout_opponent)
            bench_opponents.append(holdout_opponent)
        except Exception:
            print(f"  (holdout '{holdout_opponent}' unavailable; benchmarking on training set only)")
    return bench_opponents


def _run_benchmark(model: LinearV8Model, games: int, opponents: List[str]) -> Dict[str, tuple[int, int]]:
    if games <= 0:
        return {}
    bot_v8.set_model(model)
    results: Dict[str, tuple[int, int]] = {}
    for opp_name in opponents:
        opp = _load_opponent(opp_name)
        wins = 0
        print(f"  Benchmarking vs {opp_name}...", flush=True)
        for i in range(games):
            if i % 2 == 0:
                result = run_match([bot_v8.agent, opp], seed=1000 + i)
                wins += 1 if result["winner"] == 0 else 0
            else:
                result = run_match([opp, bot_v8.agent], seed=1000 + i)
                wins += 1 if result["winner"] == 1 else 0
        results[opp_name] = (wins, games)
    return results


def _benchmark_stats(results: Dict[str, tuple[int, int]]) -> tuple[float, float, float]:
    if not results:
        return 0.0, 0.0, 0.0
    per_opponent = [w / max(1, n) for w, n in results.values()]
    global_rate = sum(w for w, _ in results.values()) / max(1, sum(n for _, n in results.values()))
    min_rate = min(per_opponent)
    mean_rate = sum(per_opponent) / len(per_opponent)
    return float(global_rate), float(min_rate), float(mean_rate)


def _benchmark_score(results: Dict[str, tuple[int, int]]) -> float:
    per_opponent = [w / max(1, n) for w, n in results.values()]
    if not per_opponent:
        return 0.0
    global_rate = sum(w for w, _ in results.values()) / max(1, sum(n for _, n in results.values()))
    min_rate = min(per_opponent)
    mean_rate = sum(per_opponent) / len(per_opponent)
    return float(0.50 * global_rate + 0.35 * min_rate + 0.15 * mean_rate)


def _sample_buffer(buffer: Sequence[TrainingExample], n: int) -> List[TrainingExample]:
    if not buffer:
        return []
    n = min(n, len(buffer))
    if n <= 0:
        return []
    return random.sample(list(buffer), n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=0.33)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--value-lr", type=float, default=0.02)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dataset-states", type=int, default=64,
                        help="snapshot batch size while mining the offline dataset")
    parser.add_argument("--target-examples", type=int, default=256,
                        help="number of confident labeled examples to mine before training")
    parser.add_argument("--max-snapshots", type=int, default=2400,
                        help="hard cap on inspected snapshots while mining")
    parser.add_argument("--oracle-horizon", type=int, default=80)
    parser.add_argument("--min-gap", type=float, default=0.01)
    parser.add_argument("--no-variance-check", action="store_true",
                        help="disable the outcome-variance pre-filter on snapshots")
    parser.add_argument("--variance-horizon", type=int, default=VARIANCE_PROBE_HORIZON,
                        help="rollout depth (turns) used by the variance pre-filter")
    parser.add_argument("--benchmark-games", type=int, default=12)
    parser.add_argument("--benchmark-seconds", type=int, default=600)
    parser.add_argument("--save-seconds", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-epochs", type=int, default=0,
                        help="optional hard epoch cap; 0 means time-based only")
    parser.add_argument("--val-patience", type=int, default=200,
                        help="epochs without validation improvement before decaying lr")
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=0.003)
    parser.add_argument("--skip-initial-benchmark", action="store_true")
    parser.add_argument("--holdout-opponent", type=str, default="orbit_stars")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--dataset-out", type=str, default=None)
    parser.add_argument("--refresh-dataset", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-state", type=str, default=None)
    parser.add_argument("--state-out", type=str, default=None)
    parser.add_argument("--out", type=str, default=os.path.join("evaluations", "v8_policy.npz"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    dataset_path = args.dataset_out or _default_dataset_path(args.out)
    state_path = args.state_out or _default_state_path(args.out)

    if args.resume and os.path.exists(args.resume):
        model = LinearV8Model.load(args.resume)
        print(f"Resumed from {args.resume}")
    else:
        model = LinearV8Model.zero()
        print("Warm-start: zero weights (offline fixed-teacher mode)")

    bot_v7.set_scorer(None)
    bot_v8.set_model(model)

    print("\nMethod: offline fixed-teacher plan ranking + value head")
    print(
        f"Config: lr={args.lr}  value_lr={args.value_lr}  l2={args.l2}  "
        f"batch={args.batch_size}  target_examples={args.target_examples}  oracle_horizon={args.oracle_horizon}"
    )
    print("Starting training...", flush=True)
    print()

    examples = _load_or_build_dataset(
        dataset_path=dataset_path,
        refresh=args.refresh_dataset,
        n_states=args.dataset_states,
        target_examples=args.target_examples,
        max_snapshots=args.max_snapshots,
        sample_turns=DEFAULT_SAMPLE_TURNS,
        oracle_horizon=args.oracle_horizon,
        min_gap=args.min_gap,
        seed_start=args.seed_start,
        variance_check=not args.no_variance_check,
        variance_horizon=args.variance_horizon,
    )
    if not examples:
        raise SystemExit("offline dataset is empty")

    random.Random(args.seed_start ^ len(examples) ^ 0x5A17).shuffle(examples)
    split = max(1, int(0.8 * len(examples)))
    train_examples = examples[:split]
    val_examples = examples[split:] if split < len(examples) else examples[:]
    print(f"Dataset: train={len(train_examples)}  val={len(val_examples)}")

    bench_opponents = _load_opponents_for_benchmark(args.holdout_opponent)

    deadline = time.time() + args.hours * 3600.0
    best_score = -1.0
    best_eval = (-1.0, -1.0, -1.0)
    best_model = model.copy()
    last_benchmark = time.time()
    last_save = time.time()
    t_start = time.time()
    rng = random.Random(int(time.time() * 1000) % 100000)
    epoch = 0
    active_lr = float(args.lr)
    active_value_lr = float(args.value_lr)
    best_val_loss = float("inf")
    stale_val_epochs = 0

    resume_state_path = args.resume_state or (None if args.refresh_dataset else state_path)
    resumed_state = _load_pickle(resume_state_path) if resume_state_path else None
    if isinstance(resumed_state, dict):
        try:
            epoch = int(resumed_state.get("epoch", 0))
            best_score = float(resumed_state.get("best_score", best_score))
            best_eval = tuple(resumed_state.get("best_eval", best_eval))  # type: ignore[assignment]
            active_lr = float(resumed_state.get("active_lr", active_lr))
            active_value_lr = float(resumed_state.get("active_value_lr", active_value_lr))
            best_val_loss = float(resumed_state.get("best_val_loss", best_val_loss))
            stale_val_epochs = int(resumed_state.get("stale_val_epochs", stale_val_epochs))
            rng_state = resumed_state.get("rng_state")
            if rng_state is not None:
                rng.setstate(rng_state)
            best_path = args.out.replace(".npz", "_best.npz")
            if os.path.exists(best_path):
                best_model = LinearV8Model.load(best_path)
            print(f"Resumed training state from {resume_state_path} (epoch={epoch})")
        except Exception as exc:
            print(f"  (could not load training state: {exc})")

    if not args.skip_initial_benchmark:
        print("[Initial benchmark (zero weights = offline fixed teacher)]", flush=True)
        bench = _run_benchmark(model, args.benchmark_games, bench_opponents)
        for k, (wins, total) in bench.items():
            tag = " (holdout)" if k == args.holdout_opponent else ""
            print(f"  {k}{tag}: {wins}/{total}")
        global_rate, min_rate, mean_rate = _benchmark_stats(bench)
        best_score = _benchmark_score(bench)
        best_eval = (best_score, global_rate, min_rate)
        print(f"  global: {global_rate:.3f}")
        print(f"  min   : {min_rate:.3f}")
        print(f"  mean  : {mean_rate:.3f}")
        print(f"  score : {best_score:.3f}")
        print()
        last_benchmark = time.time()

    while time.time() < deadline and (args.max_epochs <= 0 or epoch < args.max_epochs):
        epoch += 1
        batch = _sample_buffer(train_examples, min(args.batch_size, len(train_examples)))
        stats = model.train_batch(batch, lr=active_lr, value_lr=active_value_lr, l2=args.l2)

        elapsed = time.time() - t_start
        val_stats = {}
        if val_examples:
            val_batch = _sample_buffer(val_examples, min(len(val_examples), max(8, args.batch_size // 2)))
            val_stats = model.train_batch(val_batch, lr=0.0, value_lr=0.0, l2=0.0)
            val_loss = float(val_stats.get("loss", 0.0))
            if val_loss + 1e-4 < best_val_loss:
                best_val_loss = val_loss
                stale_val_epochs = 0
            else:
                stale_val_epochs += 1

            if (
                args.val_patience > 0
                and stale_val_epochs >= args.val_patience
                and active_lr > args.min_lr
            ):
                active_lr = max(float(args.min_lr), active_lr * float(args.lr_decay))
                active_value_lr = max(float(args.min_lr), active_value_lr * float(args.lr_decay))
                stale_val_epochs = 0
                print(
                    f"  Validation stale; decayed lr to {active_lr:.5f} "
                    f"value_lr={active_value_lr:.5f}",
                    flush=True,
                )

        if epoch == 1 or epoch % max(1, args.log_every) == 0:
            print(
                f"[ep {epoch:5d} | {elapsed/60:5.1f}min]  "
                f"loss={stats['loss']:.3f}  rank={stats.get('rank_loss', 0.0):.3f}  "
                f"acc={stats['acc']:.2f}  val={val_stats.get('loss', 0.0):.3f}  "
                f"lr={active_lr:.5f}  |w|={np.linalg.norm(model.score_w):.4f}",
                flush=True,
            )

        now = time.time()
        if now - last_save >= args.save_seconds:
            model.save(args.out)
            print(f"  Saved: {args.out}")
            _save_pickle(
                state_path,
                {
                    "version": 1,
                    "epoch": epoch,
                    "best_score": best_score,
                    "best_eval": best_eval,
                    "rng_state": rng.getstate(),
                    "dataset_path": dataset_path,
                    "dataset_size": len(examples),
                    "active_lr": active_lr,
                    "active_value_lr": active_value_lr,
                    "best_val_loss": best_val_loss,
                    "stale_val_epochs": stale_val_epochs,
                },
            )
            last_save = now

        if now - last_benchmark >= args.benchmark_seconds:
            print(f"  [Benchmark @ ep {epoch}]")
            bench = _run_benchmark(model, args.benchmark_games, bench_opponents)
            for k, (wins, total) in bench.items():
                tag = " (holdout)" if k == args.holdout_opponent else ""
                print(f"    {k}{tag}: {wins}/{total}")
            global_rate, min_rate, mean_rate = _benchmark_stats(bench)
            score = _benchmark_score(bench)
            print(f"    global          : {global_rate:.3f}")
            print(f"    min             : {min_rate:.3f}")
            print(f"    mean            : {mean_rate:.3f}")
            print(f"    score           : {score:.3f}")
            current_eval = (score, global_rate, min_rate)
            if current_eval > best_eval:
                best_score = score
                best_eval = current_eval
                best_model = model.copy()
                best_path = args.out.replace(".npz", "_best.npz")
                best_model.save(best_path)
                print(f"    New best saved  : {best_path}")
            last_benchmark = now

        if now - last_save >= args.save_seconds or epoch == 1:
            _save_pickle(
                state_path,
                {
                    "version": 1,
                    "epoch": epoch,
                    "best_score": best_score,
                    "best_eval": best_eval,
                    "rng_state": rng.getstate(),
                    "dataset_path": dataset_path,
                    "dataset_size": len(examples),
                    "active_lr": active_lr,
                    "active_value_lr": active_value_lr,
                    "best_val_loss": best_val_loss,
                    "stale_val_epochs": stale_val_epochs,
                },
            )

    model.save(args.out)
    best_path = args.out.replace(".npz", "_best.npz")
    if not os.path.exists(best_path):
        best_model.save(best_path)
    _save_pickle(
        state_path,
        {
            "version": 1,
            "epoch": epoch,
            "best_score": best_score,
            "best_eval": best_eval,
            "rng_state": rng.getstate(),
            "dataset_path": dataset_path,
            "dataset_size": len(examples),
            "active_lr": active_lr,
            "active_value_lr": active_value_lr,
            "best_val_loss": best_val_loss,
            "stale_val_epochs": stale_val_epochs,
        },
    )

    elapsed_total = time.time() - t_start
    print(f"\nDone. {epoch} episodes in {elapsed_total/60:.1f}min")
    print(f"Saved final: {args.out}")
    print(f"Saved state: {state_path}")

    final_bench = _run_benchmark(model, args.benchmark_games, bench_opponents)
    final_global, final_min, final_mean = _benchmark_stats(final_bench)
    final_score = _benchmark_score(final_bench)

    best_bench = _run_benchmark(best_model, args.benchmark_games, bench_opponents)
    best_global, best_min, best_mean = _benchmark_stats(best_bench)
    best_score_eval = _benchmark_score(best_bench)

    if (final_score, final_global, final_min) > (best_score_eval, best_global, best_min):
        best_model = model.copy()
        best_score = final_score
        best_eval = (final_score, final_global, final_min)
        best_model.save(best_path)
        best_bench = final_bench
        best_global, best_min, best_mean = final_global, final_min, final_mean
        best_score_eval = final_score
    else:
        best_score = best_score_eval
        best_eval = (best_score_eval, best_global, best_min)
        best_model.save(best_path)

    print(f"Saved best : {best_path}  score={best_score:.3f}")

    print("\n[Final benchmark]")
    for k, (wins, total) in best_bench.items():
        tag = " (holdout)" if k == args.holdout_opponent else ""
        print(f"  {k}{tag}: {wins}/{total}")
    print(f"  global: {best_global:.3f}")
    print(f"  min   : {best_min:.3f}")
    print(f"  mean  : {best_mean:.3f}")
    print(f"  score : {best_score_eval:.3f}")


if __name__ == "__main__":
    main()
