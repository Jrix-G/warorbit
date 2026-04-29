#!/usr/bin/env python3
"""Diagnostic: does the V8 oracle correlate with real game outcomes?

For each snapshot we generate the 9 candidate plans and ask two questions:
  1. Oracle score: structural_score after an 80-turn rollout (what training uses).
  2. True outcome: actual win/margin if we continue the full game from there.

If the oracle's argmax plan rarely matches the true argmax plan, no amount of
training on this label will move the bot. The script prints the agreement rate
and the per-snapshot Pearson/Spearman correlation between oracle and truth.
"""

from __future__ import annotations

import math
import random
import time
from typing import Callable, List

import numpy as np

import bot_v7
from SimGame import SimGame, F_OWNER, F_SHIPS, P_OWNER, P_PROD, P_SHIPS
from train_v8_offline import (
    DEFAULT_SAMPLE_TURNS,
    _harvest_snapshots,
    _blob_to_state,
    _load_opponent,
    _make_opponent_policy,
    _make_rollout_policy,
    _structural_score,
)
from v8_core import build_candidate_plans


SAFE_OPPONENTS = [
    "notebook_orbitbotnext",
    "notebook_distance_prioritized",
    "notebook_physics_accurate",
]  # notebook_tactical_heuristic is broken (NameError on Planet)

N_SNAPSHOTS = 18           # 18 snapshots * 9 plans = 162 evaluations
ORACLE_HORIZON = 80        # what the trainer uses
FULL_GAME_CAP = 500        # SimGame default
TIME_BUDGET_S = 20 * 60


def oracle_score(game: SimGame, player: int, our_actions: list, rollout_policy: Callable, horizon: int) -> float:
    sim = SimGame(game.state.copy(), n_players=game.n_players)
    if sim.is_terminal():
        return _structural_score(sim, player)
    actions = {}
    for p in range(sim.n_players):
        actions[p] = our_actions if p == player else rollout_policy(sim, p)
    sim.step(actions)
    for _ in range(max(0, horizon - 1)):
        if sim.is_terminal():
            break
        actions = {p: rollout_policy(sim, p) for p in range(sim.n_players)}
        sim.step(actions)
    return _structural_score(sim, player)


def true_outcome(game: SimGame, player: int, our_actions: list, rollout_policy: Callable) -> tuple:
    """Return (win:int{-1,0,+1}, margin:float) by playing the game to terminal."""
    sim = SimGame(game.state.copy(), n_players=game.n_players)
    if not sim.is_terminal():
        actions = {}
        for p in range(sim.n_players):
            actions[p] = our_actions if p == player else rollout_policy(sim, p)
        sim.step(actions)
    while not sim.is_terminal():
        actions = {p: rollout_policy(sim, p) for p in range(sim.n_players)}
        sim.step(actions)
    scores = sim.scores()
    winner = sim.winner()
    win_term = 1 if winner == player else (-1 if winner >= 0 else 0)
    our_score = float(scores[player])
    best_other = max(float(s) for i, s in enumerate(scores) if i != player) if len(scores) > 1 else 0.0
    margin = our_score - best_other
    return win_term, margin


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    return float(np.corrcoef(ar, br)[0, 1])


def main():
    t0 = time.time()
    print(f"Harvesting {N_SNAPSHOTS} snapshots from V7 vs zoo...", flush=True)
    snapshots = _harvest_snapshots(
        n_states=N_SNAPSHOTS,
        sample_turns=DEFAULT_SAMPLE_TURNS,
        opponents=SAFE_OPPONENTS,
        seed_start=4242,
    )
    print(f"  harvested {len(snapshots)} in {time.time()-t0:.1f}s", flush=True)

    bot_v7.set_scorer(None)

    per_snapshot_oracle_top = []
    per_snapshot_true_top = []
    per_snapshot_pearson = []
    per_snapshot_spearman = []
    style_oracle_picks = {}
    style_true_picks = {}

    for i, snap in enumerate(snapshots):
        if time.time() - t0 > TIME_BUDGET_S:
            print(f"  time budget hit at snapshot {i}", flush=True)
            break
        state = _blob_to_state(snap.state_blob)
        game = SimGame(state, n_players=snap.n_players)
        obs = game.observation(snap.our_player)
        world = bot_v7._build_world(obs)
        plans = build_candidate_plans(world)
        opp = _load_opponent(snap.opponent_name)
        opp_policy = _make_opponent_policy(opp)
        rollout = _make_rollout_policy(snap.our_player, opp_policy)

        oracle_vals = []
        true_wins = []
        true_margins = []
        for plan in plans:
            ov = oracle_score(game, snap.our_player, plan.actions or [], rollout, ORACLE_HORIZON)
            tw, tm = true_outcome(game, snap.our_player, plan.actions or [], rollout)
            oracle_vals.append(ov)
            true_wins.append(tw)
            true_margins.append(tm)

        oracle_arr = np.array(oracle_vals, dtype=np.float64)
        margin_arr = np.array(true_margins, dtype=np.float64)
        win_arr = np.array(true_wins, dtype=np.float64)

        oracle_top = int(np.argmax(oracle_arr))
        # Tie-breaker on truth: prefer wins, then margin
        true_top = int(np.lexsort((margin_arr, win_arr))[-1])

        per_snapshot_oracle_top.append(oracle_top)
        per_snapshot_true_top.append(true_top)

        if oracle_arr.std() > 1e-9 and margin_arr.std() > 1e-9:
            pear = float(np.corrcoef(oracle_arr, margin_arr)[0, 1])
        else:
            pear = 0.0
        per_snapshot_pearson.append(pear)
        per_snapshot_spearman.append(spearman(oracle_arr, margin_arr))

        style_oracle_picks[plans[oracle_top].name] = style_oracle_picks.get(plans[oracle_top].name, 0) + 1
        style_true_picks[plans[true_top].name] = style_true_picks.get(plans[true_top].name, 0) + 1

        print(
            f"  [{i+1:2d}/{len(snapshots)}] turn={snap.turn:3d} opp={snap.opponent_name:30s} "
            f"oracle_top={plans[oracle_top].name:8s} true_top={plans[true_top].name:8s} "
            f"agree={'Y' if oracle_top == true_top else 'N'} "
            f"pearson={pear:+.2f} spearman={per_snapshot_spearman[-1]:+.2f} "
            f"win_spread={int(win_arr.sum())}/9",
            flush=True,
        )

    n = len(per_snapshot_oracle_top)
    if n == 0:
        print("no snapshots evaluated")
        return

    agree = sum(1 for a, b in zip(per_snapshot_oracle_top, per_snapshot_true_top) if a == b)
    print()
    print("=" * 60)
    print(f"Snapshots evaluated      : {n}")
    print(f"Argmax agreement         : {agree}/{n}  ({100.0*agree/n:.1f}%)")
    print(f"  random baseline (1/9)  : {100.0/9:.1f}%")
    print(f"Mean Pearson  (oracle, margin) : {np.mean(per_snapshot_pearson):+.3f}")
    print(f"Mean Spearman (oracle, margin) : {np.mean(per_snapshot_spearman):+.3f}")
    print(f"Total wall time          : {(time.time()-t0)/60:.1f} min")
    print()
    print("Plan picked by ORACLE:")
    for name, cnt in sorted(style_oracle_picks.items(), key=lambda x: -x[1]):
        print(f"  {name:8s}  {cnt}")
    print("Plan picked by TRUTH:")
    for name, cnt in sorted(style_true_picks.items(), key=lambda x: -x[1]):
        print(f"  {name:8s}  {cnt}")
    print()
    print("Reading the result:")
    print("  agreement < 25%  : oracle is essentially noise; training on it cannot help")
    print("  agreement 25-50% : weak signal; a lot of training may extract some gain")
    print("  agreement > 50%  : oracle is usable; bottleneck is elsewhere (capacity, distribution)")
    print("  Pearson < 0.2    : even relative ranking is poor; the proxy disagrees on direction")


if __name__ == "__main__":
    main()
