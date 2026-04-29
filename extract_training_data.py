#!/usr/bin/env python3
"""Extract (features, actions) from winning players in replay dataset.

Output: training_data.npz with:
- X: (N, F) float32 — state features per turn
- y: (N, A) float32 — action taken (ships_ratio per planet slot)
- meta: episode mode, winner submission, turn number
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_IN = ROOT / "replay_dataset" / "compact" / "episodes.jsonl.gz"
DEFAULT_OUT = ROOT / "replay_dataset" / "training_data.npz"

N_PLANET_SLOTS = 30  # max planets per episode (pad/truncate)
N_STATE_FEATURES = 22


def safe_div(a: float, b: float) -> float:
    return a / b if b > 1e-9 else 0.0


def extract_state_features(
    turn: int,
    total_turns: int,
    n_players: int,
    ownership: List[int],
    ships: List[float],
    winner_idx: int,
) -> np.ndarray:
    """22 global state features for the winner at this turn."""
    total_planets = len(ownership)
    my_planets = ownership[winner_idx] if winner_idx < len(ownership) else 0
    my_ships = ships[winner_idx] if winner_idx < len(ships) else 0.0
    total_ships = sum(ships) + 1e-9
    total_p = total_planets + 1e-9
    enemy_planets = sum(ownership[i] for i in range(n_players) if i != winner_idx and i < len(ownership))
    enemy_ships = sum(ships[i] for i in range(n_players) if i != winner_idx and i < len(ships))

    return np.array([
        turn / max(1, total_turns),              # phase de jeu [0,1]
        safe_div(my_planets, total_p),           # fraction planètes possédées
        safe_div(enemy_planets, total_p),        # fraction planètes ennemies
        safe_div(my_ships, total_ships),         # fraction ships totaux
        safe_div(enemy_ships, total_ships),      # fraction ships ennemis
        math.log1p(my_ships) / 10.0,            # ships absolus (log)
        math.log1p(enemy_ships) / 10.0,         # ships ennemis absolus (log)
        safe_div(my_planets, max(1, n_players)), # planètes par joueur normalisé
        1.0 if n_players == 2 else 0.0,         # mode 2p
        1.0 if n_players == 4 else 0.0,         # mode 4p
        safe_div(my_ships, max(1, my_planets)),  # ships par planète (force défensive)
        safe_div(enemy_ships, max(1, enemy_planets)),  # ships/planète ennemi
        1.0 if my_planets > enemy_planets else 0.0,    # on domine en planètes
        1.0 if my_ships > enemy_ships else 0.0,        # on domine en ships
        safe_div(my_planets - enemy_planets, total_p), # delta planètes normalisé
        safe_div(my_ships - enemy_ships, total_ships),  # delta ships normalisé
        min(turn / 50.0, 1.0),                  # early game indicator
        1.0 if turn > total_turns * 0.7 else 0.0,  # endgame indicator
        safe_div(n_players - 1, 3),             # nb adversaires normalisé
        0.0, 0.0, 0.0,                          # reserved
    ], dtype=np.float32)


def extract_action_features(
    actions: List[List[Any]],  # actions[player_idx] = list of [from, angle, ships]
    winner_idx: int,
    n_planets: int,
) -> np.ndarray:
    """Per-source-planet ships_ratio sent (normalized). Shape: (N_PLANET_SLOTS,)."""
    out = np.zeros(N_PLANET_SLOTS, dtype=np.float32)
    winner_actions = actions[winner_idx] if winner_idx < len(actions) else []
    if not winner_actions:
        return out
    # Sum ships sent per source planet
    ships_by_src: Dict[int, float] = {}
    for move in winner_actions:
        if isinstance(move, list) and len(move) >= 3:
            src = int(move[0]) if isinstance(move[0], (int, float)) else 0
            n_ships = float(move[2]) if isinstance(move[2], (int, float)) else 0.0
            ships_by_src[src] = ships_by_src.get(src, 0.0) + n_ships
    total_sent = sum(ships_by_src.values()) + 1e-9
    for src, n_ships in ships_by_src.items():
        if src < N_PLANET_SLOTS:
            out[src] = n_ships / total_sent
    return out


def process_episode(ep: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    winner = ep.get("winner")
    if winner is None:
        return None
    n_players = ep.get("n_players", 0)
    if n_players not in (2, 4):
        return None

    actions_by_turn = ep.get("actions", [])
    ownership_by_turn = ep.get("ownership_per_turn", [])
    ships_by_turn = ep.get("ships_per_player_per_turn", [])
    total_turns = ep.get("steps", len(actions_by_turn))

    if not actions_by_turn or len(actions_by_turn) != len(ownership_by_turn):
        return None

    X_rows = []
    y_rows = []

    for t, (turn_actions, own, ships) in enumerate(zip(actions_by_turn, ownership_by_turn, ships_by_turn)):
        # Only extract turns where winner actually acted
        winner_actions = turn_actions[winner] if winner < len(turn_actions) else []
        if not winner_actions:
            continue

        state = extract_state_features(t, total_turns, n_players, own, ships, winner)
        action = extract_action_features(turn_actions, winner, N_PLANET_SLOTS)

        X_rows.append(state)
        y_rows.append(action)

    if not X_rows:
        return None

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_IN)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--min-turns", type=int, default=10)
    p.add_argument("--mode", choices=["2p", "4p", "all"], default="all")
    args = p.parse_args()

    X_all, y_all = [], []
    n_eps = 0
    n_skipped = 0

    with gzip.open(args.input, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                ep = json.loads(line)
            except Exception:
                continue
            if args.mode == "2p" and ep.get("n_players") != 2:
                continue
            if args.mode == "4p" and ep.get("n_players") != 4:
                continue
            result = process_episode(ep)
            if result is None:
                n_skipped += 1
                continue
            X, y = result
            if len(X) < args.min_turns:
                n_skipped += 1
                continue
            X_all.append(X)
            y_all.append(y)
            n_eps += 1

    if not X_all:
        print("No data extracted.", file=sys.stderr)
        return

    X_cat = np.concatenate(X_all, axis=0)
    y_cat = np.concatenate(y_all, axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(args.output), X=X_cat, y=y_cat)

    print(f"Episodes: {n_eps} (skipped {n_skipped})")
    print(f"Samples: {len(X_cat)} turns")
    print(f"X shape: {X_cat.shape}  y shape: {y_cat.shape}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
