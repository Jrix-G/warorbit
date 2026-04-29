#!/usr/bin/env python3
"""Analyze compact replay dataset (.jsonl.gz from harvest_replays.py).

Modes:
- summary: counts, win rates per submission, 2p vs 4p split
- diagnose: focus on our submission_id, classify losses
- mining: top players' opening patterns, action density per phase
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def load(path: Path) -> Iterator[Dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    except EOFError:
        pass


def summary(eps: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_mode = Counter(e["n_players"] for e in eps)
    by_sub: Dict[int, Dict[str, int]] = defaultdict(lambda: {"games": 0, "wins": 0, "2p": 0, "4p": 0})
    for e in eps:
        winner = e.get("winner")
        for idx, sid in enumerate(e.get("submission_ids", [])):
            if sid is None:
                continue
            by_sub[sid]["games"] += 1
            by_sub[sid][f'{e["n_players"]}p'] += 1
            if winner == idx:
                by_sub[sid]["wins"] += 1
    top = sorted(by_sub.items(), key=lambda kv: kv[1]["games"], reverse=True)[:30]
    return {
        "episodes": len(eps),
        "by_mode": dict(by_mode),
        "avg_steps": statistics.mean(e["steps"] for e in eps) if eps else 0,
        "top_submissions": [
            {"submission_id": sid, **stats, "win_rate": stats["wins"] / max(1, stats["games"])}
            for sid, stats in top
        ],
    }


def diagnose(eps: List[Dict[str, Any]], my_sub: int) -> Dict[str, Any]:
    losses_by_phase = Counter()
    losses_by_opp_count = Counter()
    avg_planet_curve_loss = []
    avg_planet_curve_win = []
    rapid_collapse = 0
    sun_loss_proxy = 0
    for e in eps:
        sids = e.get("submission_ids", [])
        if my_sub not in sids:
            continue
        my_idx = sids.index(my_sub)
        winner = e.get("winner")
        own = e.get("ownership_per_turn", [])
        if not own:
            continue
        my_curve = [t[my_idx] if my_idx < len(t) else 0 for t in own]
        peak = max(my_curve) if my_curve else 0
        final = my_curve[-1] if my_curve else 0
        if winner == my_idx:
            avg_planet_curve_win.append(my_curve)
        else:
            avg_planet_curve_loss.append(my_curve)
            if peak >= 2 and final == 0:
                rapid_collapse += 1
            half = len(my_curve) // 2
            phase = "early" if my_curve[half] == 0 and peak <= 1 else (
                "mid_collapse" if my_curve[half] > 0 and final == 0 else "endgame"
            )
            losses_by_phase[phase] += 1
            losses_by_opp_count[e["n_players"]] += 1

        ships = e.get("ships_per_player_per_turn", [])
        if ships and my_idx < len(ships[0]):
            for i in range(1, len(ships)):
                if i < len(ships) and my_idx < len(ships[i]):
                    drop = ships[i - 1][my_idx] - ships[i][my_idx]
                    if drop > 50:
                        sun_loss_proxy += 1
                        break

    def avg_curve(curves: List[List[int]]) -> List[float]:
        if not curves:
            return []
        L = max(len(c) for c in curves)
        out = [0.0] * L
        cnt = [0] * L
        for c in curves:
            for i, v in enumerate(c):
                out[i] += v
                cnt[i] += 1
        return [out[i] / cnt[i] if cnt[i] else 0 for i in range(L)]

    return {
        "submission_id": my_sub,
        "wins": len(avg_planet_curve_win),
        "losses": len(avg_planet_curve_loss),
        "win_rate": len(avg_planet_curve_win) / max(1, len(avg_planet_curve_win) + len(avg_planet_curve_loss)),
        "losses_by_phase": dict(losses_by_phase),
        "losses_by_mode": dict(losses_by_opp_count),
        "rapid_collapse_count": rapid_collapse,
        "big_ship_drops": sun_loss_proxy,
        "win_curve_sample": avg_curve(avg_planet_curve_win)[::20],
        "loss_curve_sample": avg_curve(avg_planet_curve_loss)[::20],
    }


def mining(eps: List[Dict[str, Any]], top_n: int = 10) -> Dict[str, Any]:
    by_sub_actions: Dict[int, List[int]] = defaultdict(list)
    by_sub_first_attack: Dict[int, List[int]] = defaultdict(list)
    by_sub_wins: Dict[int, int] = defaultdict(int)
    by_sub_games: Dict[int, int] = defaultdict(int)
    for e in eps:
        sids = e.get("submission_ids", [])
        actions = e.get("actions", [])
        winner = e.get("winner")
        for idx, sid in enumerate(sids):
            if sid is None:
                continue
            by_sub_games[sid] += 1
            if winner == idx:
                by_sub_wins[sid] += 1
            total = sum(len(turn[idx]) if idx < len(turn) else 0 for turn in actions)
            by_sub_actions[sid].append(total)
            for t, turn in enumerate(actions):
                if idx < len(turn) and len(turn[idx]) > 0:
                    by_sub_first_attack[sid].append(t)
                    break
    leaderboard = sorted(
        ((sid, by_sub_wins[sid] / max(1, by_sub_games[sid]), by_sub_games[sid]) for sid in by_sub_games if by_sub_games[sid] >= 3),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]
    return {
        "top_winrate": [
            {
                "submission_id": sid,
                "win_rate": wr,
                "games": games,
                "avg_actions": statistics.mean(by_sub_actions[sid]),
                "avg_first_attack_turn": statistics.mean(by_sub_first_attack[sid]) if by_sub_first_attack[sid] else None,
            }
            for sid, wr, games in leaderboard
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("replay_dataset/compact/episodes.jsonl.gz"))
    p.add_argument("--mode", choices=["summary", "diagnose", "mining", "all"], default="all")
    p.add_argument("--my-sub", type=int, default=52128366)
    args = p.parse_args()

    eps = list(load(args.input))
    out: Dict[str, Any] = {}
    if args.mode in ("summary", "all"):
        out["summary"] = summary(eps)
    if args.mode in ("diagnose", "all"):
        out["diagnose"] = diagnose(eps, args.my_sub)
    if args.mode in ("mining", "all"):
        out["mining"] = mining(eps)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
