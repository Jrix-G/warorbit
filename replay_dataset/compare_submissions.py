#!/usr/bin/env python3
"""Compare groups of Orbit Wars submissions using local replay summaries.

Typical use:
  python3 replay_dataset/compare_submissions.py --ours 52128366 52018000 --enemies 52105316 51994568

If a submission has no local summary yet, the script can optionally collect it
through the Kaggle browser session used by `collect_and_analyze.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from replay_dataset.collect_and_analyze import aggregate, collect_for_submission, load_json, summarize_replay


ROOT = Path(__file__).resolve().parent / "output"


def _load_report(submission_id: int) -> Dict[str, Any] | None:
    sub_dir = ROOT / f"submission_{submission_id}"
    summary = sub_dir / "summary.json"
    if summary.exists():
        report = load_json(summary)
        episodes = report.get("episodes", [])
        if episodes and isinstance(episodes[0], dict) and "our_action_count" in episodes[0]:
            return report

        episodes_meta_path = sub_dir / "episodes.json"
        if not episodes_meta_path.exists():
            return report

        episodes_meta = load_json(episodes_meta_path)
        episodes_meta_by_id = {
            int(ep["id"]): ep
            for ep in episodes_meta
            if isinstance(ep, dict) and "id" in ep
        }
        rebuilt = {
            "submission_id": submission_id,
            "episodes": [],
        }
        for item in episodes:
            if not isinstance(item, dict):
                continue
            episode_id = int(item.get("episode_id", -1))
            replay_path = sub_dir / f"episode_{episode_id}.json"
            episode_meta = episodes_meta_by_id.get(episode_id)
            if episode_id < 0 or episode_meta is None or not replay_path.exists():
                rebuilt["episodes"].append(item)
                continue
            replay = load_json(replay_path)
            rebuilt["episodes"].append(summarize_replay(replay, episode_meta, [], submission_id).__dict__)
        return rebuilt
    return None


def _ensure_report(submission_id: int, profile: Path, headless: bool, collect_missing: bool) -> Dict[str, Any] | None:
    report = _load_report(submission_id)
    if report is not None:
        return report
    if not collect_missing:
        return None
    return collect_for_submission(submission_id, profile, ROOT, headless)


def _summarize_group(name: str, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg = {
        "name": name,
        "submission_count": len(reports),
        "episodes": 0,
        "ok_episodes": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "avg_steps": 0.0,
        "planet_captures": 0,
        "our_ships_sent": 0,
        "enemy_ships_sent": 0,
        "our_actions": 0,
        "enemy_actions": 0,
        "mean_our_owned_planets": 0.0,
        "mean_enemy_owned_planets": 0.0,
    }
    if not reports:
        return agg

    totals = [aggregate(report) for report in reports]
    agg["episodes"] = sum(item["episodes"] for item in totals)
    agg["ok_episodes"] = sum(item["ok_episodes"] for item in totals)
    agg["wins"] = sum(item["wins"] for item in totals)
    agg["losses"] = sum(item["losses"] for item in totals)
    agg["planet_captures"] = sum(item["planet_captures"] for item in totals)
    agg["our_ships_sent"] = sum(item["our_ships_sent"] for item in totals)
    agg["enemy_ships_sent"] = sum(item["enemy_ships_sent"] for item in totals)
    agg["our_actions"] = sum(item["our_actions"] for item in totals)
    agg["enemy_actions"] = sum(item["enemy_actions"] for item in totals)
    agg["mean_our_owned_planets"] = sum(item["mean_our_owned_planets"] for item in totals) / max(1, len(totals))
    agg["mean_enemy_owned_planets"] = sum(item["mean_enemy_owned_planets"] for item in totals) / max(1, len(totals))
    agg["avg_steps"] = sum(item["avg_steps"] * max(1, item["ok_episodes"]) for item in totals) / max(
        1, sum(item["ok_episodes"] for item in totals)
    )
    agg["win_rate"] = agg["wins"] / max(1, agg["ok_episodes"])
    agg["our_ships_per_action"] = agg["our_ships_sent"] / max(1, agg["our_actions"])
    agg["enemy_ships_per_action"] = agg["enemy_ships_sent"] / max(1, agg["enemy_actions"])
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple Orbit Wars submissions.")
    parser.add_argument("--ours", nargs="*", type=int, default=[], help="submission ids for your bots")
    parser.add_argument("--enemies", nargs="*", type=int, default=[], help="submission ids for enemy bots")
    parser.add_argument("--profile", type=Path, default=Path("replay_observer/output/kaggle_profile"))
    parser.add_argument("--collect-missing", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "comparison.json")
    args = parser.parse_args()

    ours_reports = []
    enemy_reports = []

    for sid in args.ours:
        report = _ensure_report(sid, args.profile, args.headless, args.collect_missing)
        if report is not None:
            ours_reports.append(report)

    for sid in args.enemies:
        report = _ensure_report(sid, args.profile, args.headless, args.collect_missing)
        if report is not None:
            enemy_reports.append(report)

    ours = _summarize_group("ours", ours_reports)
    enemies = _summarize_group("enemies", enemy_reports)

    comparison = {
        "ours": ours,
        "enemies": enemies,
        "delta": {
            "win_rate": ours["win_rate"] - enemies["win_rate"],
            "avg_steps": ours["avg_steps"] - enemies["avg_steps"],
            "our_ships_per_action": ours["our_ships_per_action"] - enemies["our_ships_per_action"],
            "mean_our_owned_planets": ours["mean_our_owned_planets"] - enemies["mean_our_owned_planets"],
        },
        "source": {
            "ours": args.ours,
            "enemies": args.enemies,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
