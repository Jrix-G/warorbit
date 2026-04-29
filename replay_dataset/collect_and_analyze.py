#!/usr/bin/env python3
"""Collect Orbit Wars replays for a submission and analyze them.

This script uses an authenticated Kaggle browser session to:
1. query all episodes for a given submissionId
2. fetch the full episode replay JSON for each episode
3. extract a compact summary for quick strategy analysis
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "output"
DEFAULT_PROFILE = Path("replay_observer/output/kaggle_profile")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class EpisodeSummary:
    episode_id: int
    submission_id: int
    reward: Optional[int]
    steps: Optional[int]
    duration_s: Optional[float]
    winner_submission_id: Optional[int]
    team_names: List[str]
    our_team_name: Optional[str]
    our_index: Optional[int]
    our_reward: Optional[int]
    our_final_ships: Optional[float]
    our_final_planets: Optional[int]
    our_planet_captures: int
    our_fleet_count: int
    enemy_fleet_count: int
    sun_loss_fleets: int
    offmap_fleets: int
    our_action_count: int = 0
    enemy_action_count: int = 0
    our_ships_sent: int = 0
    enemy_ships_sent: int = 0
    our_mean_owned_planets: float = 0.0
    enemy_mean_owned_planets: float = 0.0
    our_peak_owned_planets: int = 0
    enemy_peak_owned_planets: int = 0
    our_first_gain_turn: Optional[int] = None
    enemy_first_gain_turn: Optional[int] = None


async def fetch_json(page, url: str, body: Dict[str, Any]) -> Any:
    return await page.evaluate(
        """async ({url, body}) => {
            const r = await fetch(url, {
              method: 'POST',
              headers: {'content-type': 'application/json'},
              body: JSON.stringify(body),
              credentials: 'include',
            });
            return {status: r.status, text: await r.text()};
        }""",
        {"url": url, "body": body},
    )


def _get(obs: Dict[str, Any], *keys, default=None):
    cur: Any = obs
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def summarize_replay(replay: Dict[str, Any], episode: Dict[str, Any], teams: List[Dict[str, Any]], submission_id: int) -> EpisodeSummary:
    agents = episode.get("agents", [])
    team_names = [t.get("teamName") for t in teams]
    our_idx = None
    our_reward = None
    our_team_name = None
    winner_submission_id = None
    reward_by_submission = {}
    for idx, agent in enumerate(agents):
        sid = agent.get("submissionId")
        reward_by_submission[sid] = agent.get("reward")
        if sid == submission_id:
            our_idx = idx
            our_reward = agent.get("reward")
    for team in teams:
        if team.get("publicLeaderboardSubmissionId") == submission_id:
            our_team_name = team.get("teamName")
            break
    for agent in agents:
        if agent.get("reward") == 1:
            winner_submission_id = agent.get("submissionId")
            break

    our_final_ships = None
    our_final_planets = None
    our_fleet_count = 0
    enemy_fleet_count = 0
    sun_loss_fleets = 0
    offmap_fleets = 0
    our_planet_captures = 0
    steps = None
    duration_s = None
    our_action_count = 0
    enemy_action_count = 0
    our_ships_sent = 0
    enemy_ships_sent = 0
    our_owned_series: List[int] = []
    enemy_owned_series: List[int] = []
    our_first_gain_turn = None
    enemy_first_gain_turn = None
    prev_our_owned = None
    prev_enemy_owned = None

    steps_list = replay.get("steps") or []
    if steps_list:
        steps = len(steps_list)

        # Find the player's last visible state.
        final = steps_list[-1]
        player_step = None
        if isinstance(final, list) and our_idx is not None and our_idx < len(final):
            player_step = final[our_idx]
        elif isinstance(final, dict):
            player_step = final
        obs = (player_step or {}).get("observation", {})
        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])
        players = obs.get("players", [])

        if our_idx is not None and isinstance(players, list) and our_idx < len(players):
            player_entry = players[our_idx]
            if isinstance(player_entry, dict):
                our_final_ships = player_entry.get("ships")
        if our_idx is not None:
            def owner_of(item):
                if isinstance(item, dict):
                    return item.get("owner")
                if isinstance(item, list) and len(item) > 1:
                    return item[1]
                return None

            our_final_planets = sum(1 for p in planets if owner_of(p) == our_idx)
            our_fleet_count = sum(1 for f in fleets if owner_of(f) == our_idx)
            enemy_fleet_count = sum(1 for f in fleets if owner_of(f) is not None and owner_of(f) != our_idx)
            sun_loss_fleets = 0
            offmap_fleets = 0

        # Planet capture events are embedded in the step logs.
        for turn_idx, step in enumerate(steps_list):
            if not isinstance(step, list):
                continue
            if our_idx is None or our_idx >= len(step):
                continue
            obs_i = step[our_idx].get("observation", {})
            our_action = step[our_idx].get("action") or []
            if isinstance(our_action, list):
                our_action_count += len(our_action)
                our_ships_sent += sum(int(move[2]) for move in our_action if isinstance(move, list) and len(move) >= 3 and isinstance(move[2], (int, float)))
            for idx, agent in enumerate(step):
                if not isinstance(agent, dict):
                    continue
                action = agent.get("action") or []
                if not isinstance(action, list):
                    continue
                ships_sent = sum(int(move[2]) for move in action if isinstance(move, list) and len(move) >= 3 and isinstance(move[2], (int, float)))
                if idx == our_idx:
                    continue
                enemy_action_count += len(action)
                enemy_ships_sent += ships_sent

            # Count our current planets on the final frame only; this is a proxy.
            # The exact capture timeline is derived elsewhere if needed.
            if step is final:
                planets_i = obs_i.get("planets", [])
                our_planet_captures = sum(
                    1 for p in planets_i
                    if (p[1] if isinstance(p, list) and len(p) > 1 else p.get("owner") if isinstance(p, dict) else None) == our_idx
                )

            if isinstance(step, list) and our_idx < len(step):
                planets_now = obs_i.get("planets", [])
                our_owned = sum(
                    1
                    for p in planets_now
                    if (p[1] if isinstance(p, list) and len(p) > 1 else p.get("owner") if isinstance(p, dict) else None) == our_idx
                )
                enemy_owned = sum(
                    1
                    for p in planets_now
                    if (p[1] if isinstance(p, list) and len(p) > 1 else p.get("owner") if isinstance(p, dict) else None) not in (-1, our_idx)
                )
                our_owned_series.append(our_owned)
                enemy_owned_series.append(enemy_owned)
                if prev_our_owned is not None and our_first_gain_turn is None and our_owned > prev_our_owned:
                    our_first_gain_turn = turn_idx
                if prev_enemy_owned is not None and enemy_first_gain_turn is None and enemy_owned > prev_enemy_owned:
                    enemy_first_gain_turn = turn_idx
                prev_our_owned = our_owned
                prev_enemy_owned = enemy_owned

        configuration = replay.get("configuration", {})
        if steps > 1:
            agent_timeout = configuration.get("agentTimeout")
            if agent_timeout is not None:
                duration_s = steps * float(agent_timeout)

    return EpisodeSummary(
        episode_id=episode["id"],
        submission_id=submission_id,
        reward=our_reward,
        steps=steps,
        duration_s=duration_s,
        winner_submission_id=winner_submission_id,
        team_names=team_names,
        our_team_name=our_team_name,
        our_index=our_idx,
        our_reward=our_reward,
        our_final_ships=our_final_ships,
        our_final_planets=our_final_planets,
        our_planet_captures=our_planet_captures,
        our_fleet_count=our_fleet_count,
        enemy_fleet_count=enemy_fleet_count,
        sun_loss_fleets=sun_loss_fleets,
        offmap_fleets=offmap_fleets,
        our_action_count=our_action_count,
        enemy_action_count=enemy_action_count,
        our_ships_sent=our_ships_sent,
        enemy_ships_sent=enemy_ships_sent,
        our_mean_owned_planets=(sum(our_owned_series) / max(1, len(our_owned_series))),
        enemy_mean_owned_planets=(sum(enemy_owned_series) / max(1, len(enemy_owned_series))),
        our_peak_owned_planets=max(our_owned_series) if our_owned_series else 0,
        enemy_peak_owned_planets=max(enemy_owned_series) if enemy_owned_series else 0,
        our_first_gain_turn=our_first_gain_turn,
        enemy_first_gain_turn=enemy_first_gain_turn,
    )


async def collect_for_submission(submission_id: int, profile: Path, out_dir: Path, headless: bool) -> Dict[str, Any]:
    from playwright.async_api import async_playwright

    episodes_url = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
    replay_url = "https://www.kaggle.com/api/i/competitions.EpisodeService/GetEpisodeReplay"

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            str(profile),
            headless=headless,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto("https://www.kaggle.com/competitions/orbit-wars/submissions", wait_until="domcontentloaded")

        res = await fetch_json(page, episodes_url, {"submissionId": submission_id})
        if res["status"] != 200:
            raise RuntimeError(f"ListEpisodes failed: {res['status']} {res['text'][:500]}")
        episodes = json.loads(res["text"]).get("episodes", [])
        out = {
            "submission_id": submission_id,
            "episode_count": len(episodes),
            "episodes": [],
        }

        sub_dir = out_dir / f"submission_{submission_id}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        write_json(sub_dir / "episodes.json", episodes)

        for ep in episodes:
            episode_id = ep["id"]
            r = await fetch_json(page, replay_url, {"episodeId": episode_id})
            if r["status"] != 200:
                out["episodes"].append({"episode_id": episode_id, "error": r["text"][:500]})
                continue
            replay = json.loads(r["text"])
            write_json(sub_dir / f"episode_{episode_id}.json", replay)
            summary = summarize_replay(replay, ep, [], submission_id)
            out["episodes"].append(asdict(summary))

        await ctx.close()
        write_json(sub_dir / "summary.json", out)
        return out


def aggregate(report: Dict[str, Any]) -> Dict[str, Any]:
    episodes = report["episodes"]
    ok = [e for e in episodes if "error" not in e]
    wins = sum(1 for e in ok if e.get("our_reward") == 1)
    losses = sum(1 for e in ok if e.get("our_reward") == -1)
    captures = sum(e.get("our_planet_captures", 0) for e in ok)
    offmap = sum(e.get("offmap_fleets", 0) for e in ok)
    sun = sum(e.get("sun_loss_fleets", 0) for e in ok)
    our_ships_sent = sum(e.get("our_ships_sent", 0) for e in ok)
    enemy_ships_sent = sum(e.get("enemy_ships_sent", 0) for e in ok)
    our_actions = sum(e.get("our_action_count", 0) for e in ok)
    enemy_actions = sum(e.get("enemy_action_count", 0) for e in ok)
    counted = [e for e in ok if e.get("steps")]
    avg_steps = sum(e["steps"] for e in counted) / max(1, len(counted))
    return {
        "submission_id": report["submission_id"],
        "episodes": len(episodes),
        "ok_episodes": len(ok),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / max(1, len(ok)),
        "avg_steps": avg_steps,
        "planet_captures": captures,
        "offmap_fleets": offmap,
        "sun_loss_fleets": sun,
        "our_ships_sent": our_ships_sent,
        "enemy_ships_sent": enemy_ships_sent,
        "our_actions": our_actions,
        "enemy_actions": enemy_actions,
        "our_ships_per_action": our_ships_sent / max(1, our_actions),
        "enemy_ships_per_action": enemy_ships_sent / max(1, enemy_actions),
        "mean_our_owned_planets": sum(e.get("our_mean_owned_planets", 0.0) for e in ok) / max(1, len(ok)),
        "mean_enemy_owned_planets": sum(e.get("enemy_mean_owned_planets", 0.0) for e in ok) / max(1, len(ok)),
        "mean_our_peak_owned_planets": sum(e.get("our_peak_owned_planets", 0) for e in ok) / max(1, len(ok)),
        "mean_enemy_peak_owned_planets": sum(e.get("enemy_peak_owned_planets", 0) for e in ok) / max(1, len(ok)),
    }


def aggregate_many(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    combined = {
        "submission_ids": [r["submission_id"] for r in reports],
        "episodes": 0,
        "ok_episodes": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "avg_steps": 0.0,
        "planet_captures": 0,
        "offmap_fleets": 0,
        "sun_loss_fleets": 0,
        "our_ships_sent": 0,
        "enemy_ships_sent": 0,
        "our_actions": 0,
        "enemy_actions": 0,
        "by_submission": [],
    }
    if not reports:
        return combined

    total_steps = 0.0
    total_ok = 0
    for report in reports:
        agg = aggregate(report)
        combined["episodes"] += agg["episodes"]
        combined["ok_episodes"] += agg["ok_episodes"]
        combined["wins"] += agg["wins"]
        combined["losses"] += agg["losses"]
        combined["planet_captures"] += agg["planet_captures"]
        combined["offmap_fleets"] += agg["offmap_fleets"]
        combined["sun_loss_fleets"] += agg["sun_loss_fleets"]
        combined["our_ships_sent"] += agg["our_ships_sent"]
        combined["enemy_ships_sent"] += agg["enemy_ships_sent"]
        combined["our_actions"] += agg["our_actions"]
        combined["enemy_actions"] += agg["enemy_actions"]
        combined["by_submission"].append(agg)
        if agg["ok_episodes"] > 0:
            total_steps += agg["avg_steps"] * agg["ok_episodes"]
            total_ok += agg["ok_episodes"]

    combined["win_rate"] = combined["wins"] / max(1, combined["ok_episodes"])
    combined["avg_steps"] = total_steps / max(1, total_ok)
    combined["our_ships_per_action"] = combined["our_ships_sent"] / max(1, combined["our_actions"])
    combined["enemy_ships_per_action"] = combined["enemy_ships_sent"] / max(1, combined["enemy_actions"])
    return combined


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--submission-id", type=int, nargs="+", required=True)
    p.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--headless", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    reports = [
        asyncio.run(collect_for_submission(submission_id, args.profile, args.output, args.headless))
        for submission_id in args.submission_id
    ]
    agg = aggregate_many(reports)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
