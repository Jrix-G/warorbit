#!/usr/bin/env python3
"""Harvest Orbit Wars replays from Kaggle leaderboard at scale.

Pipeline:
1. Scrape leaderboard top N teams via LeaderboardService.
2. For each team, list submissions, pick highest-scoring active.
3. ListEpisodes per submission, GetEpisodeReplay per episode.
4. Extract compact features (initial obs + per-turn actions + outcome).
5. Append to gzipped JSONL. Drop raw replay.

Target ratio: 70% 4-player / 30% 2-player.
Output budget: <90MB total (GitHub-friendly).
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE = ROOT / "replay_observer" / "output" / "kaggle_profile"
DEFAULT_OUT = ROOT / "replay_dataset" / "compact"

LEADERBOARD_URL = "https://www.kaggle.com/api/i/competitions.LeaderboardService/GetLeaderboard"
SUBMISSIONS_URL = "https://www.kaggle.com/api/i/competitions.SubmissionService/ListEpisodeSubmissions"
EPISODES_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
REPLAY_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/GetEpisodeReplay"
COMP_SLUG = "orbit-wars"


def _owner(item) -> Optional[int]:
    if isinstance(item, dict):
        return item.get("owner")
    if isinstance(item, list) and len(item) > 1:
        return item[1]
    return None


def extract_compact(replay: Dict[str, Any], episode_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract compact representation. Returns None if malformed."""
    steps = replay.get("steps") or []
    if not steps or not isinstance(steps[0], list):
        return None

    n_players = len(steps[0])
    if n_players not in (2, 4):
        return None

    config = replay.get("configuration", {}) or {}
    info = replay.get("info", {}) or {}
    agent_names = [a.get("Name") for a in info.get("Agents", [])]

    first_step = steps[0]
    first_obs = (first_step[0] or {}).get("observation", {}) if first_step else {}

    initial = {
        "planets": first_obs.get("planets", []),
        "angular_velocity": first_obs.get("angular_velocity"),
        "initial_planets": first_obs.get("initial_planets"),
        "comet_planet_ids": first_obs.get("comet_planet_ids"),
        "comets": first_obs.get("comets"),
    }

    actions: List[List[Any]] = []
    ownership: List[List[int]] = []
    ships_per_player: List[List[float]] = []
    for step in steps:
        if not isinstance(step, list):
            continue
        turn_actions = []
        for agent in step:
            if isinstance(agent, dict):
                act = agent.get("action") or []
                turn_actions.append(act if isinstance(act, list) else [])
            else:
                turn_actions.append([])
        actions.append(turn_actions)

        obs0 = (step[0] or {}).get("observation", {}) if step else {}
        planets = obs0.get("planets", [])
        own_counts = [0] * n_players
        ships_counts = [0.0] * n_players
        for p in planets:
            o = _owner(p)
            if o is not None and 0 <= o < n_players:
                own_counts[o] += 1
                ships_counts[o] += float(p[5] if isinstance(p, list) and len(p) > 5 else (p.get("ships", 0) if isinstance(p, dict) else 0))
        for f in obs0.get("fleets", []):
            o = _owner(f)
            if o is not None and 0 <= o < n_players:
                ships_counts[o] += float(f[6] if isinstance(f, list) and len(f) > 6 else (f.get("ships", 0) if isinstance(f, dict) else 0))
        ownership.append(own_counts)
        ships_per_player.append([round(s, 1) for s in ships_counts])

    rewards = []
    sids = []
    for agent in episode_meta.get("agents", []):
        rewards.append(agent.get("reward"))
        sids.append(agent.get("submissionId"))

    winner = None
    for i, r in enumerate(rewards):
        if r == 1:
            winner = i
            break

    return {
        "episode_id": episode_meta.get("id"),
        "n_players": n_players,
        "agent_names": agent_names,
        "submission_ids": sids,
        "rewards": rewards,
        "winner": winner,
        "steps": len(steps),
        "config": {k: config.get(k) for k in ("shipSpeed", "cometSpeed", "episodeSteps", "agentTimeout")},
        "initial": initial,
        "actions": actions,
        "ownership_per_turn": ownership,
        "ships_per_player_per_turn": ships_per_player,
    }


async def fetch_json(page, url: str, body: Dict[str, Any]) -> Dict[str, Any]:
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


async def get_top_teams(page, n: int) -> List[Dict[str, Any]]:
    res = await fetch_json(page, LEADERBOARD_URL, {"competitionSlug": COMP_SLUG})
    if res["status"] != 200:
        raise RuntimeError(f"Leaderboard {res['status']}: {res['text'][:300]}")
    data = json.loads(res["text"])
    teams = data.get("teams") or data.get("leaderboard") or []
    return teams[:n]


async def list_episodes(page, submission_id: int) -> List[Dict[str, Any]]:
    res = await fetch_json(page, EPISODES_URL, {"submissionId": submission_id})
    if res["status"] != 200:
        return []
    return json.loads(res["text"]).get("episodes", [])


async def get_replay(page, episode_id: int) -> Optional[Dict[str, Any]]:
    res = await fetch_json(page, REPLAY_URL, {"episodeId": episode_id})
    if res["status"] != 200:
        return None
    try:
        return json.loads(res["text"])
    except Exception:
        return None


def open_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return gzip.open(path, "at", encoding="utf-8")


def load_seen(path: Path) -> set:
    seen = set()
    if not path.exists():
        return seen
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                ep = json.loads(line)
                seen.add(ep.get("episode_id"))
            except Exception:
                continue
    return seen


async def harvest(args) -> None:
    from playwright.async_api import async_playwright

    out_path = args.output
    seen = load_seen(out_path)
    print(f"[seen] {len(seen)} episodes already in {out_path}", file=sys.stderr)

    target_4p = int(args.target * 0.7)
    target_2p = args.target - target_4p
    counts = {2: 0, 4: 0}
    bytes_written = out_path.stat().st_size if out_path.exists() else 0
    budget = args.max_bytes

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            str(args.profile),
            headless=args.headless,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto(f"https://www.kaggle.com/competitions/{COMP_SLUG}/submissions", wait_until="domcontentloaded")
        await asyncio.sleep(2)

        # BFS: start from seed submissions, discover opponents from each ListEpisodes response.
        sub_queue: List[int] = list(args.extra_submissions or [])
        try:
            teams = await get_top_teams(page, args.top_teams)
            for t in teams:
                sid = t.get("publicLeaderboardSubmissionId") or t.get("submissionId")
                if sid:
                    sub_queue.append(int(sid))
        except Exception as e:
            print(f"[lb] skipped ({e})", file=sys.stderr)
        sub_queue = list(dict.fromkeys(sub_queue))
        print(f"[seed] {len(sub_queue)} seed submissions", file=sys.stderr)
        visited_subs: set = set()

        writer = open_writer(out_path)
        try:
            while sub_queue:
                sid = sub_queue.pop(0)
                if sid in visited_subs:
                    continue
                visited_subs.add(sid)
                if counts[2] >= target_2p and counts[4] >= target_4p:
                    break
                if bytes_written >= budget:
                    print(f"[budget] reached {bytes_written/1e6:.1f}MB cap", file=sys.stderr)
                    break
                eps = await list_episodes(page, sid)
                print(f"[sub {sid}] {len(eps)} episodes (queue={len(sub_queue)})", file=sys.stderr)
                for ep in eps:
                    for ag in ep.get("agents", []):
                        opp_sid = ag.get("submissionId")
                        if opp_sid and opp_sid not in visited_subs and opp_sid not in sub_queue:
                            sub_queue.append(int(opp_sid))
                for ep in eps:
                    eid = ep.get("id")
                    if eid in seen:
                        continue
                    if bytes_written >= budget:
                        break
                    replay = await get_replay(page, eid)
                    if not replay:
                        continue
                    compact = extract_compact(replay, ep)
                    if not compact:
                        continue
                    np = compact["n_players"]
                    if np == 4 and counts[4] >= target_4p:
                        continue
                    if np == 2 and counts[2] >= target_2p:
                        continue
                    line = json.dumps(compact, ensure_ascii=False, separators=(",", ":")) + "\n"
                    writer.write(line)
                    writer.flush()
                    counts[np] += 1
                    seen.add(eid)
                    bytes_written = out_path.stat().st_size
                    print(f"[ep {eid}] {np}p  total 2p={counts[2]} 4p={counts[4]}  size={bytes_written/1e6:.1f}MB", file=sys.stderr)
                    await asyncio.sleep(args.sleep)
        finally:
            writer.close()
            await ctx.close()

    print(json.dumps({"episodes_2p": counts[2], "episodes_4p": counts[4], "size_mb": bytes_written / 1e6}))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--top-teams", type=int, default=100)
    p.add_argument("--target", type=int, default=600, help="total episodes target (70/30 split 4p/2p)")
    p.add_argument("--max-bytes", type=int, default=88 * 1024 * 1024, help="output size cap (default 88MB)")
    p.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT / "episodes.jsonl.gz")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--extra-submissions", type=int, nargs="*", default=[52128366])
    return p


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(harvest(args))


if __name__ == "__main__":
    main()
