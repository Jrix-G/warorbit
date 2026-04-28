#!/usr/bin/env python3
"""Fetch top-1 Orbit Wars leaderboard submission and analyze its replays.

Pipeline:
1. Open the Kaggle Orbit Wars leaderboard with an authenticated Playwright session.
2. Extract the first public leaderboard submission id.
3. Download all replays for that submission.
4. Produce a wins/losses report focused on the losses.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import async_playwright

# Allow imports from the repo root when this script is run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replay_dataset.collect_and_analyze import (
    DEFAULT_OUT,
    DEFAULT_PROFILE,
    aggregate,
    collect_for_submission,
    write_json,
)


LEADERBOARD_URL = "https://www.kaggle.com/competitions/orbit-wars/leaderboard"


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


async def get_top1_submission(profile: Path, headless: bool) -> Dict[str, Any]:
    """Return metadata for the top public leaderboard row."""
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            str(profile),
            headless=headless,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        seen_payloads: List[Any] = []

        async def capture_response(response):
            try:
                ctype = response.headers.get("content-type", "")
                if "json" not in ctype.lower():
                    return
                body = await response.text()
                parsed = json.loads(body)
            except Exception:
                return
            seen_payloads.append(parsed)

        page.on("response", lambda response: asyncio.create_task(capture_response(response)))
        await page.goto(LEADERBOARD_URL, wait_until="domcontentloaded")
        try:
            await page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        await page.wait_for_timeout(2000)

        # Prefer network payloads because they usually contain the leaderboard
        # rows directly, including `publicLeaderboardSubmissionId`.
        for payload in seen_payloads:
            row = _extract_top_row(payload)
            if row:
                await ctx.close()
                return row

        # DOM fallback: parse the rendered page and look for submission/team ids.
        text = await page.locator("body").inner_text(timeout=10000)
        await ctx.close()
        raise RuntimeError(
            "Unable to extract leaderboard top row from Kaggle responses. "
            f"Body preview: {text[:1000]!r}"
        )


def _extract_top_row(payload: Any) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of the top public leaderboard submission."""
    candidates: List[Dict[str, Any]] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if "publicLeaderboardSubmissionId" in obj and (
                "publicLeaderboardScoreFormatted" in obj or "teamName" in obj
            ):
                candidates.append(obj)
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(payload)
    if not candidates:
        return None

    def sort_key(item: Dict[str, Any]) -> Tuple[float, int]:
        score = item.get("publicLeaderboardScore")
        if score is None:
            formatted = item.get("publicLeaderboardScoreFormatted")
            try:
                score = float(formatted)
            except Exception:
                score = -1e18
        try:
            sub_id = int(item.get("publicLeaderboardSubmissionId") or -1)
        except Exception:
            sub_id = -1
        return float(score), sub_id

    return max(candidates, key=sort_key)


def _normalize_summary_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the row returned by Kaggle payloads to a compact dict."""
    team_name = row.get("teamName") or row.get("name") or row.get("displayName")
    score = row.get("publicLeaderboardScoreFormatted") or row.get("score")
    submission_id = row.get("publicLeaderboardSubmissionId") or row.get("submissionId")
    return {
        "teamName": team_name,
        "publicLeaderboardScoreFormatted": score,
        "publicLeaderboardSubmissionId": submission_id,
        "teamId": row.get("teamId"),
        "raw": row,
    }


def _build_loss_report(report: Dict[str, Any]) -> Dict[str, Any]:
    episodes = report["episodes"]
    ok = [e for e in episodes if "error" not in e]
    losses = [e for e in ok if e.get("our_reward") == -1]
    wins = [e for e in ok if e.get("our_reward") == 1]

    by_players = defaultdict(list)
    for e in losses:
        by_players[str(e.get("players"))].append(e)

    return {
        "submission_id": report["submission_id"],
        "episodes": len(episodes),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / max(len(ok), 1),
        "losses_only": losses,
        "losses_by_players": {
            k: {
                "count": len(v),
                "avg_steps": sum(x.get("steps") or 0 for x in v) / max(len(v), 1),
                "avg_max_planets": sum(x.get("max_planets") or 0 for x in v) / max(len(v), 1),
                "avg_final_planets": sum(x.get("final_planets") or 0 for x in v) / max(len(v), 1),
                "avg_max_ships": sum(x.get("max_ships") or 0 for x in v) / max(len(v), 1),
                "avg_score_gap": sum(x.get("score_gap") or 0 for x in v) / max(len(v), 1),
            }
            for k, v in by_players.items()
        },
    }


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    top_row = await get_top1_submission(args.profile, args.headless)
    top_row = _normalize_summary_row(top_row)
    submission_id = int(top_row["publicLeaderboardSubmissionId"])
    team_name = top_row.get("teamName")
    score = top_row.get("publicLeaderboardScoreFormatted")

    print(f"Top-1 leaderboard submission: {submission_id} | team={team_name} | score={score}")

    report = await collect_for_submission(submission_id, args.profile, args.output, args.headless)
    agg = aggregate(report)
    loss_report = _build_loss_report(report)

    sub_dir = args.output / f"submission_{submission_id}"
    write_json(sub_dir / "leaderboard_top1.json", {
        "submission_id": submission_id,
        "team_name": team_name,
        "score": score,
        "raw_top_row": top_row,
    })
    write_json(sub_dir / "leaderboard_summary.json", {
        "aggregate": agg,
        "loss_report": loss_report,
    })

    print(json.dumps({
        "top1": {
            "submission_id": submission_id,
            "team_name": team_name,
            "score": score,
        },
        "aggregate": agg,
        "loss_report": {
            "wins": loss_report["wins"],
            "losses": loss_report["losses"],
            "losses_by_players": loss_report["losses_by_players"],
        },
    }, indent=2, ensure_ascii=False))
    return {
        "submission_id": submission_id,
        "aggregate": agg,
        "loss_report": loss_report,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--headless", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
