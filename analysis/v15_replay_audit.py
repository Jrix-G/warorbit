#!/usr/bin/env python3
"""V15 D1: replay-driven strategic audit over Kaggle top-10 4p replays.

Output:
  analysis/v15_replay_audit.json  — full bucketed metrics
  analysis/V15_FINDINGS.md        — top 5 actionable patterns

Replay schema (per JSON):
  steps[t][p]: {action, observation, reward, status}
  observation.planets[i] = [id, owner, x, y, radius, ships, prod]
  observation.fleets[i]  = [id, owner, x, y, angle, dest_id, ships]
  action = [[src_id, angle, ships], ...]
  rewards = [-1|0|+1] per player (final)
  info.EpisodeId / info.TeamNames

Manifest: episode_id, scores (parallel to submission_ids).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                eid = int(row["episode_id"])
                scores = json.loads(row["scores"])
                sub_ids = json.loads(row["submission_ids"])
            except Exception:
                continue
            out[eid] = {"scores": scores, "submission_ids": sub_ids}
    return out


def elo_bucket(score: float) -> str:
    # Calibrated to manifest distribution (n=10520, p95=1373, p99=1536, max=1833).
    if score >= 1500:
        return "top"          # top ~1% — leaderboard top 10
    if score >= 1300:
        return "high"         # top ~10%
    if score >= 1050:
        return "mid"          # middle 40%
    return "low"


def _infer_target(planets: list, src_id: int, angle: float) -> list | None:
    src = next((p for p in planets if int(p[0]) == src_id), None)
    if src is None:
        return None
    sx, sy = float(src[2]), float(src[3])
    dx, dy = math.cos(angle), math.sin(angle)
    best = None
    for p in planets:
        if int(p[0]) == src_id:
            continue
        vx = float(p[2]) - sx
        vy = float(p[3]) - sy
        proj = vx * dx + vy * dy
        if proj <= 0:
            continue
        perp = abs(vx * dy - vy * dx)
        threshold = float(p[4]) + 5.5
        if perp > threshold:
            continue
        score = (perp, proj)
        if best is None or score < best[0]:
            best = (score, p)
    return None if best is None else best[1]


def _classify(target_owner: int, me: int) -> str:
    if target_owner == -1:
        return "expand"
    if target_owner == me:
        return "support"
    return "attack"


def _dist(a: list, b: list) -> float:
    return math.hypot(float(a[2]) - float(b[2]), float(a[3]) - float(b[3]))


def analyze_episode(replay: dict, manifest_row: dict[str, Any] | None) -> list[dict] | None:
    """Return list of per-player stat dicts for this episode, or None on error."""
    steps = replay.get("steps") or []
    if len(steps) < 5 or not isinstance(steps[0], list) or len(steps[0]) < 2:
        return None
    rewards = replay.get("rewards") or []
    n_players = len(steps[0])
    if n_players != 4:
        return None

    scores = manifest_row["scores"] if manifest_row else [None] * n_players
    sub_ids = manifest_row["submission_ids"] if manifest_row else [None] * n_players

    # Per-player accumulators
    stats: list[dict[str, Any]] = []
    for p in range(n_players):
        stats.append({
            "elo": scores[p] if p < len(scores) else None,
            "submission_id": sub_ids[p] if p < len(sub_ids) else None,
            "won": int(rewards[p] == 1) if p < len(rewards) else 0,
            "first_action_turn": None,
            "first_attack_turn": None,
            "action_counts": Counter(),       # by classification
            "ships_sent_by_class": Counter(), # totals
            "send_ratios": [],                # ships / src_ships at time of send
            "target_dists": defaultdict(list),  # phase -> list of distances
            "multi_source_turns": 0,
            "total_action_turns": 0,
            "moves_total": 0,
            "ships_built_proxy": 0.0,  # sum of own prod per turn at planets
            "final_planets": 0,
            "peak_planets": 0,
            "total_steps_alive": 0,
            "ship_send_buckets": Counter(),   # bucketed send fraction
        })

    total_turns = len(steps)

    for t, step_entries in enumerate(steps):
        phase = "early" if t < 50 else ("mid" if t < 130 else "late")
        for p, entry in enumerate(step_entries):
            obs = entry.get("observation") or {}
            planets = obs.get("planets") or []
            if not planets:
                continue
            my_planets = [pl for pl in planets if int(pl[1]) == p]
            stats[p]["peak_planets"] = max(stats[p]["peak_planets"], len(my_planets))
            if my_planets:
                stats[p]["total_steps_alive"] += 1
                if t == total_turns - 1:
                    stats[p]["final_planets"] = len(my_planets)
                stats[p]["ships_built_proxy"] += sum(float(pl[6]) for pl in my_planets)

            action = entry.get("action") or []
            if not action:
                continue
            stats[p]["total_action_turns"] += 1
            if stats[p]["first_action_turn"] is None:
                stats[p]["first_action_turn"] = t

            # Multi-source: count distinct sources used this turn (>=2)
            distinct_srcs = {int(m[0]) for m in action if len(m) >= 3}
            if len(distinct_srcs) >= 2:
                stats[p]["multi_source_turns"] += 1

            for m in action:
                if not isinstance(m, (list, tuple)) or len(m) < 3:
                    continue
                src_id = int(m[0])
                angle = float(m[1])
                sent = int(m[2])
                if sent <= 0:
                    continue
                stats[p]["moves_total"] += 1
                src = next((pl for pl in planets if int(pl[0]) == src_id), None)
                if src is None:
                    continue
                src_ships = float(src[5])
                if src_ships > 0:
                    ratio = min(1.0, sent / src_ships)
                    stats[p]["send_ratios"].append(ratio)
                    bucket = "0-40" if ratio < 0.4 else "40-70" if ratio < 0.7 else "70-90" if ratio < 0.9 else "90-100"
                    stats[p]["ship_send_buckets"][bucket] += 1

                tgt = _infer_target(planets, src_id, angle)
                if tgt is None:
                    cls = "unknown"
                else:
                    cls = _classify(int(tgt[1]), p)
                    stats[p]["target_dists"][phase].append(_dist(src, tgt))
                stats[p]["action_counts"][cls] += 1
                stats[p]["ships_sent_by_class"][cls] += sent
                if cls == "attack" and stats[p]["first_attack_turn"] is None:
                    stats[p]["first_attack_turn"] = t

    # Reduce
    for s in stats:
        s["send_ratio_mean"] = statistics.mean(s["send_ratios"]) if s["send_ratios"] else None
        s["send_ratio_median"] = statistics.median(s["send_ratios"]) if s["send_ratios"] else None
        s["target_dist_mean"] = {
            ph: statistics.mean(v) if v else None for ph, v in s["target_dists"].items()
        }
        # Mission distribution
        total_acts = sum(s["action_counts"].values()) or 1
        s["action_dist"] = {k: v / total_acts for k, v in s["action_counts"].items()}
        total_ships = sum(s["ships_sent_by_class"].values()) or 1
        s["ship_alloc"] = {k: v / total_ships for k, v in s["ships_sent_by_class"].items()}
        # Strip raw lists for output
        del s["send_ratios"]
        del s["target_dists"]
        s["action_counts"] = dict(s["action_counts"])
        s["ships_sent_by_class"] = dict(s["ships_sent_by_class"])
        s["ship_send_buckets"] = dict(s["ship_send_buckets"])

    return stats


def aggregate(per_player_records: list[dict]) -> dict[str, Any]:
    """Group by ELO bucket, then by won/lost. Compute summary stats."""
    buckets: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in per_player_records:
        if r["elo"] is None:
            continue
        b = elo_bucket(r["elo"])
        buckets[(b, r["won"])].append(r)

    def _agg_list(records: list[dict]) -> dict[str, Any]:
        if not records:
            return {"n": 0}

        def _vals(key: str) -> list[float]:
            return [r[key] for r in records if isinstance(r.get(key), (int, float))]

        def _mean(xs: list[float]) -> float | None:
            return statistics.mean(xs) if xs else None

        first_acts = _vals("first_action_turn")
        first_atks = _vals("first_attack_turn")
        send_ratios = _vals("send_ratio_mean")

        # Mission distribution averaged
        dist_keys = ("expand", "attack", "support", "unknown")
        dist_avg: dict[str, float] = {}
        for k in dist_keys:
            xs = [r["action_dist"].get(k, 0.0) for r in records]
            dist_avg[k] = _mean(xs) if xs else None
        alloc_avg: dict[str, float] = {}
        for k in dist_keys:
            xs = [r["ship_alloc"].get(k, 0.0) for r in records]
            alloc_avg[k] = _mean(xs) if xs else None

        td_phase: dict[str, float] = {}
        for ph in ("early", "mid", "late"):
            xs = [r["target_dist_mean"].get(ph) for r in records if r["target_dist_mean"].get(ph) is not None]
            td_phase[ph] = _mean(xs)

        send_bucket_sum: Counter = Counter()
        for r in records:
            send_bucket_sum.update(r.get("ship_send_buckets", {}))
        total_b = sum(send_bucket_sum.values()) or 1
        send_bucket_dist = {k: v / total_b for k, v in send_bucket_sum.items()}

        ms = _vals("multi_source_turns")
        tat = _vals("total_action_turns")
        ms_frac = [r["multi_source_turns"] / r["total_action_turns"] for r in records if r.get("total_action_turns")]

        return {
            "n": len(records),
            "first_action_turn_mean": _mean(first_acts),
            "first_attack_turn_mean": _mean(first_atks),
            "send_ratio_mean": _mean(send_ratios),
            "mission_dist": dist_avg,
            "ship_alloc": alloc_avg,
            "target_dist_by_phase": td_phase,
            "send_ratio_buckets": send_bucket_dist,
            "multi_source_turn_frac_mean": _mean(ms_frac),
            "peak_planets_mean": _mean(_vals("peak_planets")),
            "final_planets_mean": _mean(_vals("final_planets")),
            "elo_mean": _mean(_vals("elo")),
        }

    out: dict[str, Any] = {}
    for bkt in ("top", "high", "mid", "low"):
        out[bkt] = {
            "won": _agg_list(buckets.get((bkt, 1), [])),
            "lost": _agg_list(buckets.get((bkt, 0), [])),
        }
    out["all_records"] = len(per_player_records)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-dir", default=r"D:\warorbit_kaggle_raw\orbit-wars-top10-episodes-2026-05-04\episodes\episodes")
    ap.add_argument("--manifest", default=r"D:\warorbit_kaggle_raw\orbit-wars-top10-episodes-2026-05-04\manifest.csv")
    ap.add_argument("--out", default=r"analysis\v15_replay_audit.json")
    ap.add_argument("--limit", type=int, default=0, help="limit number of episodes (0=all)")
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    files = [p for p in sorted(Path(args.episodes_dir).glob("*.json")) if p.stem.isdigit()]
    if args.limit > 0:
        files = files[: args.limit]

    all_records: list[dict] = []
    t0 = time.time()
    n_ok = 0
    n_skip = 0
    for i, fp in enumerate(files):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                replay = json.load(f)
        except Exception:
            n_skip += 1
            continue
        eid = int(replay.get("info", {}).get("EpisodeId") or fp.stem)
        m = manifest.get(eid)
        recs = analyze_episode(replay, m)
        if recs is None:
            n_skip += 1
            continue
        all_records.extend(recs)
        n_ok += 1
        if (i + 1) % 50 == 0:
            sys.stderr.write(f"  processed {i+1}/{len(files)} ok={n_ok} skip={n_skip} elapsed={time.time()-t0:.1f}s\n")

    agg = aggregate(all_records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_episodes": n_ok,
            "n_skipped": n_skip,
            "n_player_records": len(all_records),
            "buckets": agg,
        }, f, indent=2)
    print(f"OK episodes={n_ok} skip={n_skip} records={len(all_records)} -> {out_path}")


if __name__ == "__main__":
    main()
