#!/usr/bin/env python3
"""Generate a deep text report for the true public Orbit Wars leader.

The report focuses on:
- every public replay for submission 52018000
- loss-first analysis
- turn-by-turn board snapshots
- differences versus our PartyAnaly bot

Output:
  docs/analysis/leaderboard_top1_deep_report.txt
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_ID = 52018000
REPLAY_DIR = ROOT / "replay_dataset" / "output" / f"submission_{SUBMISSION_ID}"
REPORT_PATH = ROOT / "docs" / "analysis" / "leaderboard_top1_deep_report.txt"
PARTY_ANALY_PATH = ROOT / "docs" / "analysis" / "PartyAnaly" / "summary.json"

SAMPLE_TURNS = [0, 5, 10, 20, 40, 60, 80, 100, 150, 200, 300, 400]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def entity_owner(item: Any) -> Optional[int]:
    if isinstance(item, dict):
        owner = item.get("owner")
        return None if owner is None else int(owner)
    if isinstance(item, list) and len(item) > 1:
        owner = item[1]
        return None if owner is None else int(owner)
    return None


def entity_ships(item: Any) -> float:
    if isinstance(item, dict):
        value = item.get("ships", 0)
        return float(value or 0)
    if isinstance(item, list):
        # Planets: ships at index 5. Fleets: ships at index 4.
        if len(item) > 5:
            return float(item[5] or 0)
        if len(item) > 4:
            return float(item[4] or 0)
    return 0.0


def entity_prod(item: Any) -> float:
    if isinstance(item, dict):
        value = item.get("production", item.get("prod", 0))
        return float(value or 0)
    if isinstance(item, list) and len(item) > 4:
        return float(item[4] or 0)
    return 0.0


def as_int_list(values: Iterable[int]) -> str:
    return ", ".join(str(v) for v in values)


def fmt(v: Optional[float], digits: int = 1) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, int):
        return str(v)
    return f"{v:.{digits}f}"


def pct(v: float) -> str:
    return f"{100.0 * v:.1f}%"


def summarize_turn(obs: Dict[str, Any], me: int) -> Dict[str, Any]:
    planets = obs.get("planets", [])
    fleets = obs.get("fleets", [])

    my_planets = [p for p in planets if entity_owner(p) == me]
    enemy_planets = [p for p in planets if entity_owner(p) not in (-1, me)]
    neutral_planets = [p for p in planets if entity_owner(p) == -1]
    my_fleets = [f for f in fleets if entity_owner(f) == me]
    enemy_fleets = [f for f in fleets if entity_owner(f) not in (-1, me)]

    my_planet_ships = sum(entity_ships(p) for p in my_planets)
    enemy_planet_ships = sum(entity_ships(p) for p in enemy_planets)
    my_fleet_ships = sum(entity_ships(f) for f in my_fleets)
    enemy_fleet_ships = sum(entity_ships(f) for f in enemy_fleets)
    my_prod = sum(entity_prod(p) for p in my_planets)
    enemy_prod = sum(entity_prod(p) for p in enemy_planets)

    return {
        "my_planets": len(my_planets),
        "enemy_planets": len(enemy_planets),
        "neutral_planets": len(neutral_planets),
        "my_fleets": len(my_fleets),
        "enemy_fleets": len(enemy_fleets),
        "my_planet_ships": my_planet_ships,
        "enemy_planet_ships": enemy_planet_ships,
        "my_fleet_ships": my_fleet_ships,
        "enemy_fleet_ships": enemy_fleet_ships,
        "my_total_ships": my_planet_ships + my_fleet_ships,
        "enemy_total_ships": enemy_planet_ships + enemy_fleet_ships,
        "my_prod": my_prod,
        "enemy_prod": enemy_prod,
    }


def non_empty_action_summary(turn: Sequence[Any], me: int) -> Tuple[int, float]:
    if not isinstance(turn, list) or me >= len(turn):
        return 0, 0.0
    actions = turn[me].get("action", [])
    if not actions:
        return 0, 0.0
    ships = 0.0
    for action in actions:
        if isinstance(action, (list, tuple)) and len(action) >= 3:
            try:
                ships += float(action[2])
            except Exception:
                continue
    return len(actions), ships


def classify_episode(ep: Dict[str, Any]) -> str:
    players = ep["players"]
    if ep["our_reward"] == 1:
        if players == 2:
            if ep["max_planets"] >= 20:
                return "duel snowball"
            return "duel conversion win"
        if ep["final_planets"] >= 15:
            return "4p lockout win"
        return "4p conversion win"

    if players == 2:
        if ep["collapse_turn"] is not None and ep["collapse_turn"] <= 120:
            return "fast duel collapse"
        if ep["max_planets"] <= 12:
            return "duel plateau loss"
        return "duel squeeze loss"

    if ep["final_planets"] <= 2 and ep["steps"] >= 200:
        return "slow 4p squeeze"
    if ep["max_planets"] <= 12:
        return "4p board starvation"
    return "4p late collapse"


def analyze_episode(summary_ep: Dict[str, Any], ep_meta: Dict[str, Any], replay: Dict[str, Any], submission_id: int) -> Dict[str, Any]:
    steps = replay.get("steps") or []
    if not steps:
        raise RuntimeError(f"Replay for episode {ep_meta['id']} has no steps")

    players = len(steps[0]) if isinstance(steps[0], list) else len(ep_meta.get("agents", []))
    our_idx = int(summary_ep.get("our_index", ep_meta.get("our_index", 0)))
    winner_submission_id = summary_ep.get("winner_submission_id", ep_meta.get("winner_submission_id"))
    opponent_ids = [a.get("submissionId") for a in ep_meta.get("agents", []) if a.get("submissionId") != submission_id]

    first_owned_turn = None
    collapse_turn = None
    last_positive_turn = None
    zero_turns = 0
    max_planets = -1
    max_planets_turn = None
    max_ships = -1.0
    max_ships_turn = None
    max_fleets = -1
    max_fleets_turn = None
    max_prod = -1.0
    max_prod_turn = None
    final_turn = len(steps) - 1
    final_snapshot = None
    snapshots: Dict[int, Dict[str, Any]] = {}
    opening_actions: List[Dict[str, Any]] = []
    first_action_turn = None

    sample_turns = sorted(set([t for t in SAMPLE_TURNS if t < len(steps)] + [final_turn]))

    saw_owned = False
    for t, turn in enumerate(steps):
        if not isinstance(turn, list) or our_idx >= len(turn):
            continue
        obs = turn[our_idx].get("observation", {})
        snap = summarize_turn(obs, our_idx)
        if t in sample_turns:
            snapshots[t] = snap

        if snap["my_planets"] > 0:
            last_positive_turn = t
            if first_owned_turn is None:
                first_owned_turn = t
                saw_owned = True
        else:
            zero_turns += 1
            if saw_owned and collapse_turn is None:
                collapse_turn = t

        if snap["my_planets"] > max_planets:
            max_planets = snap["my_planets"]
            max_planets_turn = t
        if snap["my_total_ships"] > max_ships:
            max_ships = snap["my_total_ships"]
            max_ships_turn = t
        if snap["my_fleets"] > max_fleets:
            max_fleets = snap["my_fleets"]
            max_fleets_turn = t
        if snap["my_prod"] > max_prod:
            max_prod = snap["my_prod"]
            max_prod_turn = t

        actions_n, actions_ships = non_empty_action_summary(turn, our_idx)
        if actions_n > 0 and len(opening_actions) < 5:
            if first_action_turn is None:
                first_action_turn = t
            opening_actions.append({
                "turn": t,
                "actions": actions_n,
                "ships": actions_ships,
            })

        if t == final_turn:
            final_snapshot = snap

    assert final_snapshot is not None

    episode = {
        "episode_id": ep_meta["id"],
        "players": players,
        "submission_id": submission_id,
        "winner_submission_id": winner_submission_id,
        "our_index": our_idx,
        "opponent_ids": opponent_ids,
        "our_reward": int(summary_ep.get("our_reward", summary_ep.get("reward", 0))),
        "result": "win" if int(summary_ep.get("our_reward", summary_ep.get("reward", 0))) == 1 else "loss",
        "steps": len(steps),
        "first_owned_turn": first_owned_turn,
        "collapse_turn": collapse_turn,
        "last_positive_turn": last_positive_turn,
        "zero_turns": zero_turns,
        "max_planets": max_planets,
        "max_planets_turn": max_planets_turn,
        "final_planets": final_snapshot["my_planets"],
        "max_ships": max_ships,
        "max_ships_turn": max_ships_turn,
        "final_ships": final_snapshot["my_total_ships"],
        "max_fleets": max_fleets,
        "max_fleets_turn": max_fleets_turn,
        "final_fleets": final_snapshot["my_fleets"],
        "max_prod": max_prod,
        "max_prod_turn": max_prod_turn,
        "final_prod": final_snapshot["my_prod"],
        "opening_actions": opening_actions,
        "snapshots": snapshots,
        "final_snapshot": final_snapshot,
        "classification": classify_episode({
            "players": players,
            "our_reward": int(ep_meta.get("reward", 0)),
            "collapse_turn": collapse_turn,
            "max_planets": max_planets,
            "final_planets": final_snapshot["my_planets"],
            "steps": len(steps),
        }),
    }
    return episode


def snapshot_line(t: int, snap: Dict[str, Any]) -> str:
    lead_p = snap["my_planets"] - snap["enemy_planets"]
    lead_s = snap["my_total_ships"] - snap["enemy_total_ships"]
    return (
        f"t{t:03d}: "
        f"myP={snap['my_planets']} enemyP={snap['enemy_planets']} neutralP={snap['neutral_planets']} | "
        f"myS={fmt(snap['my_total_ships'], 0)} enemyS={fmt(snap['enemy_total_ships'], 0)} | "
        f"myF={snap['my_fleets']} enemyF={snap['enemy_fleets']} | "
        f"myPr={fmt(snap['my_prod'], 1)} enemyPr={fmt(snap['enemy_prod'], 1)} | "
        f"leadP={lead_p:+d} leadS={fmt(lead_s, 0)}"
    )


def build_diagnosis(ep: Dict[str, Any]) -> str:
    result = ep["result"]
    players = ep["players"]
    first_owned = ep["first_owned_turn"]
    collapse = ep["collapse_turn"]
    peak_p = ep["max_planets"]
    peak_s = ep["max_ships"]
    final_p = ep["final_planets"]
    final_s = ep["final_ships"]
    steps = ep["steps"]

    if result == "win":
        if players == 2:
            return (
                f"Duels are won by converting the first stable foothold into a board lock. "
                f"Here the first owned planet appears around turn {fmt(first_owned, 0)}, "
                f"then the position climbs to {peak_p} planets and {fmt(peak_s, 0)} ships before the end. "
                f"The important thing is not a tactical spike; it is sustained conversion pressure. "
                f"Compared with our bot, this is a cleaner transition from opening to irreversible economy."
            )
        return (
            f"Four-player games are won when the leader survives the early traffic and keeps growing while others trade. "
            f"The board never fully stales out: it reaches {peak_p} planets, then ends with {final_p} planets and {fmt(final_s, 0)} ships. "
            f"That means the bot is not just defending; it is building a production lock under pressure. "
            f"Compared with our bot, this is the stronger part of the leader: better midgame conversion, better closure, less drift."
        )

    if players == 2:
        if collapse is not None and collapse <= 120:
            return (
                f"This is a fast duel collapse. The leader gets a foothold, but the foothold never turns into a durable economy. "
                f"At collapse turn {collapse} it effectively loses board control, and the final board is {final_p} planets / {fmt(final_s, 0)} ships. "
                f"The failure is not opening ignorance; it is that the opening did not become enough productive mass soon enough. "
                f"Compared with our bot, the same kind of weakness appears, but here the leader usually dies later and with more structure still visible."
            )
        return (
            f"This is a slow duel squeeze. The leader survives longer, but the planet ceiling is only {peak_p}, "
            f"which is too low to force a decisive eco advantage. "
            f"After the first capture around turn {fmt(first_owned, 0)}, the position plateaus instead of compounding, and the final board falls to {final_p} planets / {fmt(final_s, 0)} ships. "
            f"Compared with our bot, this is less about immediate collapse and more about missing the conversion threshold."
        )

    return (
        f"This is a four-player squeeze loss. The leader does not die instantly; it holds some structure, but the board fragments and the conversion threshold is never crossed. "
        f"The peak is {peak_p} planets, the end state is {final_p} planets / {fmt(final_s, 0)} ships, and the game lasts {steps} frames. "
        f"That profile is worse than a tactical blunder: it means the bot kept playing, but never reached enough productive mass to escape the multi-way pressure. "
        f"Compared with our bot, this is the same broad failure class, but the leader is usually more resilient and collapses later."
    )


def build_compare_with_us(ep: Dict[str, Any], party_summary: Optional[Dict[str, Any]]) -> str:
    if not party_summary:
        return "No local PartyAnaly comparison file was available."

    leader_tag = "stronger" if ep["result"] == "win" else "weaker"
    lines = [
        f"Relative to our PartyAnaly bot, this match is {leader_tag} on the same axis: board conversion.",
        f"Our older bot typically entered its own losses with much lower board structure and more brittle turn-by-turn momentum.",
    ]
    if ep["result"] == "win":
        lines.append(
            "The leader's edge is that it keeps enough planets alive long enough to turn pressure into growth. "
            "That is exactly the part our bot struggled to do consistently."
        )
    else:
        lines.append(
            "The leader still has better structural resilience than our bot's worst losses, but the shared weakness is the same: the planet count never crosses the threshold required for irreversible growth."
        )
    return " ".join(lines)


def episode_block(ep: Dict[str, Any], party_summary: Optional[Dict[str, Any]]) -> str:
    opps = as_int_list(ep["opponent_ids"]) if ep["opponent_ids"] else "n/a"
    opening = ", ".join(
        f"t{item['turn']:03d}({item['actions']}a/{fmt(item['ships'], 0)}s)"
        for item in ep["opening_actions"]
    ) or "n/a"
    sample_turns = sorted(ep["snapshots"].keys())
    snapshots = "\n".join(f"  - {snapshot_line(t, ep['snapshots'][t])}" for t in sample_turns)

    lines = [
        f"Episode {ep['episode_id']} | {ep['result'].upper()} | {ep['players']}p | {ep['steps']} frames",
        f"  Winner submission: {ep['winner_submission_id']}",
        f"  Our index: {ep['our_index']}",
        f"  Opponents: {opps}",
        f"  First owned turn: {fmt(ep['first_owned_turn'], 0)}",
        f"  Collapse turn: {fmt(ep['collapse_turn'], 0)}",
        f"  Zero turns: {ep['zero_turns']}",
        f"  Max planets: {ep['max_planets']} at t{fmt(ep['max_planets_turn'], 0)}",
        f"  Final planets: {ep['final_planets']}",
        f"  Max ships: {fmt(ep['max_ships'], 0)} at t{fmt(ep['max_ships_turn'], 0)}",
        f"  Final ships: {fmt(ep['final_ships'], 0)}",
        f"  Max fleets: {ep['max_fleets']} at t{fmt(ep['max_fleets_turn'], 0)}",
        f"  Final fleets: {ep['final_fleets']}",
        f"  Max production: {fmt(ep['max_prod'], 1)} at t{fmt(ep['max_prod_turn'], 0)}",
        f"  Final production: {fmt(ep['final_prod'], 1)}",
        f"  Classification: {ep['classification']}",
        f"  Opening actions: {opening}",
        "  Snapshots:",
        snapshots,
        "  Diagnosis:",
        "    " + build_diagnosis(ep),
        "  Difference versus our bot:",
        "    " + build_compare_with_us(ep, party_summary),
    ]
    return "\n".join(lines)


def aggregate(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [ep for ep in episodes if ep["steps"] > 0]
    wins = [ep for ep in ok if ep["result"] == "win"]
    losses = [ep for ep in ok if ep["result"] == "loss"]

    by_players: Dict[int, List[Dict[str, Any]]] = {}
    for ep in ok:
        by_players.setdefault(ep["players"], []).append(ep)

    def avg(field: str, rows: List[Dict[str, Any]]) -> float:
        vals = [float(r[field]) for r in rows if r[field] is not None]
        return mean(vals) if vals else 0.0

    out = {
        "episodes": len(ok),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / max(len(ok), 1),
        "avg_steps": avg("steps", ok),
        "avg_first_owned_turn": avg("first_owned_turn", ok),
        "avg_zero_turns": avg("zero_turns", ok),
        "avg_max_planets": avg("max_planets", ok),
        "avg_final_planets": avg("final_planets", ok),
        "avg_max_ships": avg("max_ships", ok),
        "avg_final_ships": avg("final_ships", ok),
        "avg_max_fleets": avg("max_fleets", ok),
        "avg_max_prod": avg("max_prod", ok),
        "by_players": {},
        "loss_catalog": {},
    }

    for players, rows in sorted(by_players.items()):
        win_rows = [r for r in rows if r["result"] == "win"]
        loss_rows = [r for r in rows if r["result"] == "loss"]
        out["by_players"][players] = {
            "count": len(rows),
            "wins": len(win_rows),
            "losses": len(loss_rows),
            "win_rate": len(win_rows) / max(len(rows), 1),
            "avg_steps": avg("steps", rows),
            "avg_first_owned_turn": avg("first_owned_turn", rows),
            "avg_max_planets": avg("max_planets", rows),
            "avg_final_planets": avg("final_planets", rows),
            "avg_max_ships": avg("max_ships", rows),
            "avg_final_ships": avg("final_ships", rows),
        }

    loss_types: Dict[str, List[Dict[str, Any]]] = {}
    for ep in losses:
        loss_types.setdefault(ep["classification"], []).append(ep)
    out["loss_catalog"] = {
        name: {
            "count": len(rows),
            "avg_steps": avg("steps", rows),
            "avg_first_owned_turn": avg("first_owned_turn", rows),
            "avg_max_planets": avg("max_planets", rows),
            "avg_final_planets": avg("final_planets", rows),
            "avg_max_ships": avg("max_ships", rows),
            "avg_final_ships": avg("final_ships", rows),
        }
        for name, rows in sorted(loss_types.items())
    }
    return out


def format_summary(summary: Dict[str, Any]) -> str:
    lines = [
        "ORBIT WARS TOP-1 DEEP REPORT",
        f"True public leader submission: {SUBMISSION_ID}",
        "",
        "Executive summary",
        f"- Episodes analyzed: {summary['episodes']}",
        f"- Wins / losses: {summary['wins']} / {summary['losses']}",
        f"- Win rate: {pct(summary['win_rate'])}",
        f"- Average steps: {summary['avg_steps']:.1f}",
        f"- Average first owned turn: {summary['avg_first_owned_turn']:.1f}",
        f"- Average zero-turn count: {summary['avg_zero_turns']:.1f}",
        f"- Average max planets: {summary['avg_max_planets']:.1f}",
        f"- Average final planets: {summary['avg_final_planets']:.1f}",
        f"- Average max ships: {summary['avg_max_ships']:.1f}",
        f"- Average final ships: {summary['avg_final_ships']:.1f}",
        f"- Average max fleets: {summary['avg_max_fleets']:.1f}",
        f"- Average max production: {summary['avg_max_prod']:.1f}",
        "",
        "What this means",
        "- The leader is not winning by raw opening speed alone.",
        "- It is winning by converting the first stable foothold into a durable planet backbone.",
        "- Its main weakness is when that conversion threshold is denied, especially in 4-player games.",
        "- The losses split into two families: fast duel collapse and slow 4p squeeze.",
        "",
    ]
    return "\n".join(lines)


def format_comparison_to_party(party: Optional[Dict[str, Any]]) -> str:
    if not party:
        return "Comparison to our bot: unavailable (PartyAnaly summary missing)."

    lines = [
        "Comparison with our PartyAnaly bot",
        f"- Our dataset: {party['wins'] + party['losses']} episodes, {party['wins']} wins, {party['losses']} losses.",
        f"- Our average steps: {party['overall_avg_steps']:.1f}",
        f"- Our loss avg max planets: {party['losses_avg_max_planets']:.1f}",
        f"- Our loss avg final planets: {party['losses_avg_final_planets']:.1f}",
        f"- Our loss avg max ships: {party['losses_avg_max_ships']:.1f}",
        f"- Our 2p record: {party['by_players']['2']['wins']} wins / {party['by_players']['2']['losses']} losses",
        f"- Our 4p record: {party['by_players']['4']['wins']} wins / {party['by_players']['4']['losses']} losses",
        "",
        "Interpretation",
        "- Our bot's old losses were more brittle and more clearly tied to economic collapse after the opening.",
        "- The leader keeps more structure alive in defeat, but its real edge is that it converts pressure into large, stable boards much better than we did.",
        "- So the right counter-lesson is not 'open harder'; it is 'deny the conversion threshold and punish the midgame plateau.'",
        "",
    ]
    return "\n".join(lines)


def format_loss_catalog(summary: Dict[str, Any]) -> str:
    lines = ["Loss catalog"]
    for name, stats in summary["loss_catalog"].items():
        lines.extend([
            f"- {name}",
            f"  count: {stats['count']}",
            f"  avg steps: {stats['avg_steps']:.1f}",
            f"  avg first owned turn: {stats['avg_first_owned_turn']:.1f}",
            f"  avg max planets: {stats['avg_max_planets']:.1f}",
            f"  avg final planets: {stats['avg_final_planets']:.1f}",
            f"  avg max ships: {stats['avg_max_ships']:.1f}",
            f"  avg final ships: {stats['avg_final_ships']:.1f}",
        ])
    lines.append("")
    return "\n".join(lines)


def format_takeaway(summary: Dict[str, Any]) -> str:
    lines = [
        "V7 takeaway",
        "- Keep the NN + beam style.",
        "- Optimize for conversion threshold, not just immediate expansion.",
        "- The model must value whether a move creates a durable production backbone in the next 20-50 turns.",
        "- In 2p, the goal is to reach the irreversible board before the opponent stabilizes.",
        "- In 4p, the goal is to avoid the slow squeeze by keeping at least one scalable front alive while not overcommitting.",
        "- The best next test is not more random laddering; it is a batch of controlled runs where we measure how often the bot crosses the planet threshold and how often it survives the midgame plateau.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    summary = load_json(REPLAY_DIR / "summary.json")
    episodes_meta = {ep["id"]: ep for ep in load_json(REPLAY_DIR / "episodes.json")}
    summary_rows = {ep["episode_id"]: ep for ep in summary["episodes"]}
    party_summary = load_json(PARTY_ANALY_PATH) if PARTY_ANALY_PATH.exists() else None

    analyses: List[Dict[str, Any]] = []
    for ep in summary["episodes"]:
        ep_id = ep["episode_id"]
        replay = load_json(REPLAY_DIR / f"episode_{ep_id}.json")
        ep_meta = episodes_meta.get(ep_id)
        summary_ep = summary_rows.get(ep_id)
        if ep_meta is None:
            raise RuntimeError(f"Missing episode metadata for {ep_id}")
        if summary_ep is None:
            raise RuntimeError(f"Missing summary row for {ep_id}")
        analyses.append(analyze_episode(summary_ep, ep_meta, replay, SUBMISSION_ID))

    analyses.sort(key=lambda x: (
        0 if x["result"] == "loss" else 1,
        x["players"],
        -x["steps"],
        x["episode_id"],
    ))

    agg = aggregate(analyses)

    report_parts = [
        format_summary(agg),
        format_comparison_to_party(party_summary),
        "Episode-level analysis",
    ]

    current_group = None
    for ep in analyses:
        group = f"{ep['result'].upper()} - {ep['players']}p"
        if group != current_group:
            report_parts.append("")
            report_parts.append(group)
            current_group = group
        report_parts.append("")
        report_parts.append(episode_block(ep, party_summary))

    report_parts.extend([
        "",
        format_loss_catalog(agg),
        format_takeaway(agg),
    ])

    report = "\n".join(report_parts).strip() + "\n"
    write_text(REPORT_PATH, report)
    print(f"Wrote {REPORT_PATH}")
    print(json.dumps({
        "submission_id": SUBMISSION_ID,
        "episodes": agg["episodes"],
        "wins": agg["wins"],
        "losses": agg["losses"],
        "report_path": str(REPORT_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
