from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Iterable

from v21_pool import (
    DEFAULT_STATE_PATH,
    LeagueStateManager,
    add_player,
    empty_state,
    normalize_state,
    rating_mu,
    rebuild_pools,
    record_match_result,
    retire_player,
    summarize_state,
    utc_now,
)


@dataclass(frozen=True)
class PromotionRules:
    min_games: int = 8
    promote_mu: float = 1550.0
    relegate_mu: float = 1450.0
    exploiter_promote_mu: float = 1575.0
    historical_keep_top: int = 8
    max_exploiters: int = 4
    retire_below_mu: float = 1300.0


def player_win_rate(player: dict[str, Any]) -> float:
    games = int(player.get("games", 0))
    if games <= 0:
        return 0.0
    return float(player.get("wins", 0)) / games


def classify_player(player: dict[str, Any], rules: PromotionRules = PromotionRules()) -> str:
    if bool(player.get("retired", False)):
        return "retired"
    kind = str(player.get("kind", "active"))
    if kind == "anchor":
        return "anchor"
    games = int(player.get("games", 0))
    mu = rating_mu(player)
    if games >= rules.min_games and mu <= rules.retire_below_mu:
        return "retired"
    if kind == "exploiter":
        if games >= rules.min_games and mu >= rules.exploiter_promote_mu:
            return "active"
        return "exploiter"
    if kind == "historical":
        if games >= rules.min_games and mu >= rules.promote_mu:
            return "active"
        return "historical"
    if games >= rules.min_games and mu <= rules.relegate_mu:
        return "historical"
    return "active"


def plan_transitions(state: dict[str, Any], rules: PromotionRules = PromotionRules()) -> list[dict[str, Any]]:
    state = normalize_state(state)
    transitions: list[dict[str, Any]] = []
    for player_id, player in sorted(state["players"].items()):
        current_kind = "retired" if player.get("retired") else player.get("kind", "active")
        target_kind = classify_player(player, rules)
        if target_kind != current_kind:
            transitions.append(
                {
                    "player": player_id,
                    "from": current_kind,
                    "to": target_kind,
                    "mu": rating_mu(player),
                    "games": int(player.get("games", 0)),
                }
            )
    return transitions


def league_metrics(state: dict[str, Any]) -> dict[str, Any]:
    try:
        from v21_metrics import summarize_league

        metrics = summarize_league(state)
        if isinstance(metrics, dict):
            return metrics
    except Exception:
        pass
    state = normalize_state(state)
    players = list(state["players"].values())
    rated = [rating_mu(player) for player in players]
    return {
        "players": len(players),
        "games": sum(int(player.get("games", 0)) for player in players),
        "rating_mu_avg": (sum(rated) / len(rated)) if rated else 0.0,
    }


def apply_transitions(
    state: dict[str, Any],
    transitions: Iterable[dict[str, Any]],
    now: str | None = None,
) -> dict[str, Any]:
    state = normalize_state(state)
    ts = now or utc_now()
    for transition in transitions:
        player_id = str(transition["player"])
        target = str(transition["to"])
        if player_id not in state["players"]:
            raise KeyError(player_id)
        player = state["players"][player_id]
        if target == "retired":
            player["retired"] = True
            player["retire_reason"] = "league_rule"
        else:
            player["kind"] = target
            player["retired"] = False
            player["retire_reason"] = ""
        state["events"].append(
            {
                "ts": ts,
                "type": "league_transition",
                "player": player_id,
                "from": transition.get("from"),
                "to": target,
            }
        )
    state["updated_at"] = ts
    rebuild_pools(state)
    return state


def select_pairings(state: dict[str, Any], limit: int = 8) -> list[tuple[str, str]]:
    state = normalize_state(state)
    active = [
        pid
        for pid, player in state["players"].items()
        if not player.get("retired") and player.get("kind") in {"active", "exploiter"}
    ]
    anchors = [
        pid
        for pid, player in state["players"].items()
        if not player.get("retired") and player.get("kind") == "anchor"
    ]
    active = sorted(active, key=lambda pid: (-rating_mu(state["players"][pid]), pid))
    anchors = sorted(anchors)
    pairs: list[tuple[str, str]] = []
    for player_id in active:
        for anchor_id in anchors:
            if player_id != anchor_id:
                pairs.append((player_id, anchor_id))
                if len(pairs) >= limit:
                    return pairs
    for idx, player_id in enumerate(active):
        for opponent_id in active[idx + 1 :]:
            pairs.append((player_id, opponent_id))
            if len(pairs) >= limit:
                return pairs
    return pairs


def make_smoke_state() -> dict[str, Any]:
    state = empty_state()
    state = add_player(state, "anchor_v15", "anchor", rating={"mu": 1500.0, "sigma": 50.0})
    state = add_player(state, "candidate_v21", "active", rating={"mu": 1600.0, "sigma": 80.0})
    state = add_player(state, "exploiter_fast", "exploiter", rating={"mu": 1580.0, "sigma": 90.0})
    for player_id in ("candidate_v21", "exploiter_fast"):
        state["players"][player_id]["games"] = 8
    state = record_match_result(state, {"anchor_v15": 0.0, "candidate_v21": 1.0})
    return state


def _rules_from_args(args: argparse.Namespace) -> PromotionRules:
    return PromotionRules(
        min_games=args.min_games,
        promote_mu=args.promote_mu,
        relegate_mu=args.relegate_mu,
        exploiter_promote_mu=args.exploiter_promote_mu,
        retire_below_mu=args.retire_below_mu,
    )


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V21 league planner and league_state.json manager")
    parser.add_argument("--state", default=DEFAULT_STATE_PATH)
    parser.add_argument("--min-games", type=int, default=PromotionRules.min_games)
    parser.add_argument("--promote-mu", type=float, default=PromotionRules.promote_mu)
    parser.add_argument("--relegate-mu", type=float, default=PromotionRules.relegate_mu)
    parser.add_argument("--exploiter-promote-mu", type=float, default=PromotionRules.exploiter_promote_mu)
    parser.add_argument("--retire-below-mu", type=float, default=PromotionRules.retire_below_mu)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("status")
    sub.add_parser("plan")
    apply_cmd = sub.add_parser("apply")
    apply_cmd.add_argument("--yes", action="store_true")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--write", action="store_true")

    args = parser.parse_args(list(argv) if argv is not None else None)
    manager = LeagueStateManager(args.state)
    rules = _rules_from_args(args)

    if args.cmd == "init":
        _print_json(summarize_state(manager.save(manager.load())))
        return 0
    if args.cmd == "status":
        state = manager.load()
        _print_json({"summary": summarize_state(state), "pairings": select_pairings(state)})
        return 0
    if args.cmd == "plan":
        _print_json({"transitions": plan_transitions(manager.load(), rules)})
        return 0
    if args.cmd == "apply":
        if not args.yes:
            raise SystemExit("apply requires --yes")
        state = manager.load()
        transitions = plan_transitions(state, rules)
        _print_json({"transitions": transitions, "summary": summarize_state(manager.save(apply_transitions(state, transitions)))})
        return 0
    if args.cmd == "smoke":
        state = make_smoke_state()
        transitions = plan_transitions(state, rules)
        result = {
            "summary": summarize_state(state),
            "metrics": league_metrics(state),
            "pairings": select_pairings(state, limit=4),
            "transitions": transitions,
        }
        if args.write:
            manager.save(state)
        _print_json(result)
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
