from __future__ import annotations

import argparse
import json
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


STATE_VERSION = 1
DEFAULT_STATE_PATH = "league_state.json"
ACTIVE_KINDS = ("active", "historical", "exploiter")
POOL_KINDS = ("anchors", "historical", "exploiters", "active", "retired")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _fallback_initial_rating() -> dict[str, float]:
    return {"mu": 1500.0, "sigma": 350.0}


def initial_rating() -> dict[str, float]:
    try:
        from v21_rating import initial_rating as imported_initial_rating

        value = imported_initial_rating()
        if isinstance(value, dict):
            return {str(k): float(v) for k, v in value.items()}
        if hasattr(value, "mu") and hasattr(value, "sigma"):
            return {"mu": float(value.mu), "sigma": float(value.sigma)}
    except Exception:
        pass
    return _fallback_initial_rating()


def rating_mu(player: dict[str, Any]) -> float:
    rating = player.get("rating") or {}
    return float(rating.get("mu", 1500.0))


def empty_state(now: str | None = None) -> dict[str, Any]:
    ts = now or utc_now()
    return {
        "version": STATE_VERSION,
        "created_at": ts,
        "updated_at": ts,
        "players": {},
        "pools": {name: [] for name in POOL_KINDS},
        "events": [],
    }


def normalize_state(state: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(state)
    state.setdefault("version", STATE_VERSION)
    state.setdefault("created_at", utc_now())
    state.setdefault("updated_at", state["created_at"])
    state.setdefault("players", {})
    state.setdefault("events", [])
    pools = state.setdefault("pools", {})
    for name in POOL_KINDS:
        pools.setdefault(name, [])
    rebuild_pools(state)
    return state


def pool_for_kind(kind: str, retired: bool = False) -> str:
    if retired:
        return "retired"
    if kind == "anchor":
        return "anchors"
    if kind == "historical":
        return "historical"
    if kind == "exploiter":
        return "exploiters"
    if kind == "active":
        return "active"
    raise ValueError(f"unknown player kind: {kind}")


def make_player(
    player_id: str,
    kind: str = "active",
    rating: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    retired: bool = False,
) -> dict[str, Any]:
    if not player_id:
        raise ValueError("player_id is required")
    pool_for_kind(kind, retired)
    return {
        "id": str(player_id),
        "kind": kind,
        "rating": rating or initial_rating(),
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "score": 0.0,
        "retired": bool(retired),
        "retire_reason": "",
        "metadata": dict(metadata or {}),
    }


def rebuild_pools(state: dict[str, Any]) -> None:
    pools = {name: [] for name in POOL_KINDS}
    for player_id, player in sorted(state.get("players", {}).items()):
        pool = pool_for_kind(str(player.get("kind", "active")), bool(player.get("retired", False)))
        pools[pool].append(player_id)
    state["pools"] = pools


def add_player(
    state: dict[str, Any],
    player_id: str,
    kind: str = "active",
    rating: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    state = normalize_state(state)
    if player_id in state["players"]:
        raise ValueError(f"player already exists: {player_id}")
    state["players"][player_id] = make_player(player_id, kind, rating, metadata)
    state["events"].append({"ts": now or utc_now(), "type": "add_player", "player": player_id, "kind": kind})
    state["updated_at"] = now or utc_now()
    rebuild_pools(state)
    return state


def retire_player(state: dict[str, Any], player_id: str, reason: str = "", now: str | None = None) -> dict[str, Any]:
    state = normalize_state(state)
    if player_id not in state["players"]:
        raise KeyError(player_id)
    player = state["players"][player_id]
    player["retired"] = True
    player["retire_reason"] = reason
    state["events"].append({"ts": now or utc_now(), "type": "retire_player", "player": player_id, "reason": reason})
    state["updated_at"] = now or utc_now()
    rebuild_pools(state)
    return state


def unretire_player(state: dict[str, Any], player_id: str, kind: str = "active", now: str | None = None) -> dict[str, Any]:
    state = normalize_state(state)
    if player_id not in state["players"]:
        raise KeyError(player_id)
    pool_for_kind(kind)
    player = state["players"][player_id]
    player["kind"] = kind
    player["retired"] = False
    player["retire_reason"] = ""
    state["events"].append({"ts": now or utc_now(), "type": "unretire_player", "player": player_id, "kind": kind})
    state["updated_at"] = now or utc_now()
    rebuild_pools(state)
    return state


def active_player_ids(state: dict[str, Any], include_anchors: bool = True) -> list[str]:
    state = normalize_state(state)
    ids: list[str] = []
    pools = state["pools"]
    if include_anchors:
        ids.extend(pools["anchors"])
    ids.extend(pools["active"])
    ids.extend(pools["historical"])
    ids.extend(pools["exploiters"])
    return ids


def record_match_result(state: dict[str, Any], scores: dict[str, float], now: str | None = None) -> dict[str, Any]:
    state = normalize_state(state)
    missing = [pid for pid in scores if pid not in state["players"]]
    if missing:
        raise KeyError(",".join(sorted(missing)))
    if not scores:
        raise ValueError("scores must not be empty")
    best = max(float(v) for v in scores.values())
    winners = {pid for pid, value in scores.items() if float(value) == best}
    for player_id, score in scores.items():
        player = state["players"][player_id]
        player["games"] = int(player.get("games", 0)) + 1
        player["score"] = float(player.get("score", 0.0)) + float(score)
        if len(winners) != 1:
            player["draws"] = int(player.get("draws", 0)) + 1
        elif player_id in winners:
            player["wins"] = int(player.get("wins", 0)) + 1
        else:
            player["losses"] = int(player.get("losses", 0)) + 1
    state["events"].append({"ts": now or utc_now(), "type": "match_result", "scores": dict(scores)})
    state["updated_at"] = now or utc_now()
    return state


class LeagueStateManager:
    def __init__(self, path: str | os.PathLike[str] = DEFAULT_STATE_PATH) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return empty_state()
        with self.path.open("r", encoding="utf-8") as handle:
            return normalize_state(json.load(handle))

    def save(self, state: dict[str, Any]) -> dict[str, Any]:
        state = normalize_state(state)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        return state

    def update(self, mutator, *args: Any, **kwargs: Any) -> dict[str, Any]:
        state = self.load()
        state = mutator(state, *args, **kwargs)
        return self.save(state)


def summarize_state(state: dict[str, Any]) -> dict[str, Any]:
    state = normalize_state(state)
    return {
        "version": state["version"],
        "players": len(state["players"]),
        "pools": {name: len(ids) for name, ids in state["pools"].items()},
        "top": sorted(
            ((pid, rating_mu(player), player.get("kind", "active")) for pid, player in state["players"].items()),
            key=lambda row: (-row[1], row[0]),
        )[:5],
    }


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V21 league_state.json pool manager")
    parser.add_argument("--state", default=DEFAULT_STATE_PATH)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")
    sub.add_parser("status")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--write", action="store_true")

    add = sub.add_parser("add")
    add.add_argument("player_id")
    add.add_argument("--kind", default="active", choices=("anchor", "historical", "exploiter", "active"))

    retire = sub.add_parser("retire")
    retire.add_argument("player_id")
    retire.add_argument("--reason", default="")

    args = parser.parse_args(list(argv) if argv is not None else None)
    manager = LeagueStateManager(args.state)

    if args.cmd == "init":
        state = manager.save(manager.load())
        _print_json(summarize_state(state))
        return 0
    if args.cmd == "status":
        _print_json(summarize_state(manager.load()))
        return 0
    if args.cmd == "add":
        _print_json(summarize_state(manager.update(add_player, args.player_id, args.kind)))
        return 0
    if args.cmd == "retire":
        _print_json(summarize_state(manager.update(retire_player, args.player_id, args.reason)))
        return 0
    if args.cmd == "smoke":
        state = empty_state()
        state = add_player(state, "anchor_v15", "anchor")
        state = add_player(state, "candidate_v21", "active", rating={"mu": 1600.0, "sigma": 80.0})
        state = record_match_result(state, {"anchor_v15": 0.0, "candidate_v21": 1.0})
        if args.write:
            manager.save(state)
        _print_json(summarize_state(state))
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
