from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence


ROOT = Path(__file__).resolve().parents[2]

RANK_RE = re.compile(r"(?:^|[^0-9a-z])top\s*([1-3])(?:[^0-9a-z]|$)|\btop([1-3])\b", re.IGNORECASE)
EPISODE_ID_RE = re.compile(r"episode[-_ ]?(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class ReplayRecord:
    source_path: Path
    output_path: Path
    rank: int
    turns: int
    players: int
    winner: int
    rewards: List[float]
    statuses: List[str]
    outcome: str
    episode_id: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Kaggle Warorbit replays with rank / turn filters.")
    parser.add_argument("--source-type", choices=("local", "dataset", "competition"), default="local")
    parser.add_argument("--source-root", type=Path, default=ROOT / "replays")
    parser.add_argument("--output-root", type=Path, default=ROOT / "replay_corpus" / "kaggle_top123_2p")
    parser.add_argument("--kaggle-id", type=str, default="", help="Kaggle dataset or competition slug.")
    parser.add_argument(
        "--kaggle-ids",
        type=str,
        default="",
        help="Comma-separated Kaggle dataset or competition slugs to download in sequence.",
    )
    parser.add_argument("--download-root", type=Path, default=ROOT / ".tmp" / "kaggle_downloads")
    parser.add_argument("--top-ranks", type=str, default="1,2,3")
    parser.add_argument(
        "--default-rank",
        type=int,
        default=0,
        help="Use this rank when replay paths do not contain top1/top2/top3. 0 means skip unknown ranks.",
    )
    parser.add_argument("--max-turns", type=int, default=250)
    parser.add_argument("--max-file-mb", type=float, default=45.0)
    parser.add_argument("--max-source-mb", type=float, default=100.0, help="Skip Kaggle sources larger than this size before download.")
    parser.add_argument("--limit", type=int, default=1000, help="Stop after this many accepted replays.")
    parser.add_argument("--allow-draws", action="store_true", help="Keep two-player draws instead of skipping them.")
    parser.add_argument("--no-keep-raw", action="store_true", help="Only write manifests and skip raw replay copies.")
    return parser.parse_args()


def _iter_json_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".json":
            yield path


def _run_kaggle_download(source_type: str, kaggle_id: str, download_root: Path) -> Path:
    if not kaggle_id:
        raise SystemExit("--kaggle-id is required when --source-type is dataset or competition")

    download_root.mkdir(parents=True, exist_ok=True)
    cmd_base = shutil.which("kaggle")
    if cmd_base:
        exe = cmd_base
        base_cmd = [exe]
    else:
        base_cmd = [sys.executable, "-m", "kaggle"]

    if source_type == "dataset":
        cmd = base_cmd + ["datasets", "download", "-d", kaggle_id, "-p", str(download_root), "--unzip"]
    else:
        cmd = base_cmd + ["competitions", "download", "-c", kaggle_id, "-p", str(download_root), "--unzip"]
    subprocess.run(cmd, check=True)

    # Kaggle CLI leaves the downloaded archive behind even after --unzip.
    # Remove it so the project only keeps extracted replay files and manifests.
    for archive in list(download_root.rglob("*.zip")):
        try:
            archive.unlink()
        except FileNotFoundError:
            pass
    return download_root


def _query_remote_source_bytes(source_type: str, kaggle_id: str) -> int | None:
    cmd_base = shutil.which("kaggle")
    if cmd_base:
        base_cmd = [cmd_base]
    else:
        base_cmd = [sys.executable, "-m", "kaggle"]

    if source_type == "dataset":
        cmd = base_cmd + ["datasets", "files", "-v", kaggle_id]
    else:
        cmd = base_cmd + ["competitions", "files", "-v", kaggle_id]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception:
        return None

    total = 0
    try:
        reader = csv.DictReader(result.stdout.splitlines())
        for row in reader:
            raw = (row.get("total_bytes") or row.get("totalBytes") or row.get("bytes") or "").strip()
            if not raw:
                continue
            total += int(float(raw))
    except Exception:
        return None
    return total if total > 0 else None


def _source_size_limit_bytes(args: argparse.Namespace) -> int:
    return int(float(args.max_source_mb) * 1024 * 1024)


def _parse_kaggle_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = []
    if args.kaggle_ids:
        ids.extend([item.strip() for item in str(args.kaggle_ids).split(",") if item.strip()])
    if args.kaggle_id:
        ids.append(str(args.kaggle_id).strip())
    return [item for item in ids if item]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _rank_from_path(path: Path) -> int | None:
    text = " ".join(path.parts)
    match = RANK_RE.search(text)
    if not match:
        return None
    group = match.group(1) or match.group(2)
    return int(group) if group else None


def _episode_id_from_path(path: Path, data: Any) -> str:
    if isinstance(data, dict):
        episode_id = data.get("id")
        if episode_id:
            return str(episode_id)
        info = data.get("info") or {}
        episode = info.get("EpisodeId")
        if episode:
            return str(episode)
    match = EPISODE_ID_RE.search(path.stem)
    if match:
        return match.group(1)
    return path.stem


def _two_player_dict_schema(data: dict[str, Any]) -> tuple[int, int, List[float], List[str], int]:
    steps = data.get("steps") or []
    rewards = data.get("rewards") or []
    statuses = data.get("statuses") or []
    players = len(rewards) if isinstance(rewards, list) and rewards else len(statuses)
    if players <= 0:
        info = data.get("info") or {}
        agents = info.get("Agents") or []
        players = len(agents) if isinstance(agents, list) else 0
    turns = len(steps) if isinstance(steps, list) else 0
    reward_values = [float(x) for x in rewards] if isinstance(rewards, list) else []
    winner = _winner_from_rewards(reward_values)
    return turns, players, reward_values, [str(x) for x in statuses], winner


def _two_player_list_schema(data: list[Any]) -> tuple[int, int, List[float], List[str], int]:
    turns = len(data)
    players = 0
    reward_values: List[float] = []
    statuses: List[str] = []
    if data and isinstance(data[-1], list):
        players = len(data[-1])
        last_turn = data[-1]
        for entry in last_turn:
            if isinstance(entry, dict):
                reward_values.append(float(entry.get("reward", 0.0)))
                statuses.append(str(entry.get("status", "")))
            else:
                reward_values.append(0.0)
                statuses.append("")
    winner = _winner_from_rewards(reward_values)
    return turns, players, reward_values, statuses, winner


def _winner_from_rewards(rewards: Sequence[float]) -> int:
    if len(rewards) != 2:
        return -1
    if abs(rewards[0] - rewards[1]) < 1e-9:
        return -1
    return 0 if rewards[0] > rewards[1] else 1


def _normalize_replay(
    path: Path,
    data: Any,
    rank: int,
    source_root: Path,
    output_root: Path,
    max_turns: int,
) -> ReplayRecord | None:
    if isinstance(data, dict):
        turns, players, rewards, statuses, winner = _two_player_dict_schema(data)
    elif isinstance(data, list):
        turns, players, rewards, statuses, winner = _two_player_list_schema(data)
    else:
        return None

    if rank not in {1, 2, 3}:
        return None
    if players != 2:
        return None
    if turns > max_turns:
        return None

    if len(rewards) < 2:
        rewards = (rewards + [0.0, 0.0])[:2]
    if len(statuses) < 2:
        statuses = (statuses + ["", ""])[:2]

    outcome = "draw" if winner < 0 else ("win" if winner == 0 else "loss")
    episode_id = _episode_id_from_path(path, data)
    output_dir = output_root / f"top{rank}" / path.relative_to(source_root).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}.json"
    return ReplayRecord(
        source_path=path,
        output_path=output_path,
        rank=rank,
        turns=turns,
        players=players,
        winner=winner,
        rewards=list(rewards[:2]),
        statuses=list(statuses[:2]),
        outcome=outcome,
        episode_id=episode_id,
    )


def _serialize_minified(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _candidate_replay_files(source_root: Path) -> Iterator[Path]:
    for path in _iter_json_files(source_root):
        if path.name.startswith("."):
            continue
        yield path


def _summarize_outcome(record: ReplayRecord) -> dict[str, Any]:
    if record.winner < 0:
        outcomes = ["draw", "draw"]
    else:
        outcomes = ["win" if record.winner == 0 else "loss", "win" if record.winner == 1 else "loss"]
    return {
        "episode_id": record.episode_id,
        "turns": record.turns,
        "players": record.players,
        "winner": record.winner,
        "rewards": record.rewards,
        "statuses": record.statuses,
        "outcomes": outcomes,
        "source_rank": record.rank,
        "source_path": str(record.source_path),
        "output_path": str(record.output_path),
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    args = _parse_args()
    ranks = {int(x) for x in args.top_ranks.split(",") if x.strip()}
    max_bytes = int(args.max_file_mb * 1024 * 1024)

    output_root = args.output_root
    if output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "manifest.jsonl"
    skipped_path = output_root / "skipped.jsonl"
    raw_keep = not args.no_keep_raw

    accepted = 0
    skipped = 0
    written_bytes = 0

    source_roots: list[Path] = []
    kaggle_ids = _parse_kaggle_ids(args)

    def scan_root(scan_root: Path) -> None:
        nonlocal accepted, skipped, written_bytes
        source_roots.append(scan_root)
        for path in _candidate_replay_files(scan_root):
            rank = _rank_from_path(path)
            if rank is None and int(args.default_rank) > 0:
                rank = int(args.default_rank)
            if rank is None or rank not in ranks:
                skipped += 1
                _append_jsonl(skipped_path, {"path": str(path), "reason": "rank_not_top123"})
                continue

            try:
                data = _load_json(path)
            except Exception as exc:
                skipped += 1
                _append_jsonl(skipped_path, {"path": str(path), "reason": "json_error", "error": str(exc)})
                continue

            record = _normalize_replay(path, data, rank, scan_root, output_root, int(args.max_turns))
            if record is None:
                skipped += 1
                _append_jsonl(skipped_path, {"path": str(path), "reason": "schema_or_filter_mismatch"})
                continue

            if record.winner < 0 and not bool(args.allow_draws):
                skipped += 1
                _append_jsonl(skipped_path, {"path": str(path), "reason": "draw_skipped"})
                continue

            minified = _serialize_minified(data)
            if len(minified) > max_bytes:
                skipped += 1
                _append_jsonl(
                    skipped_path,
                    {
                        "path": str(path),
                        "reason": "oversize_after_minify",
                        "size_bytes": len(minified),
                        "limit_bytes": max_bytes,
                    },
                )
                continue

            if raw_keep:
                record.output_path.parent.mkdir(parents=True, exist_ok=True)
                record.output_path.write_bytes(minified)
                written_bytes += len(minified)

            row = _summarize_outcome(record)
            row["kept_raw"] = raw_keep
            row["file_bytes"] = len(minified)
            row["players_kept"] = 2
            row["source_dataset_root"] = str(scan_root)
            _append_jsonl(manifest_path, row)
            accepted += 1

            if accepted % 50 == 0:
                print(f"accepted={accepted} skipped={skipped} latest={path.name}")
            if accepted >= int(args.limit):
                return

    def cleanup_root(scan_root: Path) -> None:
        if args.source_type not in {"dataset", "competition"}:
            return
        try:
            resolved = scan_root.resolve()
            download_resolved = args.download_root.resolve()
        except Exception:
            return
        if str(resolved).startswith(str(download_resolved)):
            shutil.rmtree(scan_root, ignore_errors=True)

    if args.source_type in {"dataset", "competition"}:
        if kaggle_ids:
            for kaggle_id in kaggle_ids:
                limit_bytes = _source_size_limit_bytes(args)
                remote_bytes = _query_remote_source_bytes(args.source_type, kaggle_id)
                if remote_bytes is None:
                    skipped += 1
                    _append_jsonl(
                        skipped_path,
                        {
                            "path": kaggle_id,
                            "reason": "remote_source_size_unknown",
                            "limit_bytes": limit_bytes,
                        },
                    )
                    print(f"skip source={kaggle_id} remote_bytes=unknown limit_bytes={limit_bytes}")
                    continue
                if remote_bytes > limit_bytes:
                    skipped += 1
                    _append_jsonl(
                        skipped_path,
                        {
                            "path": kaggle_id,
                            "reason": "remote_source_too_large",
                            "size_bytes": remote_bytes,
                            "limit_bytes": limit_bytes,
                        },
                    )
                    print(
                        f"skip source={kaggle_id} remote_bytes={remote_bytes} "
                        f"limit_bytes={limit_bytes}"
                    )
                    continue
                sub_root = args.download_root / re.sub(r"[^A-Za-z0-9_.-]+", "_", kaggle_id)
                downloaded_root = _run_kaggle_download(args.source_type, kaggle_id, sub_root)
                scan_root(downloaded_root)
                cleanup_root(downloaded_root)
                if accepted >= int(args.limit):
                    break
        else:
            limit_bytes = _source_size_limit_bytes(args)
            remote_bytes = _query_remote_source_bytes(args.source_type, args.kaggle_id.strip())
            if remote_bytes is None:
                raise SystemExit(f"remote source size unknown for {args.kaggle_id.strip()} (limit {limit_bytes} bytes)")
            if remote_bytes > limit_bytes:
                raise SystemExit(
                    f"remote source too large: {remote_bytes} bytes > {limit_bytes} bytes"
                )
            downloaded_root = _run_kaggle_download(args.source_type, args.kaggle_id.strip(), args.download_root)
            scan_root(downloaded_root)
            cleanup_root(downloaded_root)
    else:
        scan_root(args.source_root)

    report = {
        "source_type": args.source_type,
        "scan_root": str(source_roots[0] if source_roots else args.source_root),
        "output_root": str(output_root),
        "accepted": accepted,
        "skipped": skipped,
        "raw_bytes_written": written_bytes,
        "max_turns": int(args.max_turns),
        "max_file_mb": float(args.max_file_mb),
        "limit": int(args.limit),
        "ranks": sorted(ranks),
        "keep_raw": raw_keep,
        "kaggle_ids": kaggle_ids,
        "source_roots": [str(root) for root in source_roots],
    }
    (output_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
