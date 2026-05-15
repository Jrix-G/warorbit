from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "kaggle_submission_stage") not in sys.path:
    sys.path.insert(0, str(ROOT / "kaggle_submission_stage"))

from neural_network.src.encoder import encode_game_state  # noqa: E402
from neural_network.src.policy import build_action_candidates  # noqa: E402
from neural_network.src.trajectory import safe_plan_shot  # noqa: E402


ENCODER_CONFIG: dict[str, Any] = {
    "max_planets": 64,
    "max_fleets": 128,
    "max_players": 4,
    "board_scale": 100.0,
    "ship_scale": 2000.0,
    "production_scale": 10.0,
    "radius_scale": 10.0,
    "horizon_scale": 100.0,
    "planet_id_scale": 100.0,
}


def _angle_diff(a: float, b: float) -> float:
    return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)


def _planet(row: list[Any]) -> dict[str, Any] | None:
    if not isinstance(row, list) or len(row) < 7:
        return None
    return {
        "id": int(row[0]),
        "owner": int(row[1]),
        "x": float(row[2]),
        "y": float(row[3]),
        "radius": float(row[4]),
        "ships": float(row[5]),
        "production": float(row[6]),
    }


def _fleet(row: list[Any]) -> dict[str, Any] | None:
    if not isinstance(row, list) or len(row) < 7:
        return None
    return {
        "id": int(row[0]),
        "owner": int(row[1]),
        "x": float(row[2]),
        "y": float(row[3]),
        "angle": float(row[4]),
        "source_id": int(row[5]),
        "target_id": -1,
        "ships": float(row[6]),
        "eta": 0.0,
    }


def _game_from_observation(obs: dict[str, Any], player_id: int) -> dict[str, Any] | None:
    planets = [_planet(p) for p in obs.get("planets", [])]
    planets = [p for p in planets if p is not None]
    if not planets:
        return None
    initial_planets = [_planet(p) for p in obs.get("initial_planets", obs.get("planets", []))]
    fleets = [_fleet(f) for f in obs.get("fleets", [])]
    return {
        "turn": int(obs.get("step", 0) or 0),
        "my_id": int(player_id),
        "player_ids": [0, 1, 2, 3],
        "is_four_player": True,
        "angular_velocity": float(obs.get("angular_velocity", 0.0) or 0.0),
        "planets": planets,
        "initial_planets": [p for p in initial_planets if p is not None],
        "fleets": [f for f in fleets if f is not None],
    }


def _load_manifest(source_root: Path) -> dict[str, dict[str, Any]]:
    matches = list(source_root.rglob("manifest.csv"))
    if not matches:
        return {}
    manifest: dict[str, dict[str, Any]] = {}
    with matches[0].open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            episode_id = str(row.get("episode_id", "")).strip()
            if episode_id:
                manifest[episode_id] = row
    return manifest


def _iter_replays(source_root: Path) -> list[Path]:
    return sorted(p for p in source_root.rglob("*.json") if p.is_file())


def _winners(rewards: Any) -> set[int]:
    if not isinstance(rewards, list) or not rewards:
        return set()
    vals = [float(r) if r is not None else -1e9 for r in rewards]
    best = max(vals)
    return {i for i, v in enumerate(vals) if v == best}


def _best_move(action: Any) -> list[Any] | None:
    if not isinstance(action, list) or not action:
        return None
    moves = [m for m in action if isinstance(m, list) and len(m) >= 3 and float(m[2]) > 0]
    if not moves:
        return None
    return max(moves, key=lambda m: float(m[2]))


def _candidate_label(game: dict[str, Any], candidates: list[Any], move: list[Any], max_angle_error: float) -> tuple[int, float, str] | None:
    src_id = int(move[0])
    action_angle = float(move[1])
    action_amount = max(1.0, float(move[2]))
    planets = {int(p["id"]): p for p in game["planets"]}
    src = planets.get(src_id)
    if src is None or int(src["owner"]) != int(game["my_id"]):
        return None

    best: tuple[float, float, int, str] | None = None
    angle_cache: dict[int, float | None] = {}
    for idx, candidate in enumerate(candidates):
        if candidate.mission == "do_nothing" or int(candidate.source_id) != src_id:
            continue
        tgt = planets.get(int(candidate.target_id))
        if tgt is None:
            continue
        target_id = int(candidate.target_id)
        if target_id not in angle_cache:
            angle_cache[target_id] = safe_plan_shot(src, tgt, game)
        candidate_angle = angle_cache[target_id]
        if candidate_angle is None:
            continue
        angle_error = _angle_diff(action_angle, candidate_angle)
        if angle_error > max_angle_error:
            continue
        amount_error = abs(float(candidate.amount) - action_amount) / max(1.0, float(src.get("ships", 1.0)))
        score = angle_error + 0.35 * amount_error
        if best is None or score < best[0]:
            best = (score, angle_error, idx, candidate.mission)
    if best is None:
        return None
    return best[2], best[1], best[3]


def _trim_candidates(candidates: list[Any], label_idx: int, max_candidates: int) -> tuple[list[Any], int]:
    if len(candidates) <= max_candidates:
        return candidates, label_idx
    label = candidates[label_idx]
    selected: list[int] = []
    for idx in (0, label_idx):
        if 0 <= idx < len(candidates) and idx not in selected:
            selected.append(idx)
    if getattr(label, "source_id", -1) >= 0:
        for idx, candidate in enumerate(candidates):
            if idx in selected:
                continue
            if int(getattr(candidate, "source_id", -999)) == int(label.source_id):
                selected.append(idx)
                if len(selected) >= max_candidates:
                    break
    if len(selected) < max_candidates:
        for idx in range(len(candidates)):
            if idx not in selected:
                selected.append(idx)
                if len(selected) >= max_candidates:
                    break
    remap = {old_idx: new_idx for new_idx, old_idx in enumerate(selected)}
    return [candidates[idx] for idx in selected], int(remap[label_idx])


def _flush_shard(
    out_dir: Path,
    shard_idx: int,
    states: list[np.ndarray],
    candidates: list[np.ndarray],
    masks: list[np.ndarray],
    labels: list[int],
    weights: list[float],
) -> Path:
    path = out_dir / f"shard_{shard_idx:05d}.npz"
    np.savez_compressed(
        path,
        states=np.stack(states).astype(np.float32),
        candidates=np.stack(candidates).astype(np.float32),
        masks=np.stack(masks).astype(np.bool_),
        labels=np.asarray(labels, dtype=np.int64),
        weights=np.asarray(weights, dtype=np.float32),
    )
    return path


def extract(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    source_root = Path(args.source_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(source_root)
    replay_paths = _iter_replays(source_root)
    if args.max_episodes:
        replay_paths = replay_paths[: int(args.max_episodes)]

    state_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    label_rows: list[int] = []
    weight_rows: list[float] = []
    meta_f = (out_dir / "samples_meta.jsonl").open("w", encoding="utf-8")
    skip = Counter()
    missions = Counter()
    per_episode = defaultdict(int)
    shard_idx = 0
    total_samples = 0

    for path_idx, path in enumerate(replay_paths, start=1):
        if args.max_samples and total_samples >= args.max_samples:
            break
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            skip["json_error"] += 1
            continue
        steps = replay.get("steps")
        if not isinstance(steps, list) or len(steps) < 2:
            skip["bad_steps"] += 1
            continue
        if len(steps[0]) != 4:
            skip["not_4p"] += 1
            continue
        if len(steps) - 1 > int(args.max_turns):
            skip["too_many_turns"] += 1
            continue
        winners = _winners(replay.get("rewards"))
        if not winners:
            skip["no_winner"] += 1
            continue
        episode_id = str(replay.get("id") or path.stem)

        for step_idx in range(1, len(steps)):
            if args.max_samples and total_samples >= args.max_samples:
                break
            for player_id in range(4):
                if not args.include_non_winners and player_id not in winners:
                    continue
                action = steps[step_idx][player_id].get("action")
                move = _best_move(action)
                if move is None:
                    if float(args.noop_keep_rate) <= 0.0 or random.random() > float(args.noop_keep_rate):
                        continue
                    label_idx = 0
                    angle_error = 0.0
                    mission = "do_nothing"
                else:
                    prev_obs = steps[step_idx - 1][player_id].get("observation") or {}
                    game = _game_from_observation(prev_obs, player_id)
                    if game is None:
                        skip["bad_observation"] += 1
                        continue
                    candidates = build_action_candidates(game)
                    label = _candidate_label(game, candidates, move, float(args.max_angle_error))
                    if label is None:
                        skip["unmatched_action"] += 1
                        continue
                    label_idx, angle_error, mission = label

                prev_obs = steps[step_idx - 1][player_id].get("observation") or {}
                game = _game_from_observation(prev_obs, player_id)
                if game is None:
                    skip["bad_observation"] += 1
                    continue
                if move is None:
                    candidates = build_action_candidates(game)
                if label_idx >= len(candidates):
                    skip["candidate_count"] += 1
                    continue
                candidates, label_idx = _trim_candidates(candidates, label_idx, int(args.max_candidates))

                encoded = encode_game_state(game, ENCODER_CONFIG).features
                cand = np.zeros((int(args.max_candidates), 16), dtype=np.float32)
                mask = np.zeros((int(args.max_candidates),), dtype=np.bool_)
                cand_features = np.stack([c.score_features for c in candidates]).astype(np.float32)
                cand[: len(candidates)] = cand_features
                mask[: len(candidates)] = True

                state_rows.append(encoded)
                candidate_rows.append(cand)
                mask_rows.append(mask)
                label_rows.append(int(label_idx))
                weight_rows.append(1.0 if mission != "do_nothing" else float(args.noop_weight))
                missions[mission] += 1
                per_episode[episode_id] += 1
                total_samples += 1
                meta_f.write(json.dumps({
                    "episode_id": episode_id,
                    "source": str(path),
                    "step": step_idx - 1,
                    "player": player_id,
                    "winner": player_id in winners,
                    "mission": mission,
                    "label": int(label_idx),
                    "angle_error": float(angle_error),
                    "manifest": manifest.get(episode_id, {}),
                }) + "\n")

                if len(state_rows) >= int(args.shard_size):
                    _flush_shard(out_dir, shard_idx, state_rows, candidate_rows, mask_rows, label_rows, weight_rows)
                    shard_idx += 1
                    state_rows.clear()
                    candidate_rows.clear()
                    mask_rows.clear()
                    label_rows.clear()
                    weight_rows.clear()

        if path_idx % int(args.log_every) == 0:
            print(f"scan episodes={path_idx}/{len(replay_paths)} samples={total_samples} skipped={dict(skip)} missions={dict(missions)}", flush=True)

    if state_rows:
        _flush_shard(out_dir, shard_idx, state_rows, candidate_rows, mask_rows, label_rows, weight_rows)
        shard_idx += 1
    meta_f.close()

    report = {
        "source_root": str(source_root),
        "output_dir": str(out_dir),
        "episodes_scanned": len(replay_paths),
        "episodes_with_samples": len(per_episode),
        "samples": total_samples,
        "shards": shard_idx,
        "missions": dict(missions),
        "skipped": dict(skip),
        "encoder_config": ENCODER_CONFIG,
        "args": vars(args),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract compact behavior-cloning shards from Orbit Wars 4p Kaggle replays.")
    parser.add_argument("--source-root", default=r"D:\warorbit_kaggle_raw")
    parser.add_argument("--output-dir", default=str(ROOT / "replay_corpus" / "imitation_4p_top10_v1"))
    parser.add_argument("--max-turns", type=int, default=250)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=250_000)
    parser.add_argument("--max-candidates", type=int, default=2048)
    parser.add_argument("--max-angle-error", type=float, default=0.28)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--noop-keep-rate", type=float, default=0.0)
    parser.add_argument("--noop-weight", type=float, default=0.10)
    parser.add_argument("--include-non-winners", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    extract(parse_args())
