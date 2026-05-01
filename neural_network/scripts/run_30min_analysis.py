from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_network.src.utils import load_json, ensure_dir
from neural_network.src.encoder import encode_game_state
from neural_network.src.model import NeuralNetworkModel, ModelConfig
from neural_network.src.policy import choose_action, reconstruct_action, build_action_candidates, is_valid_action
from neural_network.src.reward import compute_reward
from neural_network.src.self_play import make_synthetic_game, play_episode
from neural_network.src.orbit_wars_adapter import obs_to_game_dict, action_to_kaggle_list
from neural_network.src.storage import save_checkpoint, append_jsonl
from neural_network.src.benchmark import benchmark_model
from neural_network.src.trainer import _infer_input_dim
from neural_network.src.diagnostics import diagnose_run


def _clone_game(game: Dict[str, Any]) -> Dict[str, Any]:
    clone = dict(game)
    clone["planets"] = [dict(p) for p in game.get("planets", [])]
    clone["fleets"] = [dict(f) for f in game.get("fleets", [])]
    clone["player_ids"] = list(game.get("player_ids", []))
    return clone


def _simulate_step(game: Dict[str, Any], action_tuple: tuple[int, int, int], turn: int) -> Dict[str, Any]:
    next_game = _clone_game(game)
    next_game["turn"] = turn + 1
    if action_tuple[0] >= 0:
        src = next(p for p in next_game["planets"] if p["id"] == action_tuple[0])
        tgt = next(p for p in next_game["planets"] if p["id"] == action_tuple[1])
        moved = min(action_tuple[2], max(0, int(src["ships"]) - 1))
        src["ships"] = max(0.0, src["ships"] - moved)
        if tgt["owner"] in (-1, game["my_id"]):
            tgt["ships"] += 0.65 * moved
            if tgt["ships"] > 60 and tgt["owner"] != game["my_id"]:
                tgt["owner"] = game["my_id"]
        else:
            tgt["ships"] -= 0.55 * moved
            if tgt["ships"] <= 0:
                tgt["owner"] = game["my_id"]
                tgt["ships"] = max(5.0, abs(tgt["ships"]) + 3.0)
    return next_game


def _episode_stats(prev_game: Dict[str, Any], next_game: Dict[str, Any], action_tuple: tuple[int, int, int], invalid_filtered: int) -> Dict[str, float]:
    my_id = prev_game["my_id"]
    prev_planets = prev_game["planets"]
    next_planets = next_game["planets"]
    prev_my = [p for p in prev_planets if p["owner"] == my_id]
    next_my = [p for p in next_planets if p["owner"] == my_id]
    prev_enemy = [p for p in prev_planets if p["owner"] not in (-1, my_id)]
    next_enemy = [p for p in next_planets if p["owner"] not in (-1, my_id)]
    sent = float(action_tuple[2]) if action_tuple[0] >= 0 else 0.0
    captured = float(len([p for p in next_my if p["id"] not in {x["id"] for x in prev_my}]))
    lost = float(len([p for p in prev_my if p["id"] not in {x["id"] for x in next_my}]))
    return {
        "planets_captured": captured,
        "planets_lost": lost,
        "ships_sent": sent,
        "ships_lost": max(0.0, sent - 0.65 * sent),
        "attack_efficiency": captured / max(1.0, sent),
        "defense_efficiency": max(0.0, 1.0 - lost),
        "filtered_actions": float(invalid_filtered),
        "my_planets": float(len(next_my)),
        "enemy_planets": float(len(next_enemy)),
        "production": float(sum(p["production"] for p in next_my)),
    }


def _phase_for_turn(turn: int) -> str:
    if turn < 120:
        return "early"
    if turn < 280:
        return "mid"
    return "late"


def _run_episode(model: NeuralNetworkModel, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    game = make_synthetic_game(seed)
    episode = []
    metrics = {
        "reward_sum": 0.0,
        "reward_min": float("inf"),
        "reward_max": float("-inf"),
        "actions": 0,
        "invalid_actions": 0,
        "filtered_actions": 0,
        "ships_sent": 0.0,
        "ships_lost": 0.0,
        "planets_captured": 0.0,
        "planets_lost": 0.0,
        "attack_efficiency": 0.0,
        "defense_efficiency": 0.0,
        "early_reward": 0.0,
        "mid_reward": 0.0,
        "late_reward": 0.0,
    }
    for turn in range(12):
        prev_game = _clone_game(game)
        encoded = encode_game_state(game, config)
        outputs = model.forward(encoded.features)
        candidates = build_action_candidates(game)
        valid_flags = [is_valid_action(c, game) for c in candidates]
        cand = choose_action(outputs, game, explore=False)
        action_tuple = reconstruct_action(cand, game)
        if not is_valid_action(cand, game):
            metrics["invalid_actions"] += 1
        metrics["filtered_actions"] += float(len(candidates) - sum(valid_flags))
        next_game = _simulate_step(game, action_tuple, turn)
        reward = compute_reward(game, {"ships": action_tuple[2]}, next_game, terminal=(turn == 11))
        stats = _episode_stats(prev_game, next_game, action_tuple, len(candidates) - sum(valid_flags))
        metrics["reward_sum"] += float(reward)
        metrics["reward_min"] = min(metrics["reward_min"], float(reward))
        metrics["reward_max"] = max(metrics["reward_max"], float(reward))
        metrics["actions"] += 1
        metrics["ships_sent"] += stats["ships_sent"]
        metrics["ships_lost"] += stats["ships_lost"]
        metrics["planets_captured"] += stats["planets_captured"]
        metrics["planets_lost"] += stats["planets_lost"]
        metrics["attack_efficiency"] += stats["attack_efficiency"]
        metrics["defense_efficiency"] += stats["defense_efficiency"]
        phase = _phase_for_turn(turn)
        metrics[f"{phase}_reward"] += float(reward)
        episode.append({"turn": turn, "reward": float(reward), "action": action_tuple})
        game = next_game
    metrics["episode_len"] = len(episode)
    metrics["final_win"] = 1.0 if metrics["reward_sum"] > 0 else 0.0
    metrics["avg_reward"] = metrics["reward_sum"] / max(1, metrics["actions"])
    metrics["avg_action_count"] = metrics["actions"]
    metrics["avg_filtered_actions"] = metrics["filtered_actions"] / max(1, metrics["actions"])
    metrics["avg_ships_sent"] = metrics["ships_sent"] / max(1, metrics["actions"])
    metrics["avg_ships_lost"] = metrics["ships_lost"] / max(1, metrics["actions"])
    metrics["attack_efficiency"] /= max(1, metrics["actions"])
    metrics["defense_efficiency"] /= max(1, metrics["actions"])
    if metrics["reward_min"] == float("inf"):
        metrics["reward_min"] = 0.0
    if metrics["reward_max"] == float("-inf"):
        metrics["reward_max"] = 0.0
    return {"metrics": metrics, "episode": episode}


def _run_real_episode(model: NeuralNetworkModel, baseline_agent, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    from kaggle_environments import make
    import bot_v7

    env = make("orbit_wars")

    policy_mode = config.get("policy_mode", "model_first")
    temperature = float(config.get("temperature", 1.0))
    min_ratio = float(config.get("min_ratio", 0.1))
    explore = bool(config.get("explore", False))

    model_action_count = 0
    fallback_action_count = 0
    invalid_action_count = 0
    empty_action_count = 0

    def our_agent(obs, _config=None):
        nonlocal model_action_count, fallback_action_count, invalid_action_count, empty_action_count
        game = obs_to_game_dict(obs)
        encoded = encode_game_state(game, config)
        outputs = model.forward(encoded.features)
        cand = choose_action(outputs, game, temperature=temperature, explore=explore)
        if not is_valid_action(cand, game):
            invalid_action_count += 1
        action = reconstruct_action(cand, game, min_ratio=min_ratio)
        if action[0] < 0:
            empty_action_count += 1
        model_action_count += 1
        if policy_mode == "baseline_first" and (not is_valid_action(cand, game) or action[2] <= 0):
            fallback_action_count += 1
            baseline_action = baseline_agent(obs, _config) or []
            return baseline_action
        return action_to_kaggle_list(action)

    def opponent_agent(obs, _config=None):
        return baseline_agent(obs, _config)

    steps = env.run([our_agent, opponent_agent])
    metrics = {
        "reward_sum": 0.0,
        "reward_min": float("inf"),
        "reward_max": float("-inf"),
        "actions": 0,
        "invalid_actions": 0,
        "filtered_actions": 0,
        "ships_sent": 0.0,
        "ships_lost": 0.0,
        "planets_captured": 0.0,
        "planets_lost": 0.0,
        "attack_efficiency": 0.0,
        "defense_efficiency": 0.0,
        "early_reward": 0.0,
        "mid_reward": 0.0,
        "late_reward": 0.0,
        "model_action_count": float(model_action_count),
        "fallback_action_count": float(fallback_action_count),
        "invalid_action_count": float(invalid_action_count),
        "empty_action_count": float(empty_action_count),
        "policy_mode": policy_mode,
    }
    prev_game = None
    for step_idx, step_pair in enumerate(steps):
        if not step_pair:
            continue
        me = step_pair[0]
        obs = getattr(me, "observation", None)
        if obs is None:
            continue
        game = obs_to_game_dict(obs)
        if prev_game is None:
            prev_game = game
            continue
        action = getattr(me, "action", []) or []
        if action:
            ships = int(action[0][2]) if isinstance(action[0], (list, tuple)) and len(action[0]) >= 3 else 0
            metrics["actions"] += 1
            metrics["ships_sent"] += ships
        reward = float(getattr(me, "reward", 0.0) or 0.0)
        metrics["reward_sum"] += reward
        metrics["reward_min"] = min(metrics["reward_min"], reward)
        metrics["reward_max"] = max(metrics["reward_max"], reward)
        phase = _phase_for_turn(int(game.get("turn", 0)))
        metrics[f"{phase}_reward"] += reward
        prev_game = game
    metrics["episode_len"] = len(steps)
    metrics["final_win"] = 1.0 if metrics["reward_sum"] > 0 else 0.0
    metrics["avg_reward"] = metrics["reward_sum"] / max(1, metrics["actions"])
    metrics["avg_action_count"] = metrics["actions"]
    metrics["avg_filtered_actions"] = 0.0
    metrics["avg_ships_sent"] = metrics["ships_sent"] / max(1, metrics["actions"])
    metrics["avg_ships_lost"] = 0.0
    metrics["attack_efficiency"] = 0.0
    metrics["defense_efficiency"] = 0.0
    metrics["invalid_action_rate"] = metrics["invalid_action_count"] / max(1, metrics["model_action_count"])
    metrics["empty_action_rate"] = metrics["empty_action_count"] / max(1, metrics["model_action_count"])
    metrics["fallback_action_rate"] = metrics["fallback_action_count"] / max(1, metrics["model_action_count"])
    if metrics["reward_min"] == float("inf"):
        metrics["reward_min"] = 0.0
    if metrics["reward_max"] == float("-inf"):
        metrics["reward_max"] = 0.0
    return {"metrics": metrics, "episode": []}


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize(rows: List[Dict[str, Any]], best_metrics: Dict[str, Any], elapsed_sec: float) -> str:
    avg = lambda k: float(np.mean([r.get(k, 0.0) for r in rows])) if rows else 0.0
    return "\n".join([
        "# 30 min analysis report",
        "",
        f"- episodes: {len(rows)}",
        f"- elapsed_min: {elapsed_sec / 60.0:.2f}",
        f"- avg_reward: {avg('avg_reward'):.4f}",
        f"- reward_min: {min((r.get('reward_min', 0.0) for r in rows), default=0.0):.4f}",
        f"- reward_max: {max((r.get('reward_max', 0.0) for r in rows), default=0.0):.4f}",
        f"- winrate: {avg('final_win'):.3f}",
        f"- invalid_action_rate: {avg('invalid_actions') / max(1.0, avg('actions')):.4f}",
        f"- avg_filtered_actions: {avg('avg_filtered_actions'):.4f}",
        f"- avg_ships_sent: {avg('avg_ships_sent'):.2f}",
        f"- avg_ships_lost: {avg('avg_ships_lost'):.2f}",
        f"- avg_planets_captured: {avg('planets_captured'):.2f}",
        f"- avg_planets_lost: {avg('planets_lost'):.2f}",
        f"- attack_efficiency: {avg('attack_efficiency'):.4f}",
        f"- defense_efficiency: {avg('defense_efficiency'):.4f}",
        f"- baseline_avg_reward: {avg('baseline_avg_reward'):.4f}",
        f"- baseline_winrate: {avg('baseline_winrate'):.3f}",
        f"- best_checkpoint: {best_metrics.get('checkpoint_path', '')}",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    parser.add_argument("--config", default="neural_network/configs/default_config.json")
    parser.add_argument("--auto-correct", action="store_true")
    parser.add_argument("--auto-correct-every-minutes", type=float, default=5.0)
    args = parser.parse_args()

    cfg = load_json(args.config)
    out_dir = ensure_dir("neural_network/logs/analysis_30min")
    ensure_dir(cfg["checkpoint_dir"])

    config_used_path = out_dir / "config_used.json"
    config_used_path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")

    model = NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(cfg)))
    baseline_model = NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(cfg)), rng=np.random.default_rng(int(cfg["seed"]) + 1))
    start = time.monotonic()
    deadline = start + args.duration_minutes * 60.0
    episode_rows: List[Dict[str, Any]] = []
    jsonl_path = out_dir / "metrics.jsonl"
    best_score = -1e9
    best_checkpoint = out_dir / "best_checkpoint.pt"
    checkpoint_count = 0
    episode_index = 0
    baseline_metrics = benchmark_model(baseline_model, cfg, games=max(1, int(cfg["benchmark_games"]) // 2))
    last_heartbeat = start
    last_autocorrect = start

    print(
        {
            "status": "started",
            "duration_minutes": args.duration_minutes,
            "out_dir": str(out_dir),
            "deadline_utc_like": round(deadline, 3),
        },
        flush=True,
    )

    use_real_env = True
    try:
        from kaggle_environments import make  # noqa: F401
        import bot_v7  # noqa: F401
    except Exception:
        use_real_env = False

    while time.monotonic() < deadline:
        if use_real_env:
            import bot_v7
            result = _run_real_episode(model, bot_v7.agent, cfg, seed=episode_index)
        else:
            result = _run_episode(model, cfg, seed=episode_index)
        metrics = result["metrics"]
        metrics["episode_index"] = episode_index
        metrics["wall_time_sec"] = round(time.monotonic() - start, 3)
        sample_game = make_synthetic_game(episode_index)
        sample_encoded = encode_game_state(sample_game, cfg)
        sample_out = model.forward(sample_encoded.features)
        metrics["policy_mean"] = float(np.mean(sample_out["policy_logits"]))
        metrics["value_mean"] = float(np.mean(sample_out["value"]))
        metrics["learning_loss"] = float(abs(metrics["avg_reward"]))
        metrics["model_version"] = int(checkpoint_count)
        metrics["baseline_avg_reward"] = float(baseline_metrics["avg_reward"])
        metrics["baseline_winrate"] = float(baseline_metrics["winrate"])
        append_jsonl(jsonl_path, metrics)
        episode_rows.append(metrics)

        if metrics["avg_reward"] > best_score:
            best_score = metrics["avg_reward"]
            save_checkpoint(best_checkpoint, model.state_dict(), {"best_score": best_score, "episode_index": episode_index})
            checkpoint_count += 1

        if episode_index % 5 == 0:
            latest_path = Path(cfg["latest_checkpoint"])
            save_checkpoint(latest_path, model.state_dict(), {"episode_index": episode_index, "avg_reward": metrics["avg_reward"]})
            checkpoint_count += 1

        episode_index += 1

        if args.auto_correct and (time.monotonic() - last_autocorrect) >= args.auto_correct_every_minutes * 60.0:
            recent_rows = episode_rows[-max(10, int(len(episode_rows) * 0.25)) :]
            diag = diagnose_run(recent_rows)
            cfg = dict(cfg)
            cfg["policy_mode"] = diag.suggested_fix.get("policy_mode", cfg.get("policy_mode", "model_first"))
            cfg["temperature"] = diag.suggested_fix.get("temperature", cfg.get("temperature", 1.0))
            cfg["min_ratio"] = diag.suggested_fix.get("min_ratio", cfg.get("min_ratio", 0.1))
            cfg["explore"] = diag.suggested_fix.get("explore", cfg.get("explore", False))
            append_jsonl(
                jsonl_path,
                {
                    "type": "autocorrect",
                    "episode_index": episode_index,
                    "mode": diag.mode,
                    "severity": diag.severity,
                    "policy_mode": cfg["policy_mode"],
                    "temperature": cfg["temperature"],
                    "min_ratio": cfg["min_ratio"],
                    "explore": cfg["explore"],
                },
            )
            print(
                {
                    "status": "autocorrect",
                    "episode_index": episode_index,
                    "mode": diag.mode,
                    "severity": diag.severity,
                    "policy_mode": cfg["policy_mode"],
                    "temperature": cfg["temperature"],
                    "min_ratio": cfg["min_ratio"],
                    "explore": cfg["explore"],
                },
                flush=True,
            )
            last_autocorrect = time.monotonic()

        now = time.monotonic()
        if now - last_heartbeat >= 60.0:
            print(
                {
                    "status": "running",
                    "episodes": len(episode_rows),
                    "elapsed_min": round((now - start) / 60.0, 2),
                    "avg_reward": round(float(np.mean([r["avg_reward"] for r in episode_rows])), 4) if episode_rows else 0.0,
                    "best_score": round(best_score, 4),
                },
                flush=True,
            )
            last_heartbeat = now

    elapsed_sec = time.monotonic() - start
    summary_csv = out_dir / "summary.csv"
    _write_csv(summary_csv, episode_rows)
    report_md = out_dir / "report.md"
    report_md.write_text(_summarize(episode_rows, {"checkpoint_path": str(best_checkpoint)}, elapsed_sec), encoding="utf-8")

    print({
        "episodes": len(episode_rows),
        "elapsed_min": round(elapsed_sec / 60.0, 2),
        "out_dir": str(out_dir),
        "best_checkpoint": str(best_checkpoint),
        "metrics_jsonl": str(jsonl_path),
        "summary_csv": str(summary_csv),
        "report_md": str(report_md),
    }, flush=True)


if __name__ == "__main__":
    main()
