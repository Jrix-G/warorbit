from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PACKAGE_DIR = Path(__file__).resolve().parents[1]

from neural_network.src.model import ModelConfig, NeuralNetworkModel, count_parameters, load_compatible_state_dict
from neural_network.src.notebook_4p_training import (
    _action_summary,
    _agent_for_name,
    _infer_input_dim,
    _make_model_agent,
    _make_our_agent,
    _sample_opponents,
    _train_episode,
    run_match,
)
from neural_network.src.population_4p_training import (
    _activity_shaping_reward,
    _composite_score,
    _score_record,
    _strategic_dense_reward,
    configure_run_logging,
)
from neural_network.src.storage import append_jsonl, load_checkpoint, save_checkpoint
from neural_network.src.torch_compat import ensure_torch_dynamo_stub
from neural_network.src.utils import ensure_dir, load_json, save_json

MAX_DURATION_MINUTES = 600.0


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(path: Path, message: str) -> None:
    line = f"[{_now()}] {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _load_config(path: str | None) -> dict:
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.append(PACKAGE_DIR / "configs" / "default_config.json")
    candidates.append(ROOT / "neural_network" / "configs" / "default_config.json")
    for candidate in candidates:
        if candidate.exists():
            return load_json(str(candidate))
    raise FileNotFoundError(f"Config not found. Tried: {[str(candidate) for candidate in candidates]}")


def _run_tag() -> str:
    return datetime.now(timezone.utc).strftime("run_10h_2p4p_%Y%m%d_%H%M%S")


def _state_to_cpu(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in state.items()}


def _load_model(config: Dict[str, Any], checkpoint_path: str | None) -> NeuralNetworkModel:
    ensure_torch_dynamo_stub()
    model = NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(config), hidden_dim=int(config.get("hidden_dim", 320))))
    if checkpoint_path and Path(checkpoint_path).exists():
        state, _ = load_checkpoint(checkpoint_path)
        load_compatible_state_dict(model, state)
    return model


def _build_agents_n(
    model: NeuralNetworkModel,
    config: Dict[str, Any],
    seed: int,
    our_index: int,
    n_players: int,
    temperature: float,
    pool: Sequence[str],
    explore: bool,
):
    log_probs: List[torch.Tensor] = []
    action_records: List[Dict[str, Any]] = []
    opp_names = _sample_opponents(pool, seed, max(1, n_players - 1))
    opp_iter = iter(opp_names)
    agents = []
    our_agent = _make_our_agent(model, config, log_probs, action_records, temperature, explore=explore)
    for slot in range(n_players):
        if slot == our_index:
            agents.append(our_agent)
            continue
        name = next(opp_iter, None) or "random"
        if name == "self":
            agents.append(_make_model_agent(model, config, temperature, explore=False))
        else:
            agents.append(_agent_for_name(name))
    return agents, log_probs, action_records, opp_names


def _rank_from_scores(scores: Sequence[float], our_index: int, default: int) -> int:
    ordered = sorted(((float(score), idx) for idx, score in enumerate(scores)), reverse=True)
    return next((rank for rank, (_, idx) in enumerate(ordered, start=1) if idx == our_index), default)


def _play_train_episode(
    model: NeuralNetworkModel,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    *,
    stage: str,
    n_players: int,
    pool: Sequence[str],
    seed: int,
    our_index: int,
    generation: int,
    local_step: int,
    progress: float,
    baseline: float,
) -> tuple[float, Dict[str, Any]]:
    temperature = float(config.get("temperature_start", 1.05)) + (
        float(config.get("temperature_end", 0.18)) - float(config.get("temperature_start", 1.05))
    ) * min(1.0, max(0.0, progress))
    agents, log_probs, action_records, opp_names = _build_agents_n(
        model,
        config,
        seed,
        our_index,
        n_players,
        temperature,
        pool,
        explore=True,
    )
    result = run_match(
        agents,
        seed=seed,
        n_players=n_players,
        max_steps=int(config.get("max_turns", 100)),
        stop_player=our_index if bool(config.get("train_stop_on_elimination", True)) else None,
        game_engine=str(config.get("game_engine", "official_fast")),
        use_c_accel=bool(config.get("official_fast_c_accel", True)),
    )
    terminal_reward = 1.0 if int(result.get("winner", -1)) == int(our_index) else -1.0
    dense_reward = _strategic_dense_reward(result, our_index, config) if bool(config.get("dense_reward_enabled", True)) else 0.0
    action_metrics = _action_summary(action_records)
    activity_reward = _activity_shaping_reward(action_metrics, config)
    reward = float(terminal_reward + dense_reward + activity_reward)
    train_metrics = _train_episode(
        model,
        optimizer,
        log_probs,
        reward,
        baseline,
        entropy_coef=float(config.get("entropy_coef_start", 0.025)),
        action_records=action_records,
        value_coef=float(config.get("value_loss_coef", 0.25)),
    )
    scores = result.get("scores", [])
    rank = _rank_from_scores(scores, our_index, n_players)
    record = {
        "stage": stage,
        "n_players": n_players,
        "generation": generation,
        "local_step": local_step,
        "seed": seed,
        "our_index": our_index,
        "winner": int(result.get("winner", -1)),
        "win": 1.0 if int(result.get("winner", -1)) == int(our_index) else 0.0,
        "rank": rank,
        "episode_length": int(result.get("steps", 0)),
        "reward": reward,
        "terminal_reward": terminal_reward,
        "dense_reward": dense_reward,
        "activity_reward": activity_reward,
        "opponents": list(opp_names),
        **action_metrics,
        **train_metrics,
    }
    return reward, record


def _worker_train(task: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(task["config"])
    stage = str(task["stage"])
    n_players = int(task["n_players"])
    pool = list(task["pool"])
    worker_id = int(task["worker_id"])
    generation = int(task["generation"])
    seed = int(task["seed"])
    checkpoint_path = task.get("checkpoint_path")
    train_games = max(1, int(task["train_games"]))
    deadline_epoch = float(task.get("deadline_epoch", 0.0))

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = _load_model(config, checkpoint_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 0.0002)))
    baseline = float(config.get("moving_average_baseline", 0.0))
    baseline_momentum = max(0.0, min(1.0, float(config.get("baseline_momentum", 0.10))))
    rewards: List[float] = []
    wins: List[float] = []
    ranks: List[float] = []
    noops: List[float] = []
    ships_sent: List[float] = []
    action_counts: List[float] = []
    last_record: Dict[str, Any] = {}

    for local_step in range(train_games):
        if deadline_epoch > 0.0 and time.time() >= deadline_epoch:
            break
        our_index = (worker_id + local_step) % n_players
        episode_seed = seed + local_step * 9973
        progress = float(task.get("progress", 0.0))
        reward, record = _play_train_episode(
            model,
            optimizer,
            config,
            stage=stage,
            n_players=n_players,
            pool=pool,
            seed=episode_seed,
            our_index=our_index,
            generation=generation,
            local_step=local_step,
            progress=progress,
            baseline=baseline,
        )
        baseline = (1.0 - baseline_momentum) * baseline + baseline_momentum * reward
        rewards.append(reward)
        wins.append(float(record["win"]))
        ranks.append(float(record["rank"]))
        noops.append(float(record["do_nothing_rate"]))
        ships_sent.append(float(record["avg_ships_sent"]))
        action_counts.append(float(record["action_count"]))
        last_record = record

    return {
        "worker_id": worker_id,
        "state": _state_to_cpu(model.state_dict()),
        "train_games_completed": len(rewards),
        "train_reward_mean": float(np.mean(rewards) if rewards else 0.0),
        "train_winrate": float(np.mean(wins) if wins else 0.0),
        "train_rank_mean": float(np.mean(ranks) if ranks else float(n_players)),
        "train_do_nothing_rate": float(np.mean(noops) if noops else 1.0),
        "train_avg_ships_sent": float(np.mean(ships_sent) if ships_sent else 0.0),
        "train_action_count": float(np.mean(action_counts) if action_counts else 0.0),
        "last_record": last_record,
    }


def _evaluate_state(
    state: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    *,
    stage: str,
    n_players: int,
    pool: Sequence[str],
    episodes: int,
    seed_offset: int,
) -> Dict[str, Any]:
    model = NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(config), hidden_dim=int(config.get("hidden_dim", 320))))
    load_compatible_state_dict(model, state)
    rows: List[Dict[str, Any]] = []
    for i in range(max(1, int(episodes))):
        seed = seed_offset + i
        our_index = i % n_players
        agents, _log_probs, action_records, opp_names = _build_agents_n(
            model,
            config,
            seed,
            our_index,
            n_players,
            temperature=0.0,
            pool=pool,
            explore=False,
        )
        result = run_match(
            agents,
            seed=seed,
            n_players=n_players,
            max_steps=int(config.get("max_turns", 100)),
            stop_player=our_index,
            game_engine=str(config.get("game_engine", "official_fast")),
            use_c_accel=bool(config.get("official_fast_c_accel", True)),
        )
        scores = result.get("scores", [])
        rank = _rank_from_scores(scores, our_index, n_players)
        rows.append(
            {
                "winner": int(result.get("winner", -1)),
                "our_index": our_index,
                "win": 1.0 if int(result.get("winner", -1)) == int(our_index) else 0.0,
                "rank": rank,
                "scores": scores,
                "steps": int(result.get("steps", 0)),
                "opponents": list(opp_names),
                **_action_summary(action_records),
            }
        )
    opponent_names = sorted({name for row in rows for name in row.get("opponents", [])})
    eval_by_opponent = {}
    for name in opponent_names:
        matched = [row for row in rows if name in row.get("opponents", [])]
        eval_by_opponent[name] = {
            "games": len(matched),
            "winrate": float(np.mean([row["win"] for row in matched]) if matched else 0.0),
            "rank_mean": float(np.mean([row["rank"] for row in matched]) if matched else float(n_players)),
        }
    wins = [row["win"] for row in rows]
    ranks = [row["rank"] for row in rows]
    our_scores = [float(row["scores"][row["our_index"]]) for row in rows if len(row["scores"]) > row["our_index"]]
    by_position = {
        f"p{pos}": float(np.mean([row["win"] for row in rows if row["our_index"] == pos]) if any(row["our_index"] == pos for row in rows) else 0.0)
        for pos in range(n_players)
    }
    record = {
        "stage": stage,
        "n_players": n_players,
        "eval_episodes": len(rows),
        "winrate": float(np.mean(wins) if wins else 0.0),
        "rank_mean": float(np.mean(ranks) if ranks else float(n_players)),
        "avg_score": float(np.mean(our_scores) if our_scores else 0.0),
        "avg_episode_length": float(np.mean([row["steps"] for row in rows]) if rows else 0.0),
        "eval_action_count": float(np.mean([row["action_count"] for row in rows]) if rows else 0.0),
        "eval_do_nothing_rate": float(np.mean([row["do_nothing_rate"] for row in rows]) if rows else 1.0),
        "eval_avg_ships_sent": float(np.mean([row["avg_ships_sent"] for row in rows]) if rows else 0.0),
        "winrate_by_position": by_position,
        "eval_by_opponent": eval_by_opponent,
        "seeds": [seed_offset + i for i in range(len(rows))],
    }
    record["eval_mean"] = 2.0 * float(record["winrate"]) - 1.0
    record["score"] = _composite_score(record)
    record["composite_score"] = record["score"]
    return record


def _worker_eval(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "worker_id": int(task["worker_id"]),
        "record": _evaluate_state(
            task["state"],
            task["config"],
            stage=str(task["stage"]),
            n_players=int(task["n_players"]),
            pool=list(task["pool"]),
            episodes=int(task["episodes"]),
            seed_offset=int(task["seed_offset"]),
        ),
    }


def _run_parallel(tasks: List[Dict[str, Any]], workers: int, fn, label: str, log_path: Path, started_at: float, deadline_epoch: float) -> List[Dict[str, Any]]:
    if not tasks:
        return []
    results: List[Dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(max(1, workers), len(tasks))) as process_pool:
        pending = [process_pool.apply_async(fn, (task,)) for task in tasks]
        done = 0
        while pending:
            if deadline_epoch > 0.0 and time.time() >= deadline_epoch:
                process_pool.terminate()
                process_pool.join()
                _log(log_path, f"{label} deadline_reached done={done}/{len(tasks)} elapsed={(time.time() - started_at)/60.0:.1f}m")
                return results
            ready = []
            for idx, job in enumerate(pending):
                if job.ready():
                    try:
                        results.append(job.get())
                    except Exception as exc:
                        _log(log_path, f"{label} task_failed {type(exc).__name__}: {exc}")
                    ready.append(idx)
            if not ready:
                time.sleep(0.2)
                continue
            for idx in reversed(ready):
                pending.pop(idx)
                done += 1
                _log(log_path, f"{label} done {done}/{len(tasks)} elapsed={(time.time() - started_at)/60.0:.1f}m")
    return results


def _promote(
    checkpoint: Path,
    best_path: Path,
    stage_best_path: Path,
    final_path: Path | None,
    state: Dict[str, torch.Tensor],
    record: Dict[str, Any],
) -> None:
    save_checkpoint(checkpoint, state, record)
    save_checkpoint(best_path, state, record)
    save_checkpoint(stage_best_path, state, record)
    if final_path is not None:
        save_checkpoint(final_path, state, record)


def _stage_score_key(record: Dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    return (
        float(record.get("winrate", 0.0)),
        -float(record.get("rank_mean", 99.0)),
        *_score_record(record),
    )


def _run_stage(
    cfg: Dict[str, Any],
    *,
    stage: str,
    n_players: int,
    pool: Sequence[str],
    target_winrate: float,
    train_games_per_worker: int,
    eval_episodes: int,
    confirm_episodes: int,
    workers: int,
    base_checkpoint: Path | None,
    run_dir: Path,
    log_path: Path,
    jsonl_path: Path,
    started_at: float,
    deadline_epoch: float | None,
) -> Dict[str, Any]:
    stage_dir = run_dir / stage
    checkpoint_dir = stage_dir / "checkpoints"
    ensure_dir(checkpoint_dir)
    best_path = checkpoint_dir / "best.npz"
    latest_path = checkpoint_dir / "latest.npz"
    candidate_path = checkpoint_dir / "candidate.npz"
    stage_best_path = run_dir / f"best_{stage}.npz"
    final_stage_path = run_dir / "final_agent.npz" if stage == "stage2_4p" else None

    best_record: Dict[str, Any] = {}
    best_state: Dict[str, torch.Tensor] | None = None
    best_winrate = -1.0
    generation = 0
    base = base_checkpoint
    parameter_count = count_parameters(NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(cfg), hidden_dim=int(cfg.get("hidden_dim", 320)))))
    _log(log_path, f"{stage} start n_players={n_players} target_winrate={target_winrate:.3f} pool={list(pool)} base={base or ''}")

    while True:
        if deadline_epoch is not None:
            now = time.time()
            remaining_minutes = (deadline_epoch - now) / 60.0
            if remaining_minutes <= 0.0:
                _log(log_path, f"{stage} stop: deadline reached")
                break
            if remaining_minutes < float(cfg.get("min_generation_remaining_minutes", 8.0)):
                _log(log_path, f"{stage} stop: remaining={remaining_minutes:.1f}m below min_generation_remaining")
                break
        base_str = str(base) if base and base.exists() else None
        elapsed = (time.time() - started_at) / 60.0
        _log(
            log_path,
            f"{stage} generation {generation} start elapsed={elapsed:.1f}m best_winrate={best_winrate:.3f} "
            f"workers={workers} train_games={train_games_per_worker} eval={eval_episodes}",
        )
        train_tasks = [
            {
                "config": cfg,
                "stage": stage,
                "n_players": n_players,
                "pool": list(pool),
                "worker_id": worker_id,
                "generation": generation,
                "seed": int(cfg["seed"]) + generation * 100000 + worker_id * 4099 + (0 if n_players == 2 else 7000000),
                "checkpoint_path": base_str,
                "train_games": train_games_per_worker,
                "progress": min(1.0, elapsed / max(1.0, float(cfg.get("duration_minutes", 600.0)))),
                "deadline_epoch": deadline_epoch or 0.0,
            }
            for worker_id in range(workers)
        ]
        trained = _run_parallel(train_tasks, workers, _worker_train, f"{stage} generation {generation} train", log_path, started_at, deadline_epoch)
        if not trained:
            break
        eval_tasks = [
            {
                "config": cfg,
                "stage": stage,
                "n_players": n_players,
                "pool": list(pool),
                "worker_id": int(item["worker_id"]),
                "state": item["state"],
                "episodes": eval_episodes,
                "seed_offset": int(cfg["seed"]) + 50000 + generation * 1000 + int(item["worker_id"]) * 100 + (0 if n_players == 2 else 7000000),
            }
            for item in trained
        ]
        evaluated_results = _run_parallel(eval_tasks, workers, _worker_eval, f"{stage} generation {generation} eval", log_path, started_at, deadline_epoch)
        if not evaluated_results:
            break
        train_by_worker = {int(item["worker_id"]): item for item in trained}
        evaluated = []
        for item in evaluated_results:
            worker_id = int(item["worker_id"])
            record = {
                **item["record"],
                **{k: v for k, v in train_by_worker[worker_id].items() if k not in {"state", "last_record"}},
                "worker_last_record": train_by_worker[worker_id].get("last_record", {}),
                "generation": generation,
                "worker_id": worker_id,
                "parameter_count": parameter_count,
                "pool": list(pool),
                "target_winrate": target_winrate,
                "games_per_hour_estimate": 0.0,
                "checkpoint_promoted": False,
                "promotion_reason": "no promotion",
            }
            evaluated.append({"record": record, "state": train_by_worker[worker_id]["state"]})
        evaluated.sort(key=lambda item: _stage_score_key(item["record"]), reverse=True)
        generation_best = evaluated[0]
        save_checkpoint(candidate_path, generation_best["state"], generation_best["record"])
        save_checkpoint(latest_path, generation_best["state"], generation_best["record"])
        # Keep base anchored on best checkpoint (avoid drifting onto failed candidates).
        # Previously: base = latest_path (regardless of promotion) -> caused stage2 collapse
        # when generations got worse (each gen trained from previous failed candidate).
        if best_path.exists():
            base = best_path
        else:
            base = latest_path

        best_candidate_record = dict(generation_best["record"])
        # Loosen confirm gate: any candidate within striking distance of best gets a confirm
        # eval (reduces 24-episode promotion variance ~half).
        candidate_winrate_raw = float(best_candidate_record.get("winrate", 0.0))
        confirm_floor = max(0.0, best_winrate - float(cfg.get("promotion_margin", 0.01)))
        should_confirm = (
            candidate_winrate_raw >= confirm_floor
            or best_winrate < 0.0
        )
        promoted = False
        if should_confirm and (deadline_epoch is None or time.time() < deadline_epoch):
            confirmed = _evaluate_state(
                generation_best["state"],
                cfg,
                stage=stage,
                n_players=n_players,
                pool=pool,
                episodes=confirm_episodes,
                seed_offset=int(cfg["seed"]) + 900000 + generation * 1000 + (0 if n_players == 2 else 7000000),
            )
            best_candidate_record.update(confirmed)
            best_candidate_record["eval_phase"] = "confirmed"
            best_candidate_record["promotion_eval_episodes"] = confirm_episodes
            best_candidate_record["generation"] = generation
            best_candidate_record["worker_id"] = int(generation_best["record"]["worker_id"])
            generation_best["record"].update(best_candidate_record)
        candidate_winrate = float(generation_best["record"].get("winrate", 0.0))
        if candidate_winrate > best_winrate + float(cfg.get("promotion_margin", 0.01)) or best_winrate < 0.0:
            best_winrate = candidate_winrate
            best_record = dict(generation_best["record"])
            best_record["checkpoint_promoted"] = True
            best_record["promotion_reason"] = f"stage best winrate improved to {best_winrate:.4f}"
            best_state = generation_best["state"]
            _promote(candidate_path, best_path, stage_best_path, final_stage_path if candidate_winrate >= target_winrate else None, best_state, best_record)
            base = best_path
            promoted = True
        generation_best["record"]["checkpoint_promoted"] = promoted
        if promoted:
            generation_best["record"]["promotion_reason"] = best_record["promotion_reason"]
        for item in evaluated:
            if item is generation_best:
                item["record"].update(generation_best["record"])
            append_jsonl(jsonl_path, item["record"])
        games_this_generation = sum(int(item["record"].get("train_games_completed", 0)) for item in evaluated) + len(evaluated) * eval_episodes
        _log(
            log_path,
            f"{stage} generation {generation} candidate_winrate={float(generation_best['record'].get('winrate', 0.0)):.3f} "
            f"rank={float(generation_best['record'].get('rank_mean', 0.0)):.2f} noop={float(generation_best['record'].get('eval_do_nothing_rate', 0.0)):.2f} "
            f"promoted={int(promoted)} best_winrate={best_winrate:.3f} target={target_winrate:.3f} games={games_this_generation}",
        )
        if best_winrate >= target_winrate:
            _log(log_path, f"{stage} target_reached best_winrate={best_winrate:.3f} checkpoint={best_path}")
            break
        generation += 1
    return {
        "stage": stage,
        "n_players": n_players,
        "target_winrate": target_winrate,
        "best_winrate": best_winrate,
        "best_checkpoint": str(best_path) if best_path.exists() else "",
        "stage_best_checkpoint": str(stage_best_path) if stage_best_path.exists() else "",
        "best_record": best_record,
        "target_reached": best_winrate >= target_winrate,
    }


def _prepare_config(cfg: Dict[str, Any], args: argparse.Namespace, run_name: str) -> Dict[str, Any]:
    cfg = dict(cfg)
    cfg["run_name"] = run_name
    cfg["duration_minutes"] = float(args.duration_minutes)
    cfg["workers"] = max(1, int(args.workers))
    cfg["hidden_dim"] = max(320, int(cfg.get("hidden_dim", 320)))
    cfg["learning_rate"] = min(float(cfg.get("learning_rate", 0.00025)), 0.0002)
    cfg["game_engine"] = "official_fast"
    cfg["official_fast_c_accel"] = True
    cfg["train_stop_on_elimination"] = True
    cfg["max_turns"] = min(100, int(cfg.get("max_turns", 100)))
    cfg["max_actions_per_turn"] = 4
    cfg["min_expand_attack_ships"] = max(4, int(cfg.get("min_expand_attack_ships", 4)))
    cfg["send_ratios"] = [0.45, 0.65, 0.85]
    cfg["policy_prior_strength"] = max(1.30, float(cfg.get("policy_prior_strength", 1.30)))
    cfg["imitation_warmstart_steps"] = 0
    cfg["dense_reward_enabled"] = True
    cfg["dense_planet_coef"] = 0.05
    cfg["dense_production_coef"] = 0.04
    cfg["dense_ship_share_coef"] = 0.14
    cfg["dense_score_coef"] = 0.10
    cfg["dense_survival_coef"] = 0.05
    cfg["dense_reward_clip"] = 0.30
    cfg["train_target_do_nothing_rate"] = 0.48
    cfg["train_noop_penalty_coef"] = 0.45
    cfg["train_action_bonus_coef"] = 0.09
    cfg["train_ships_sent_bonus_coef"] = 0.05
    cfg["train_activity_reward_clip"] = 0.35
    cfg["temperature_start"] = 1.05
    cfg["temperature_end"] = 0.18
    cfg["entropy_coef_start"] = 0.025
    cfg["entropy_coef_end"] = 0.004
    cfg["baseline_momentum"] = 0.10
    cfg["promotion_margin"] = 0.01
    cfg["min_generation_remaining_minutes"] = 8.0
    cfg["stage1_target_winrate"] = float(args.stage1_target)
    cfg["stage2_target_winrate"] = float(args.stage2_target)
    cfg["stage1_pool"] = ["random", "greedy", "starter", "distance"]
    cfg["stage2_pool"] = ["random", "greedy", "starter"]
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official_fast 2p pretrain then 4p target training without changing the game engine.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--duration-minutes", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stage1-target", type=float, default=0.85)
    parser.add_argument("--stage2-target", type=float, default=0.70)
    parser.add_argument("--stage1-train-games", type=int, default=160)
    parser.add_argument("--stage2-train-games", type=int, default=96)
    parser.add_argument("--stage1-eval-episodes", type=int, default=64)
    parser.add_argument("--stage2-eval-episodes", type=int, default=48)
    parser.add_argument("--stage1-confirm-episodes", type=int, default=192)
    parser.add_argument("--stage2-confirm-episodes", type=int, default=160)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-checkpoint", default="")
    # Resume directly from a stage1 best checkpoint, skipping stage1 entirely.
    parser.add_argument("--resume-stage1-best", default="", help="Path to best_stage1_2p.npz (or equivalent) — skip stage1 if provided")
    # Resume stage2 from an in-progress best_stage2_4p.npz (e.g. continuing the VPS run locally).
    parser.add_argument("--resume-stage2-best", default="", help="Path to best_stage2_4p.npz to resume stage2 best from")
    parser.add_argument("--stage2-lr-scale", type=float, default=0.5, help="LR multiplier applied at stage2 start to slow forgetting")
    parser.add_argument("--stage2-entropy-start", type=float, default=0.040, help="Higher entropy at stage2 start to escape noop attractor")
    parser.add_argument("--stage2-entropy-end", type=float, default=0.010)
    args = parser.parse_args()

    run_name = args.run_name or _run_tag()
    cfg = _prepare_config(_load_config(args.config), args, run_name)
    run_dir = PACKAGE_DIR / "runs" / run_name
    logs_dir = run_dir / "logs"
    ensure_dir(logs_dir)
    log_path = logs_dir / "direct.log"
    jsonl_path = logs_dir / "training.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    save_json(manifest_path, cfg)
    configure_run_logging(log_path)
    started_at = time.time()
    deadline_epoch = None if float(cfg["duration_minutes"]) <= 0.0 else started_at + float(cfg["duration_minutes"]) * 60.0
    _log(log_path, f"run_start run_name={run_name} host={socket.gethostname()} workers={cfg['workers']} manifest={manifest_path}")

    base_checkpoint = Path(args.resume_checkpoint) if args.resume_checkpoint else None
    cfg["stage2_lr_scale"] = float(args.stage2_lr_scale)
    cfg["stage2_entropy_start"] = float(args.stage2_entropy_start)
    cfg["stage2_entropy_end"] = float(args.stage2_entropy_end)

    # Skip stage1 if a stage1 best checkpoint is supplied (e.g. resume from VPS run).
    if args.resume_stage1_best:
        prebuilt_stage1_best = Path(args.resume_stage1_best)
        if not prebuilt_stage1_best.exists():
            raise FileNotFoundError(f"--resume-stage1-best path not found: {prebuilt_stage1_best}")
        # Copy into expected stage1 layout so downstream code finds it.
        stage1_dir = run_dir / "stage1_2p" / "checkpoints"
        ensure_dir(stage1_dir)
        stage1_best_path = stage1_dir / "best.npz"
        from neural_network.src.storage import load_checkpoint as _ldc, save_checkpoint as _svc
        st, meta = _ldc(str(prebuilt_stage1_best))
        _svc(stage1_best_path, st, meta if isinstance(meta, dict) else {})
        _svc(run_dir / "best_stage1_2p.npz", st, meta if isinstance(meta, dict) else {})
        _log(log_path, f"stage1_2p resumed from {prebuilt_stage1_best} -> {stage1_best_path}")
        stage1 = {
            "stage": "stage1_2p",
            "n_players": 2,
            "target_winrate": float(cfg["stage1_target_winrate"]),
            "best_winrate": float((meta or {}).get("winrate", cfg["stage1_target_winrate"])) if isinstance(meta, dict) else float(cfg["stage1_target_winrate"]),
            "best_checkpoint": str(stage1_best_path),
            "stage_best_checkpoint": str(run_dir / "best_stage1_2p.npz"),
            "best_record": meta if isinstance(meta, dict) else {},
            "target_reached": True,
        }
    else:
        stage1 = _run_stage(
            cfg,
            stage="stage1_2p",
            n_players=2,
            pool=cfg["stage1_pool"],
            target_winrate=float(cfg["stage1_target_winrate"]),
            train_games_per_worker=int(args.stage1_train_games),
            eval_episodes=int(args.stage1_eval_episodes),
            confirm_episodes=int(args.stage1_confirm_episodes),
            workers=int(cfg["workers"]),
            base_checkpoint=base_checkpoint,
            run_dir=run_dir,
            log_path=log_path,
            jsonl_path=jsonl_path,
            started_at=started_at,
            deadline_epoch=deadline_epoch,
        )
    if stage1["target_reached"]:
        stage2_base = Path(stage1["best_checkpoint"])
        # If user supplied an in-progress stage2 best, seed stage2 best.npz so resume picks it up.
        if args.resume_stage2_best:
            prebuilt_stage2_best = Path(args.resume_stage2_best)
            if not prebuilt_stage2_best.exists():
                raise FileNotFoundError(f"--resume-stage2-best path not found: {prebuilt_stage2_best}")
            stage2_dir = run_dir / "stage2_4p" / "checkpoints"
            ensure_dir(stage2_dir)
            from neural_network.src.storage import load_checkpoint as _ldc2, save_checkpoint as _svc2
            st2, meta2 = _ldc2(str(prebuilt_stage2_best))
            _svc2(stage2_dir / "best.npz", st2, meta2 if isinstance(meta2, dict) else {})
            _svc2(run_dir / "best_stage2_4p.npz", st2, meta2 if isinstance(meta2, dict) else {})
            stage2_base = stage2_dir / "best.npz"
            _log(log_path, f"stage2_4p resumed best from {prebuilt_stage2_best}")
        # Stage2 specific overrides: lower LR + higher entropy to mitigate
        # catastrophic forgetting when transferring from passive 2p winner.
        cfg_stage2 = dict(cfg)
        cfg_stage2["learning_rate"] = float(cfg.get("learning_rate", 0.0002)) * float(cfg.get("stage2_lr_scale", 0.5))
        cfg_stage2["entropy_coef_start"] = float(cfg.get("stage2_entropy_start", 0.040))
        cfg_stage2["entropy_coef_end"] = float(cfg.get("stage2_entropy_end", 0.010))
        # Lower noop target so penalty is gentler — model trained as noop in stage1
        # gets crushed if penalty is too aggressive at transfer.
        cfg_stage2["train_noop_penalty_coef"] = float(cfg.get("train_noop_penalty_coef", 0.45)) * 0.5
        cfg_stage2["promotion_margin"] = float(cfg.get("promotion_margin", 0.01))
        stage2 = _run_stage(
            cfg_stage2,
            stage="stage2_4p",
            n_players=4,
            pool=cfg["stage2_pool"],
            target_winrate=float(cfg["stage2_target_winrate"]),
            train_games_per_worker=int(args.stage2_train_games),
            eval_episodes=int(args.stage2_eval_episodes),
            confirm_episodes=int(args.stage2_confirm_episodes),
            workers=int(cfg["workers"]),
            base_checkpoint=stage2_base,
            run_dir=run_dir,
            log_path=log_path,
            jsonl_path=jsonl_path,
            started_at=started_at,
            deadline_epoch=deadline_epoch,
        )
    else:
        stage2 = {
            "stage": "stage2_4p",
            "target_reached": False,
            "skipped": True,
            "reason": "stage1 target not reached yet",
        }

    result = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "log": str(log_path),
        "jsonl": str(jsonl_path),
        "duration_minutes_requested": cfg["duration_minutes"],
        "elapsed_minutes": (time.time() - started_at) / 60.0,
        "stage1": stage1,
        "stage2": stage2,
        "final_agent": str(run_dir / "final_agent.npz") if (run_dir / "final_agent.npz").exists() else "",
        "completed_at": _now(),
    }
    save_json(run_dir / "agent_dossier.json", result)
    _log(log_path, f"run_done final_agent={result['final_agent']} dossier={run_dir / 'agent_dossier.json'}")
    print(json.dumps(result, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
