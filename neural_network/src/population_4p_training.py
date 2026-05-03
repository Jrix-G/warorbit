from __future__ import annotations

import multiprocessing as mp
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from opponents import ZOO, training_pool

from .model import ModelConfig, NeuralNetworkModel, count_parameters, load_compatible_state_dict
from .notebook_4p_training import _action_summary, _build_agents, _episode_reward, _infer_input_dim, _train_episode, evaluate_4p, run_match
from .storage import append_jsonl, load_checkpoint, save_checkpoint
from .torch_compat import ensure_torch_dynamo_stub
from .utils import ensure_dir

MAX_DURATION_MINUTES = 480.0


def _checkpoint_to_load(config: Dict[str, Any], resume: bool) -> Path | None:
    if not resume:
        return None
    for key in ("resume_checkpoint", "best_checkpoint", "latest_checkpoint"):
        value = config.get(key)
        if value and Path(value).exists():
            return Path(value)
    return None


def _notebook_pool(limit: int) -> List[str]:
    pool = training_pool(limit=limit)
    if not pool:
        pool = [name for name in ZOO.keys() if name.startswith("notebook_")]
    return pool[:limit] if limit > 0 else pool


def _state_to_cpu(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in state.items()}


def _load_base_model(config: Dict[str, Any], checkpoint_path: str | None) -> NeuralNetworkModel:
    ensure_torch_dynamo_stub()
    model = NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(config), hidden_dim=int(config.get("hidden_dim", 256))))
    if checkpoint_path and Path(checkpoint_path).exists():
        state, _ = load_checkpoint(checkpoint_path)
        load_compatible_state_dict(model, state)
    return model


def _worker_train_candidate(task: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(task["config"])
    worker_id = int(task["worker_id"])
    generation = int(task["generation"])
    seed = int(task["seed"])
    checkpoint_path = task.get("checkpoint_path")
    pool = list(task["pool"])
    train_steps = max(1, int(config.get("worker_train_steps", 4)))
    deadline_epoch = float(task.get("deadline_epoch", 0.0))

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = _load_base_model(config, checkpoint_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    baseline = float(config.get("moving_average_baseline", 0.0))
    rewards: List[float] = []
    wins: List[float] = []
    last_record: Dict[str, Any] = {}

    for local_step in range(train_steps):
        if deadline_epoch > 0.0 and time.time() >= deadline_epoch:
            break
        absolute_step = generation * train_steps + local_step
        progress = absolute_step / max(1, int(config.get("train_steps", train_steps)) - 1)
        temperature = float(config.get("temperature_start", 1.2)) + (
            float(config.get("temperature_end", 0.3)) - float(config.get("temperature_start", 1.2))
        ) * min(1.0, max(0.0, progress))
        our_index = (worker_id + local_step) % 4
        episode_seed = seed + local_step * 9973
        agents, log_probs, action_records, opp_names = _build_agents(model, config, episode_seed, our_index, temperature, pool)
        result = run_match(agents, seed=episode_seed, n_players=4, max_steps=int(config.get("max_turns", 100)))
        reward = _episode_reward(result, our_index)
        action_metrics = _action_summary(action_records)
        baseline = 0.95 * baseline + 0.05 * reward
        train_metrics = _train_episode(model, optimizer, log_probs, reward, baseline)
        rewards.append(reward)
        wins.append(1.0 if int(result.get("winner", -1)) == our_index else 0.0)
        last_record = {
            "worker_id": worker_id,
            "local_step": local_step,
            "reward": reward,
            "baseline": baseline,
            "temperature": temperature,
            "grad_norm": train_metrics["grad_norm"],
            "loss": train_metrics["loss"],
            "winner": int(result.get("winner", -1)),
            "our_index": our_index,
            "opponents": opp_names,
            "episode_length": int(result.get("steps", 0)),
            **action_metrics,
        }

    return {
        "worker_id": worker_id,
        "seed": seed,
        "state": _state_to_cpu(model.state_dict()),
        "train_reward_mean": float(np.mean(rewards) if rewards else 0.0),
        "train_winrate": float(np.mean(wins) if wins else 0.0),
        "last_record": last_record,
    }


def _evaluate_candidate(state: Dict[str, torch.Tensor], config: Dict[str, Any], pool: Sequence[str], episodes: int, seed_offset: int) -> Dict[str, Any]:
    model = NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(config), hidden_dim=int(config.get("hidden_dim", 256))))
    load_compatible_state_dict(model, state)
    return evaluate_4p(model, config, pool, episodes=episodes, seed_offset=seed_offset)


def _worker_evaluate_candidate(task: Dict[str, Any]) -> Dict[str, Any]:
    deadline_epoch = float(task.get("deadline_epoch", 0.0))
    if deadline_epoch > 0.0 and time.time() >= deadline_epoch:
        return {"worker_id": int(task["worker_id"]), "record": {"eval_mean": 0.0, "eval_std": 0.0, "winrate": 0.0, "winrate_by_position": {}, "rank_mean": 4.0, "avg_score": 0.0, "avg_episode_length": 0.0, "eval_action_count": 0.0, "eval_do_nothing_rate": 1.0, "eval_avg_ships_sent": 0.0, "seeds": []}}
    return {
        "worker_id": int(task["worker_id"]),
        "record": _evaluate_candidate(
            task["state"],
            task["config"],
            task["pool"],
            int(task["episodes"]),
            int(task["seed_offset"]),
        ),
    }


def _run_parallel(tasks: List[Dict[str, Any]], workers: int, fn, label: str, started_at: float, deadline_epoch: float) -> List[Dict[str, Any]]:
    if not tasks:
        return []
    if workers == 1:
        if deadline_epoch > 0.0 and time.time() >= deadline_epoch:
            return []
        results = [fn(tasks[0])]
        print(
            f"{label} done 1/{len(tasks)} elapsed={(time.time() - started_at)/60.0:.1f}m",
            flush=True,
        )
        return results

    results: List[Dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(workers, len(tasks))) as process_pool:
        pending = [process_pool.apply_async(fn, (task,)) for task in tasks]
        done = 0
        while pending:
            if deadline_epoch > 0.0 and time.time() >= deadline_epoch:
                process_pool.terminate()
                process_pool.join()
                print(
                    f"{label} hard_deadline_reached elapsed={(time.time() - started_at)/60.0:.1f}m",
                    flush=True,
                )
                return results
            newly_done = []
            for idx, job in enumerate(pending):
                if job.ready():
                    try:
                        results.append(job.get())
                    except Exception:
                        pass
                    newly_done.append(idx)
            if not newly_done:
                time.sleep(0.2)
                continue
            for idx in reversed(newly_done):
                pending.pop(idx)
                done += 1
                print(
                    f"{label} done {done}/{len(tasks)} elapsed={(time.time() - started_at)/60.0:.1f}m",
                    flush=True,
                )
    return results


def run_population_4p_training(config: Dict[str, Any], resume: bool = True) -> Dict[str, Any]:
    cfg = dict(config)
    cfg["hidden_dim"] = int(cfg.get("hidden_dim", 256))
    cfg["train_notebook_opponents"] = int(cfg.get("train_notebook_opponents", 3))
    cfg["notebook_pool_limit"] = int(cfg.get("notebook_pool_limit", 15))
    cfg["notebook_pool_limit_max"] = int(cfg.get("notebook_pool_limit_max", cfg["notebook_pool_limit"]))
    cfg["duration_minutes"] = min(float(cfg.get("duration_minutes", 90.0)), MAX_DURATION_MINUTES)

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    log_dir = Path(cfg["log_dir"])
    best_path = Path(cfg.get("best_checkpoint", checkpoint_dir / "best.npz"))
    latest_path = Path(cfg.get("latest_checkpoint", checkpoint_dir / "latest.npz"))
    candidate_path = Path(cfg.get("candidate_checkpoint", checkpoint_dir / "candidate.npz"))
    log_path = log_dir / "population_4p_training.jsonl"
    ensure_dir(checkpoint_dir)
    ensure_dir(log_dir)
    ensure_torch_dynamo_stub()

    pool = _notebook_pool(int(cfg["notebook_pool_limit"]))
    workers = max(1, int(cfg.get("workers", 6)))
    duration_seconds = max(1.0, float(cfg["duration_minutes"]) * 60.0)
    eval_episodes = max(4, int(cfg.get("eval_episodes", 16)))
    started_at = time.time()
    deadline_epoch = started_at + duration_seconds
    generation = 0
    base_checkpoint = _checkpoint_to_load(cfg, resume)
    best_score = -1e9
    best_record: Dict[str, Any] = {}

    if base_checkpoint and base_checkpoint.exists():
        _, metadata = load_checkpoint(base_checkpoint)
        best_score = float(metadata.get("winrate", metadata.get("best_score", -1e9)))

    probe = NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(cfg), hidden_dim=int(cfg["hidden_dim"])))
    parameter_count = count_parameters(probe)

    while time.time() < deadline_epoch:
        elapsed = time.time() - started_at
        print(
            f"generation {generation} start elapsed={elapsed/60.0:.1f}m "
            f"best={best_score:.3f} pool={len(pool)} workers={workers}",
            flush=True,
        )
        if time.time() >= deadline_epoch:
            break
        checkpoint_str = str(base_checkpoint) if base_checkpoint else None
        tasks = [
            {
                "config": cfg,
                "worker_id": worker_id,
                "generation": generation,
                "seed": int(cfg["seed"]) + generation * 100000 + worker_id * 4099,
                "checkpoint_path": checkpoint_str,
                "pool": pool,
                "deadline_epoch": deadline_epoch,
            }
            for worker_id in range(workers)
        ]

        candidates = _run_parallel(tasks, workers, _worker_train_candidate, f"generation {generation} train", started_at, deadline_epoch)
        if not candidates:
            print("hard stop: no completed training candidates before deadline", flush=True)
            break

        eval_tasks = [
            {
                "worker_id": int(candidate["worker_id"]),
                "state": candidate["state"],
                "config": cfg,
                "pool": pool,
                "episodes": eval_episodes,
                "seed_offset": int(cfg["seed"]) + 50000 + generation * 1000 + int(candidate["worker_id"]) * 100,
                "deadline_epoch": deadline_epoch,
            }
            for candidate in candidates
        ]
        eval_results = _run_parallel(eval_tasks, workers, _worker_evaluate_candidate, f"generation {generation} eval", started_at, deadline_epoch)
        if not eval_results:
            print("hard stop: no completed evaluation before deadline", flush=True)
            break

        evaluated: List[Dict[str, Any]] = []
        eval_by_worker = {int(item["worker_id"]): item["record"] for item in eval_results if "worker_id" in item and "record" in item}
        for candidate in candidates:
            if int(candidate["worker_id"]) not in eval_by_worker:
                continue
            eval_result = {"record": eval_by_worker[int(candidate["worker_id"])]}
            eval_stats = eval_result["record"]
            score = float(eval_stats["winrate"])
            record = {
                "generation": generation,
                "worker_id": int(candidate["worker_id"]),
                "parameter_count": parameter_count,
                "pool_size": len(pool),
                "pool": pool,
                "train_reward_mean": float(candidate["train_reward_mean"]),
                "train_winrate": float(candidate["train_winrate"]),
                **eval_stats,
                "score": score,
                "checkpoint_promoted": False,
                "promotion_reason": "no promotion",
                "worker_last_record": candidate["last_record"],
            }
            evaluated.append({"record": record, "state": candidate["state"]})
        if not evaluated:
            print("hard stop: no candidate had completed eval record before deadline", flush=True)
            break

        evaluated.sort(key=lambda item: (float(item["record"]["score"]), float(item["record"].get("eval_mean", 0.0))), reverse=True)
        generation_best = evaluated[0]
        save_checkpoint(candidate_path, generation_best["state"], generation_best["record"])
        save_checkpoint(latest_path, generation_best["state"], generation_best["record"])

        if float(generation_best["record"]["score"]) >= best_score:
            best_score = float(generation_best["record"]["score"])
            generation_best["record"]["checkpoint_promoted"] = True
            generation_best["record"]["promotion_reason"] = f"best population winrate {best_score:.4f}"
            best_record = dict(generation_best["record"])
            save_checkpoint(best_path, generation_best["state"], generation_best["record"])
            base_checkpoint = best_path
        else:
            base_checkpoint = best_path if best_path.exists() else latest_path

        for item in evaluated:
            if item is generation_best:
                item["record"].update(generation_best["record"])
            append_jsonl(log_path, item["record"])

        print(
            f"generation {generation} best_winrate={generation_best['record']['score']:.3f} "
            f"global_best={best_score:.3f} params={parameter_count} pool={len(pool)}",
            flush=True,
        )
        generation += 1

    return {
        "best_score": best_score,
        "best": str(best_path),
        "latest": str(latest_path),
        "candidate": str(candidate_path),
        "log": str(log_path),
        "generations": generation,
        "workers": workers,
        "parameter_count": parameter_count,
        "pool": pool,
        "best_record": best_record,
    }
