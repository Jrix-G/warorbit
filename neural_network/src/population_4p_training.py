from __future__ import annotations

import json
import multiprocessing as mp
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from .model import ModelConfig, NeuralNetworkModel, count_parameters, load_compatible_state_dict
from .notebook_4p_training import _action_summary, _build_agents, _episode_reward, _infer_input_dim, _train_episode, evaluate_4p, run_match, training_pool
from .storage import append_jsonl, load_checkpoint, save_checkpoint
from .torch_compat import ensure_torch_dynamo_stub
from .utils import ensure_dir

MAX_DURATION_MINUTES = 480.0

DEFAULT_CURRICULUM_TIERS: List[Dict[str, Any]] = [
    {
        "name": "basic_300",
        "label": "300 Elo: random/greedy/starter",
        "opponents": ["random", "greedy", "starter"],
        "min_generations": 2,
        "advance_score": 0.46,
        "advance_winrate": 0.50,
        "advance_rank_mean": 2.45,
        "advance_do_nothing_rate": 0.65,
        "candidate_eval_episodes": 4,
    },
    {
        "name": "heuristic_500",
        "label": "500 Elo: public heuristics",
        "opponents": ["greedy", "starter", "distance", "sun_dodge", "structured", "orbit_stars"],
        "min_generations": 2,
        "advance_score": 0.50,
        "advance_winrate": 0.45,
        "advance_rank_mean": 2.55,
        "advance_do_nothing_rate": 0.62,
        "candidate_eval_episodes": 6,
    },
    {
        "name": "mixed_700",
        "label": "700 Elo: heuristics plus starter notebooks",
        "opponents": [
            "distance",
            "sun_dodge",
            "structured",
            "orbit_stars",
            "notebook_kashiwaba_orbit_wars_reinforcement_learning_tutorial",
            "notebook_sigmaborov_orbit_wars_2026_starter",
            "notebook_pilkwang_orbit_wars_structured_baseline",
            "notebook_djenkivanov_orbit_wars_optimized_nearest_planet_sniper",
        ],
        "min_generations": 3,
        "advance_score": 0.52,
        "advance_winrate": 0.38,
        "advance_rank_mean": 2.75,
        "advance_do_nothing_rate": 0.60,
        "candidate_eval_episodes": 6,
    },
    {
        "name": "notebook_open",
        "label": "full notebook pool",
        "opponents": "notebook_pool",
        "min_generations": 0,
        "advance_score": 1.0,
        "advance_winrate": 1.0,
        "advance_rank_mean": 1.0,
        "advance_do_nothing_rate": 0.0,
        "candidate_eval_episodes": 8,
    },
]


def _checkpoint_to_load(config: Dict[str, Any], resume: bool) -> Path | None:
    if not resume:
        return None
    for key in ("resume_checkpoint", "best_checkpoint", "latest_checkpoint"):
        value = config.get(key)
        if value and Path(value).exists():
            return Path(value)
    return None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _composite_score(record: Dict[str, Any]) -> float:
    if "composite_score" in record:
        return float(record["composite_score"])
    if "score" in record and not any(key in record for key in ("winrate", "rank_mean", "eval_mean", "eval_do_nothing_rate")):
        return float(record["score"])
    winrate = _clip01(float(record.get("winrate", 0.0)))
    rank_mean = float(record.get("rank_mean", 4.0))
    rank_score = _clip01((4.0 - rank_mean) / 3.0)
    eval_mean = float(record.get("eval_mean", -1.0))
    reward_score = _clip01((eval_mean + 1.0) / 2.0)
    avg_score = _clip01(float(record.get("avg_score", 0.0)) / 1000.0)
    do_nothing = _clip01(float(record.get("eval_do_nothing_rate", record.get("do_nothing_rate", 1.0))))
    ships_sent = _clip01(float(record.get("eval_avg_ships_sent", record.get("avg_ships_sent", 0.0))) / 50.0)
    return float(
        0.50 * winrate
        + 0.25 * rank_score
        + 0.15 * reward_score
        + 0.05 * avg_score
        + 0.05 * ships_sent
        - 0.10 * do_nothing
    )


def _score_record(record: Dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    score = _composite_score(record)
    return (
        score,
        float(record.get("winrate", 0.0)),
        float(record.get("eval_mean", 0.0)),
        -float(record.get("rank_mean", 4.0)),
        float(record.get("avg_score", 0.0)),
        -float(record.get("eval_do_nothing_rate", 1.0)),
    )


def _should_try_promotion(record: Dict[str, Any], best_score: float, margin: float) -> bool:
    if best_score < -1e8:
        return True
    return _composite_score(record) >= best_score + margin


def _curriculum_tiers(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = config.get("opponent_curriculum_tiers")
    if isinstance(raw, list) and raw:
        return [dict(item) for item in raw if isinstance(item, dict)]
    return [dict(item) for item in DEFAULT_CURRICULUM_TIERS]


def _curriculum_state_path(config: Dict[str, Any], log_dir: Path) -> Path:
    value = config.get("opponent_curriculum_state")
    return Path(value) if value else log_dir / "opponent_curriculum_state.json"


def _load_curriculum_state(path: str | Path, tiers: Sequence[Dict[str, Any]], config: Dict[str, Any], resume: bool) -> Dict[str, Any]:
    path = Path(path)
    start_tier = max(0, min(len(tiers) - 1, int(config.get("opponent_curriculum_start_tier", 0))))
    state: Dict[str, Any] = {
        "tier_index": start_tier,
        "tier_generation": 0,
        "total_generations": 0,
        "history": [],
    }
    if resume and path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                state.update(loaded)
        except (OSError, json.JSONDecodeError):
            pass
    state["tier_index"] = max(0, min(len(tiers) - 1, int(state.get("tier_index", start_tier))))
    state["tier_generation"] = max(0, int(state.get("tier_generation", 0)))
    state["total_generations"] = max(0, int(state.get("total_generations", 0)))
    if not isinstance(state.get("history"), list):
        state["history"] = []
    state["tier_name"] = str(tiers[state["tier_index"]].get("name", f"tier_{state['tier_index']}"))
    return state


def _notebook_pool(limit: int) -> List[str]:
    pool = training_pool(limit=limit)
    if not pool:
        return []
    return pool[:limit] if limit > 0 else pool


def _tier_pool(config: Dict[str, Any], tier: Dict[str, Any]) -> List[str]:
    opponents = tier.get("opponents", [])
    if opponents == "notebook_pool":
        return _notebook_pool(int(config.get("notebook_pool_limit", 15)))
    if isinstance(opponents, str):
        opponents = [opponents]
    names = [str(name) for name in opponents]
    return names or ["random", "greedy", "starter"]


def _current_tier(state: Dict[str, Any], tiers: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return tiers[max(0, min(len(tiers) - 1, int(state.get("tier_index", 0))))]


def _maybe_advance_curriculum(state: Dict[str, Any], tiers: Sequence[Dict[str, Any]], record: Dict[str, Any]) -> tuple[bool, str]:
    idx = int(state.get("tier_index", 0))
    if idx >= len(tiers) - 1:
        return False, "already at final tier"
    tier = tiers[idx]
    tier_generation = int(state.get("tier_generation", 0))
    min_generations = int(tier.get("min_generations", 1))
    if tier_generation < min_generations:
        return False, f"tier generation {tier_generation} < {min_generations}"

    score = _composite_score(record)
    winrate = float(record.get("winrate", 0.0))
    rank_mean = float(record.get("rank_mean", 4.0))
    do_nothing = float(record.get("eval_do_nothing_rate", 1.0))
    score_ok = score >= float(tier.get("advance_score", 1.0))
    winrate_ok = winrate >= float(tier.get("advance_winrate", 1.0))
    rank_ok = rank_mean <= float(tier.get("advance_rank_mean", 1.0))
    action_ok = do_nothing <= float(tier.get("advance_do_nothing_rate", 0.0))
    if score_ok and winrate_ok and rank_ok and action_ok:
        next_idx = idx + 1
        state.setdefault("history", []).append(
            {
                "generation": int(state.get("total_generations", 0)),
                "from": tier.get("name", f"tier_{idx}"),
                "to": tiers[next_idx].get("name", f"tier_{next_idx}"),
                "score": score,
                "winrate": winrate,
                "rank_mean": rank_mean,
                "eval_do_nothing_rate": do_nothing,
            }
        )
        state["tier_index"] = next_idx
        state["tier_generation"] = 0
        state["tier_name"] = str(tiers[next_idx].get("name", f"tier_{next_idx}"))
        return True, f"advanced to {state['tier_name']}"
    return False, (
        f"hold score={score:.3f}/{float(tier.get('advance_score', 1.0)):.3f} "
        f"winrate={winrate:.3f}/{float(tier.get('advance_winrate', 1.0)):.3f} "
        f"rank={rank_mean:.2f}/{float(tier.get('advance_rank_mean', 1.0)):.2f} "
        f"noop={do_nothing:.2f}/{float(tier.get('advance_do_nothing_rate', 0.0)):.2f}"
    )


def _best_for_tier(state: Dict[str, Any], tier_name: str) -> float:
    tier_scores = state.get("tier_best_scores", {})
    if isinstance(tier_scores, dict) and tier_name in tier_scores:
        try:
            return float(tier_scores[tier_name])
        except (TypeError, ValueError):
            return -1e9
    return -1e9


def _write_curriculum_state(
    path: Path,
    state: Dict[str, Any],
    tier: Dict[str, Any],
    pool: Sequence[str],
    best_score: float,
    best_record: Dict[str, Any],
    last_record: Dict[str, Any] | None = None,
) -> None:
    snapshot = dict(state)
    snapshot.update(
        {
            "tier_name": tier.get("name", snapshot.get("tier_name")),
            "tier_label": tier.get("label", tier.get("name", "")),
            "pool": list(pool),
            "best_score": float(best_score),
            "best_record": best_record,
            "last_record": last_record or {},
            "updated_at": time.time(),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True, default=float), encoding="utf-8")


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
    ranks: List[float] = []
    do_nothing_rates: List[float] = []
    ships_sent: List[float] = []
    action_counts: List[float] = []
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
        stop_player = our_index if bool(config.get("train_stop_on_elimination", True)) else None
        result = run_match(agents, seed=episode_seed, n_players=4, max_steps=int(config.get("max_turns", 100)), stop_player=stop_player)
        reward = _episode_reward(result, our_index)
        action_metrics = _action_summary(action_records)
        scores = result.get("scores", [])
        ordered = sorted(((float(score), idx) for idx, score in enumerate(scores)), reverse=True)
        rank = next((rank for rank, (_, idx) in enumerate(ordered, start=1) if idx == our_index), 4)
        baseline = 0.95 * baseline + 0.05 * reward
        train_metrics = _train_episode(model, optimizer, log_probs, reward, baseline)
        rewards.append(reward)
        wins.append(1.0 if int(result.get("winner", -1)) == our_index else 0.0)
        ranks.append(float(rank))
        do_nothing_rates.append(float(action_metrics["do_nothing_rate"]))
        ships_sent.append(float(action_metrics["avg_ships_sent"]))
        action_counts.append(float(action_metrics["action_count"]))
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
            "rank": int(rank),
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
        "train_rank_mean": float(np.mean(ranks) if ranks else 4.0),
        "train_do_nothing_rate": float(np.mean(do_nothing_rates) if do_nothing_rates else 1.0),
        "train_avg_ships_sent": float(np.mean(ships_sent) if ships_sent else 0.0),
        "train_action_count": float(np.mean(action_counts) if action_counts else 0.0),
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

    curriculum_enabled = bool(cfg.get("opponent_curriculum_enabled", True))
    tiers = _curriculum_tiers(cfg)
    curriculum_state_path = _curriculum_state_path(cfg, log_dir)
    curriculum_state = _load_curriculum_state(curriculum_state_path, tiers, cfg, resume=resume)
    current_tier = _current_tier(curriculum_state, tiers) if curriculum_enabled else {"name": "notebook_open", "label": "full notebook pool", "opponents": "notebook_pool"}
    pool = _tier_pool(cfg, current_tier) if curriculum_enabled else _notebook_pool(int(cfg["notebook_pool_limit"]))
    workers = max(1, int(cfg.get("workers", 6)))
    duration_seconds = max(1.0, float(cfg["duration_minutes"]) * 60.0)
    eval_episodes = max(4, int(cfg.get("eval_episodes", 16)))
    candidate_eval_default = min(eval_episodes, int(current_tier.get("candidate_eval_episodes", cfg.get("candidate_eval_episodes", 8))))
    candidate_eval_episodes = max(4, int(cfg.get("candidate_eval_episodes", candidate_eval_default)))
    promotion_eval_episodes = max(eval_episodes, int(cfg.get("promotion_eval_episodes", eval_episodes)))
    promotion_margin = max(0.0, float(cfg.get("promotion_margin", 0.0)))
    promotion_min_remaining_seconds = max(0.0, float(cfg.get("promotion_min_remaining_minutes", 0.0)) * 60.0)
    started_at = time.time()
    deadline_epoch = started_at + duration_seconds
    generation = int(curriculum_state.get("total_generations", 0)) if resume else 0
    curriculum_state["total_generations"] = generation
    base_checkpoint = _checkpoint_to_load(cfg, resume)
    global_best_score = -1e9
    tier_best_score = -1e9
    best_record: Dict[str, Any] = {}

    if base_checkpoint and base_checkpoint.exists():
        _, metadata = load_checkpoint(base_checkpoint)
        best_record = dict(metadata)
        try:
            global_best_score = float(metadata.get("score", metadata.get("composite_score", metadata.get("winrate", metadata.get("best_score", -1e9)))))
        except (TypeError, ValueError):
            global_best_score = -1e9

    tier_best_score = _best_for_tier(curriculum_state, str(current_tier.get("name")))
    if tier_best_score < -1e8 and best_record:
        checkpoint_tier = best_record.get("curriculum_tier")
        if checkpoint_tier == current_tier.get("name"):
            try:
                tier_best_score = float(best_record.get("score", best_record.get("composite_score", best_record.get("winrate", -1e9))))
            except (TypeError, ValueError):
                tier_best_score = -1e9

    probe = NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(cfg), hidden_dim=int(cfg["hidden_dim"])))
    parameter_count = count_parameters(probe)
    _write_curriculum_state(curriculum_state_path, curriculum_state, current_tier, pool, global_best_score, best_record)

    while time.time() < deadline_epoch:
        elapsed = time.time() - started_at
        current_tier = _current_tier(curriculum_state, tiers) if curriculum_enabled else current_tier
        pool = _tier_pool(cfg, current_tier) if curriculum_enabled else pool
        tier_name = str(current_tier.get("name"))
        tier_best_score = _best_for_tier(curriculum_state, tier_name) if curriculum_enabled else global_best_score
        tier_eval_default = min(eval_episodes, int(current_tier.get("candidate_eval_episodes", candidate_eval_episodes)))
        active_candidate_eval_episodes = max(4, int(cfg.get("candidate_eval_episodes", tier_eval_default)))
        print(
            f"generation {generation} start elapsed={elapsed/60.0:.1f}m "
            f"global_best={global_best_score:.3f} tier_best={tier_best_score:.3f} tier={tier_name} pool={len(pool)} "
            f"workers={workers} eval_fast={active_candidate_eval_episodes}",
            flush=True,
        )
        if time.time() >= deadline_epoch:
            break
        checkpoint_str = str(base_checkpoint) if base_checkpoint else None
        task_config = dict(cfg)
        task_config["curriculum_tier"] = current_tier.get("name")
        task_config["curriculum_tier_label"] = current_tier.get("label", current_tier.get("name"))
        tasks = [
            {
                "config": task_config,
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
                "config": task_config,
                "pool": pool,
                "episodes": active_candidate_eval_episodes,
                "seed_offset": int(cfg["seed"]) + 50000 + generation * 1000 + int(candidate["worker_id"]) * 100,
                "deadline_epoch": deadline_epoch,
            }
            for candidate in candidates
        ]
        eval_results = _run_parallel(eval_tasks, workers, _worker_evaluate_candidate, f"generation {generation} fast_eval", started_at, deadline_epoch)
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
            record_for_score = {**eval_stats}
            score = _composite_score(record_for_score)
            record = {
                "generation": generation,
                "worker_id": int(candidate["worker_id"]),
                "parameter_count": parameter_count,
                "curriculum_tier": current_tier.get("name"),
                "curriculum_tier_label": current_tier.get("label", current_tier.get("name")),
                "tier_generation": int(curriculum_state.get("tier_generation", 0)),
                "pool_size": len(pool),
                "pool": pool,
                "train_reward_mean": float(candidate["train_reward_mean"]),
                "train_winrate": float(candidate["train_winrate"]),
                "train_rank_mean": float(candidate.get("train_rank_mean", 4.0)),
                "train_do_nothing_rate": float(candidate.get("train_do_nothing_rate", 1.0)),
                "train_avg_ships_sent": float(candidate.get("train_avg_ships_sent", 0.0)),
                "train_action_count": float(candidate.get("train_action_count", 0.0)),
                **eval_stats,
                "candidate_eval_episodes": active_candidate_eval_episodes,
                "eval_phase": "fast",
                "score": score,
                "composite_score": score,
                "checkpoint_promoted": False,
                "promotion_reason": "no promotion",
                "worker_last_record": candidate["last_record"],
            }
            evaluated.append({"record": record, "state": candidate["state"]})
        if not evaluated:
            print("hard stop: no candidate had completed eval record before deadline", flush=True)
            break

        evaluated.sort(key=lambda item: _score_record(item["record"]), reverse=True)
        generation_best = evaluated[0]
        save_checkpoint(candidate_path, generation_best["state"], generation_best["record"])
        save_checkpoint(latest_path, generation_best["state"], generation_best["record"])

        should_promote = _should_try_promotion(generation_best["record"], tier_best_score, promotion_margin)
        enough_time_for_confirmation = deadline_epoch - time.time() >= promotion_min_remaining_seconds
        if should_promote and enough_time_for_confirmation:
            preliminary_record = dict(generation_best["record"])
            confirmed_stats = _evaluate_candidate(
                generation_best["state"],
                cfg,
                pool,
                episodes=promotion_eval_episodes,
                seed_offset=int(cfg["seed"]) + 900000 + generation * 1000,
            )
            preliminary_score = float(preliminary_record["score"])
            generation_best["record"].update(confirmed_stats)
            confirmed_score = _composite_score(generation_best["record"])
            generation_best["record"]["score"] = confirmed_score
            generation_best["record"]["composite_score"] = confirmed_score
            generation_best["record"]["preliminary_score"] = preliminary_score
            generation_best["record"]["preliminary_winrate"] = float(preliminary_record["winrate"])
            generation_best["record"]["promotion_eval_episodes"] = promotion_eval_episodes
            generation_best["record"]["eval_phase"] = "confirmed"

            promoted_score = float(generation_best["record"]["score"])
            if _should_try_promotion(generation_best["record"], tier_best_score, promotion_margin):
                tier_best_score = promoted_score
                curriculum_state.setdefault("tier_best_scores", {})[tier_name] = tier_best_score
                generation_best["record"]["checkpoint_promoted"] = True
                generation_best["record"]["promotion_reason"] = f"confirmed best composite score {tier_best_score:.4f}"
                if promoted_score > global_best_score:
                    global_best_score = promoted_score
                    best_record = dict(generation_best["record"])
                    save_checkpoint(best_path, generation_best["state"], generation_best["record"])
                    base_checkpoint = best_path
                else:
                    base_checkpoint = latest_path if latest_path.exists() else base_checkpoint
            else:
                generation_best["record"]["checkpoint_promoted"] = False
                generation_best["record"]["promotion_reason"] = (
                    f"promotion rejected after confirmation "
                    f"score {generation_best['record']['score']:.4f} < required {tier_best_score + promotion_margin:.4f}"
                )
                base_checkpoint = best_path if best_path.exists() else latest_path
        else:
            if should_promote:
                generation_best["record"]["promotion_reason"] = "promotion skipped: not enough time for confirmation eval"
            base_checkpoint = best_path if best_path.exists() else latest_path

        curriculum_state["total_generations"] = generation
        curriculum_state["tier_generation"] = int(curriculum_state.get("tier_generation", 0)) + 1
        advanced = False
        advance_reason = "curriculum disabled"
        if curriculum_enabled:
            advanced, advance_reason = _maybe_advance_curriculum(curriculum_state, tiers, generation_best["record"])
            if advanced:
                next_tier = _current_tier(curriculum_state, tiers)
                next_tier_name = str(next_tier.get("name"))
                tier_best_score = _best_for_tier(curriculum_state, next_tier_name)
                current_tier = next_tier
                pool = _tier_pool(cfg, current_tier)
                base_checkpoint = best_path if best_path.exists() else latest_path
        curriculum_state["total_generations"] = generation + 1
        generation_best["record"]["curriculum_advanced"] = advanced
        generation_best["record"]["curriculum_reason"] = advance_reason
        generation_best["record"]["next_curriculum_tier"] = current_tier.get("name")

        for item in evaluated:
            if item is generation_best:
                item["record"].update(generation_best["record"])
            append_jsonl(log_path, item["record"])

        _write_curriculum_state(curriculum_state_path, curriculum_state, current_tier, pool, global_best_score, best_record, generation_best["record"])

        print(
            f"generation {generation} candidate_score={generation_best['record']['score']:.3f} "
            f"best_winrate={generation_best['record'].get('winrate', 0.0):.3f} "
            f"global_best={global_best_score:.3f} tier_best={tier_best_score:.3f} tier={current_tier.get('name')} "
            f"params={parameter_count} pool={len(pool)}",
            flush=True,
        )
        generation += 1

    return {
        "best_score": global_best_score,
        "best": str(best_path),
        "latest": str(latest_path),
        "candidate": str(candidate_path),
        "log": str(log_path),
        "generations": generation,
        "workers": workers,
        "parameter_count": parameter_count,
        "pool": pool,
        "curriculum_state": str(curriculum_state_path),
        "curriculum_tier": current_tier.get("name"),
        "curriculum_tier_label": current_tier.get("label", current_tier.get("name")),
        "best_record": best_record,
    }
