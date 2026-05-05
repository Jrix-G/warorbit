from __future__ import annotations

import json
import multiprocessing as mp
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from SimGame import SimGame

from .encoder import encode_game_state
from .model import ModelConfig, NeuralNetworkModel, count_parameters, load_compatible_state_dict
from .notebook_4p_training import (
    _action_summary,
    _agent_for_name,
    _build_agents,
    _candidate_move,
    _copy_planning_game,
    _episode_reward,
    _infer_input_dim,
    _reserve_planned_ships,
    _send_ratios,
    _train_episode,
    evaluate_4p,
    run_match,
    training_pool,
)
from .orbit_wars_adapter import obs_to_game_dict
from .policy import build_action_candidates, reconstruct_action
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
        "name": "notebook_core4",
        "label": "core notebooks plus heuristics",
        "opponents": [
            "distance",
            "sun_dodge",
            "structured",
            "orbit_stars",
            "notebook_orbitbotnext",
            "notebook_distance_prioritized",
            "notebook_physics_accurate",
            "notebook_tactical_heuristic",
        ],
        "min_generations": 4,
        "advance_score": 0.44,
        "advance_winrate": 0.24,
        "advance_rank_mean": 3.00,
        "advance_do_nothing_rate": 0.50,
        "candidate_eval_episodes": 8,
    },
    {
        "name": "notebook_mid8",
        "label": "first eight notebook opponents",
        "opponents": "notebook_pool:8",
        "min_generations": 5,
        "advance_score": 0.38,
        "advance_winrate": 0.18,
        "advance_rank_mean": 3.10,
        "advance_do_nothing_rate": 0.50,
        "candidate_eval_episodes": 8,
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


def _checkpoint_score(metadata: Dict[str, Any]) -> float:
    try:
        value = metadata.get(
            "score",
            metadata.get("composite_score", metadata.get("winrate", metadata.get("best_score", -1e9))),
        )
        return float(value)
    except (TypeError, ValueError):
        return -1e9


def _safe_checkpoint_stem(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(name))
    return safe.strip("._") or "tier"


def _tier_checkpoint_dir(config: Dict[str, Any], checkpoint_dir: Path) -> Path:
    value = config.get("tier_checkpoint_dir")
    return Path(value) if value else checkpoint_dir / "tiers"


def _tier_best_checkpoint_path(config: Dict[str, Any], checkpoint_dir: Path, tier_name: str) -> Path:
    tier_paths = config.get("tier_best_checkpoints")
    if isinstance(tier_paths, dict) and tier_name in tier_paths:
        return Path(tier_paths[tier_name])
    return _tier_checkpoint_dir(config, checkpoint_dir) / f"{_safe_checkpoint_stem(tier_name)}.npz"


def _training_base_checkpoint(
    config: Dict[str, Any],
    resume: bool,
    checkpoint_dir: Path,
    tier_name: str,
    fallback: Path | None,
) -> Path | None:
    if resume and bool(config.get("resume_from_tier_best", True)):
        tier_path = _tier_best_checkpoint_path(config, checkpoint_dir, tier_name)
        if tier_path.exists():
            return tier_path
    return fallback


def _next_base_checkpoint(
    config: Dict[str, Any],
    resume: bool,
    tier_path: Path,
    current_base: Path | None,
    fallback: Path | None,
) -> Path | None:
    if bool(config.get("resume_from_tier_best", True)) and tier_path.exists():
        if resume or current_base == tier_path:
            return tier_path
    return fallback


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _composite_score(record: Dict[str, Any]) -> float:
    metric_keys = ("winrate", "rank_mean", "eval_mean", "eval_do_nothing_rate", "avg_score", "eval_avg_ships_sent")
    if "score" in record and not any(key in record for key in metric_keys):
        return float(record["score"])
    if "composite_score" in record and not any(key in record for key in metric_keys):
        return float(record["composite_score"])
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
        float(record.get("winrate", 0.0)),
        score,
        float(record.get("eval_mean", 0.0)),
        -float(record.get("rank_mean", 4.0)),
        float(record.get("avg_score", 0.0)),
        -float(record.get("eval_do_nothing_rate", 1.0)),
    )


def _player_snapshot(game: Dict[str, Any], player_id: int) -> Dict[str, float]:
    planets = game.get("planets", [])
    fleets = game.get("fleets", [])
    my_planets = [p for p in planets if int(p.get("owner", -1)) == int(player_id)]
    my_fleets = [f for f in fleets if int(f.get("owner", -1)) == int(player_id)]
    total_ships = sum(float(p.get("ships", 0.0)) for p in planets) + sum(float(f.get("ships", 0.0)) for f in fleets)
    my_ships = sum(float(p.get("ships", 0.0)) for p in my_planets) + sum(float(f.get("ships", 0.0)) for f in my_fleets)
    return {
        "planets": float(len(my_planets)),
        "production": float(sum(float(p.get("production", 0.0)) for p in my_planets)),
        "ships": my_ships,
        "ship_share": my_ships / max(1.0, total_ships),
        "alive": 1.0 if my_planets or my_fleets else 0.0,
    }


def _strategic_dense_reward(result: Dict[str, Any], our_index: int, config: Dict[str, Any]) -> float:
    initial_obs = result.get("initial_state")
    final_obs = result.get("final_state")
    if initial_obs is None or final_obs is None:
        return 0.0
    initial = obs_to_game_dict(initial_obs)
    final = obs_to_game_dict(final_obs)
    start = _player_snapshot(initial, our_index)
    end = _player_snapshot(final, our_index)
    scores = [float(v) for v in result.get("scores", [])]
    our_score = scores[our_index] if len(scores) > our_index else 0.0
    other_scores = [score for idx, score in enumerate(scores) if idx != our_index]
    score_advantage = (our_score - float(np.mean(other_scores) if other_scores else 0.0)) / 1000.0
    reward = (
        float(config.get("dense_planet_coef", 0.04)) * (end["planets"] - start["planets"])
        + float(config.get("dense_production_coef", 0.03)) * (end["production"] - start["production"])
        + float(config.get("dense_ship_share_coef", 0.12)) * (end["ship_share"] - start["ship_share"])
        + float(config.get("dense_score_coef", 0.08)) * max(-1.0, min(1.0, score_advantage))
        + float(config.get("dense_survival_coef", 0.05)) * (2.0 * end["alive"] - 1.0)
    )
    limit = float(config.get("dense_reward_clip", 0.35))
    return float(max(-limit, min(limit, reward)))


def _angle_distance(a: float, b: float) -> float:
    return abs((float(a) - float(b) + np.pi) % (2.0 * np.pi) - np.pi)


def _teacher_action_index(candidates, planning_game: Dict[str, Any], teacher_move: Sequence[Any]) -> int | None:
    if not isinstance(teacher_move, (list, tuple)) or len(teacher_move) < 3:
        return None
    try:
        src_id = int(teacher_move[0])
        teacher_angle = float(teacher_move[1])
        teacher_ships = max(1.0, float(teacher_move[2]))
    except (TypeError, ValueError):
        return None
    best_idx: int | None = None
    best_score = float("inf")
    for idx, candidate in enumerate(candidates):
        if candidate.mission == "do_nothing" or int(candidate.source_id) != src_id:
            continue
        action = reconstruct_action(candidate, planning_game)
        move = _candidate_move(planning_game, action)
        if not move:
            continue
        angle = float(move[0][1])
        ships = max(1.0, float(move[0][2]))
        score = _angle_distance(angle, teacher_angle) + 0.15 * abs(np.log(ships / teacher_ships))
        if score < best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _train_imitation_observation(
    model: NeuralNetworkModel,
    optimizer: torch.optim.Optimizer,
    obs: Any,
    teacher_moves: Sequence[Any],
    config: Dict[str, Any],
) -> Dict[str, float]:
    if not teacher_moves:
        return {"loss": 0.0, "targets": 0.0}
    ratios = _send_ratios(config)
    planning_game = _copy_planning_game(obs_to_game_dict(obs))
    losses: List[torch.Tensor] = []
    max_actions = max(1, int(config.get("max_actions_per_turn", 4)))
    for teacher_move in list(teacher_moves)[:max_actions]:
        encoded = encode_game_state(planning_game, config)
        candidates = build_action_candidates(planning_game, send_ratios=ratios)
        target_idx = _teacher_action_index(candidates, planning_game, teacher_move)
        if target_idx is None:
            continue
        candidate_features = np.stack([c.score_features for c in candidates]).astype(np.float32)
        outputs = model(
            torch.tensor(encoded.features, dtype=torch.float32),
            torch.tensor(candidate_features, dtype=torch.float32),
        )
        logits = outputs["policy_logits"]
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        target = torch.tensor([target_idx], dtype=torch.long, device=logits.device)
        losses.append(F.cross_entropy(logits, target))
        action = reconstruct_action(candidates[target_idx], planning_game)
        move = _candidate_move(planning_game, action)
        if move:
            _reserve_planned_ships(planning_game, action[0], int(move[0][2]))
    if not losses:
        return {"loss": 0.0, "targets": 0.0}
    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item())
    optimizer.step()
    return {"loss": float(loss.item()), "targets": float(len(losses)), "grad_norm": grad_norm}


def _run_imitation_warmstart(
    model: NeuralNetworkModel,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    pool: Sequence[str],
    seed: int,
) -> Dict[str, float]:
    steps = max(0, int(config.get("imitation_warmstart_steps", 0)))
    if steps <= 0:
        return {"imitation_loss": 0.0, "imitation_targets": 0.0, "imitation_updates": 0.0}
    teacher_pool = [name for name in pool if str(name).startswith("notebook_")]
    if not teacher_pool:
        teacher_pool = list(pool)
    losses: List[float] = []
    targets: List[float] = []
    for idx in range(steps):
        teacher_name = teacher_pool[idx % len(teacher_pool)] if teacher_pool else "greedy"
        teacher = _agent_for_name(teacher_name)
        player = idx % 4
        game = SimGame.random_game(
            seed=seed + idx * 7919,
            n_players=4,
            neutral_pairs=8,
            max_steps=int(config.get("max_turns", 100)),
            overage_time=60.0,
        )
        obs = game.observation(player)
        try:
            teacher_moves = teacher(obs, None)
        except TypeError:
            teacher_moves = teacher(obs)
        if not isinstance(teacher_moves, list):
            teacher_moves = []
        metrics = _train_imitation_observation(model, optimizer, obs, teacher_moves, config)
        if metrics["targets"] > 0:
            losses.append(metrics["loss"])
            targets.append(metrics["targets"])
    return {
        "imitation_loss": float(np.mean(losses) if losses else 0.0),
        "imitation_targets": float(np.sum(targets) if targets else 0.0),
        "imitation_updates": float(len(losses)),
    }


def _promotion_min_winrate(tier: Dict[str, Any], config: Dict[str, Any]) -> float:
    if "promotion_min_winrate" in config:
        return max(0.0, min(1.0, float(config["promotion_min_winrate"])))
    advance_winrate = float(tier.get("advance_winrate", 0.0))
    return max(0.125, min(0.35, advance_winrate * 0.5))


def _should_try_promotion(record: Dict[str, Any], best_score: float, margin: float, min_winrate: float = 0.0) -> bool:
    if float(record.get("winrate", 0.0)) < float(min_winrate):
        return False
    if best_score < -1e8:
        return True
    return _composite_score(record) >= best_score + margin


def _promote_candidate(
    state: Dict[str, Any],
    record: Dict[str, Any],
    *,
    promoted_score: float,
    tier_name: str,
    tier_checkpoint_path: Path,
    best_path: Path,
    global_best_score: float,
    curriculum_state: Dict[str, Any],
) -> tuple[float, Dict[str, Any]]:
    curriculum_state.setdefault("tier_best_scores", {})[tier_name] = promoted_score
    record["checkpoint_promoted"] = True
    record["tier_best_checkpoint"] = str(tier_checkpoint_path)
    save_checkpoint(tier_checkpoint_path, state, record)
    best_record: Dict[str, Any] = {}
    if promoted_score > global_best_score:
        best_record = dict(record)
        save_checkpoint(best_path, state, record)
    return promoted_score, best_record


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
        if opponents.startswith("notebook_pool:"):
            try:
                return _notebook_pool(int(opponents.split(":", 1)[1]))
            except (TypeError, ValueError):
                return _notebook_pool(int(config.get("notebook_pool_limit", 15)))
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
    tier_best_checkpoint: str | None = None,
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
            "tier_best_checkpoint": tier_best_checkpoint or "",
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
    imitation_metrics = _run_imitation_warmstart(model, optimizer, config, pool, seed)
    baseline = float(config.get("moving_average_baseline", 0.0))
    rewards: List[float] = []
    dense_rewards: List[float] = []
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
        result = run_match(
            agents,
            seed=episode_seed,
            n_players=4,
            max_steps=int(config.get("max_turns", 100)),
            stop_player=stop_player,
            game_engine=str(config.get("game_engine", config.get("match_runner", "simgame"))),
            use_c_accel=bool(config.get("official_fast_c_accel", True)),
        )
        terminal_reward = _episode_reward(result, our_index)
        dense_reward = _strategic_dense_reward(result, our_index, config) if bool(config.get("dense_reward_enabled", True)) else 0.0
        reward = terminal_reward + dense_reward
        action_metrics = _action_summary(action_records)
        scores = result.get("scores", [])
        ordered = sorted(((float(score), idx) for idx, score in enumerate(scores)), reverse=True)
        rank = next((rank for rank, (_, idx) in enumerate(ordered, start=1) if idx == our_index), 4)
        baseline_rate = max(0.0, min(1.0, float(config.get("baseline_momentum", 0.05))))
        baseline = (1.0 - baseline_rate) * baseline + baseline_rate * reward
        entropy_coef = float(config.get("entropy_coef_start", 0.03)) + (
            float(config.get("entropy_coef_end", 0.005)) - float(config.get("entropy_coef_start", 0.03))
        ) * min(1.0, max(0.0, progress))
        train_metrics = _train_episode(
            model,
            optimizer,
            log_probs,
            reward,
            baseline,
            entropy_coef=entropy_coef,
            action_records=action_records,
            value_coef=float(config.get("value_loss_coef", 0.25)),
        )
        rewards.append(reward)
        dense_rewards.append(dense_reward)
        wins.append(1.0 if int(result.get("winner", -1)) == our_index else 0.0)
        ranks.append(float(rank))
        do_nothing_rates.append(float(action_metrics["do_nothing_rate"]))
        ships_sent.append(float(action_metrics["avg_ships_sent"]))
        action_counts.append(float(action_metrics["action_count"]))
        last_record = {
            "worker_id": worker_id,
            "local_step": local_step,
            "reward": reward,
            "terminal_reward": terminal_reward,
            "dense_reward": dense_reward,
            "baseline": baseline,
            "temperature": temperature,
            "entropy_coef": entropy_coef,
            "grad_norm": train_metrics["grad_norm"],
            "loss": train_metrics["loss"],
            "policy_loss": train_metrics.get("policy_loss", 0.0),
            "value_loss": train_metrics.get("value_loss", 0.0),
            "policy_entropy": train_metrics.get("policy_entropy", 0.0),
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
        "train_dense_reward_mean": float(np.mean(dense_rewards) if dense_rewards else 0.0),
        "train_winrate": float(np.mean(wins) if wins else 0.0),
        "train_rank_mean": float(np.mean(ranks) if ranks else 4.0),
        "train_do_nothing_rate": float(np.mean(do_nothing_rates) if do_nothing_rates else 1.0),
        "train_avg_ships_sent": float(np.mean(ships_sent) if ships_sent else 0.0),
        "train_action_count": float(np.mean(action_counts) if action_counts else 0.0),
        **imitation_metrics,
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
                    except Exception as exc:
                        print(f"{label} task_failed {type(exc).__name__}: {exc}", flush=True)
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
    fallback_checkpoint = _checkpoint_to_load(cfg, resume)
    global_best_score = -1e9
    tier_best_score = -1e9
    best_record: Dict[str, Any] = {}

    score_checkpoint = best_path if resume and best_path.exists() else fallback_checkpoint
    if score_checkpoint and score_checkpoint.exists():
        _, metadata = load_checkpoint(score_checkpoint)
        best_record = dict(metadata)
        global_best_score = _checkpoint_score(metadata)

    base_checkpoint = _training_base_checkpoint(
        cfg,
        resume,
        checkpoint_dir,
        str(current_tier.get("name")),
        fallback_checkpoint,
    )

    tier_best_score = _best_for_tier(curriculum_state, str(current_tier.get("name")))
    tier_checkpoint_path = _tier_best_checkpoint_path(cfg, checkpoint_dir, str(current_tier.get("name")))
    if tier_best_score < -1e8 and tier_checkpoint_path.exists():
        _, tier_metadata = load_checkpoint(tier_checkpoint_path)
        tier_best_score = _checkpoint_score(tier_metadata)
        curriculum_state.setdefault("tier_best_scores", {})[str(current_tier.get("name"))] = tier_best_score
    if tier_best_score < -1e8 and best_record:
        checkpoint_tier = best_record.get("curriculum_tier")
        if checkpoint_tier == current_tier.get("name"):
            tier_best_score = _checkpoint_score(best_record)

    probe = NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(cfg), hidden_dim=int(cfg["hidden_dim"])))
    parameter_count = count_parameters(probe)
    _write_curriculum_state(
        curriculum_state_path,
        curriculum_state,
        current_tier,
        pool,
        global_best_score,
        best_record,
        tier_best_checkpoint=str(tier_checkpoint_path),
    )

    while time.time() < deadline_epoch:
        elapsed = time.time() - started_at
        remaining_seconds = deadline_epoch - time.time()
        min_generation_seconds = max(0.0, float(cfg.get("min_generation_remaining_minutes", 0.0)) * 60.0)
        if min_generation_seconds > 0.0 and remaining_seconds < min_generation_seconds:
            print(
                f"stopping before generation {generation}: remaining={remaining_seconds/60.0:.1f}m "
                f"< min_generation_remaining={min_generation_seconds/60.0:.1f}m",
                flush=True,
            )
            break
        current_tier = _current_tier(curriculum_state, tiers) if curriculum_enabled else current_tier
        pool = _tier_pool(cfg, current_tier) if curriculum_enabled else pool
        tier_name = str(current_tier.get("name"))
        tier_best_score = _best_for_tier(curriculum_state, tier_name) if curriculum_enabled else global_best_score
        tier_checkpoint_path = _tier_best_checkpoint_path(cfg, checkpoint_dir, tier_name)
        promotion_min_winrate = _promotion_min_winrate(current_tier, cfg)
        if curriculum_enabled and tier_best_score < -1e8 and tier_checkpoint_path.exists():
            _, tier_metadata = load_checkpoint(tier_checkpoint_path)
            tier_best_score = _checkpoint_score(tier_metadata)
            curriculum_state.setdefault("tier_best_scores", {})[tier_name] = tier_best_score
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
        base_checkpoint = _training_base_checkpoint(cfg, resume, checkpoint_dir, tier_name, base_checkpoint)
        checkpoint_str = str(base_checkpoint) if base_checkpoint else None
        tier_checkpoint_loaded = bool(checkpoint_str and Path(checkpoint_str) == tier_checkpoint_path)
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
                "evaluated_tier": tier_name,
                "curriculum_tier_label": current_tier.get("label", current_tier.get("name")),
                "tier_generation": int(curriculum_state.get("tier_generation", 0)),
                "tier_best_checkpoint": str(tier_checkpoint_path),
                "tier_checkpoint_loaded": tier_checkpoint_loaded,
                "base_checkpoint": checkpoint_str or "",
                "global_best_before": float(global_best_score),
                "tier_best_before": float(tier_best_score),
                "pool_size": len(pool),
                "pool": pool,
                "pool_names": list(pool),
                "train_reward_mean": float(candidate["train_reward_mean"]),
                "train_dense_reward_mean": float(candidate.get("train_dense_reward_mean", 0.0)),
                "train_winrate": float(candidate["train_winrate"]),
                "train_rank_mean": float(candidate.get("train_rank_mean", 4.0)),
                "train_do_nothing_rate": float(candidate.get("train_do_nothing_rate", 1.0)),
                "train_avg_ships_sent": float(candidate.get("train_avg_ships_sent", 0.0)),
                "train_action_count": float(candidate.get("train_action_count", 0.0)),
                "imitation_loss": float(candidate.get("imitation_loss", 0.0)),
                "imitation_targets": float(candidate.get("imitation_targets", 0.0)),
                "imitation_updates": float(candidate.get("imitation_updates", 0.0)),
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

        should_promote = _should_try_promotion(
            generation_best["record"],
            tier_best_score,
            promotion_margin,
            min_winrate=promotion_min_winrate,
        )
        enough_time_for_confirmation = deadline_epoch - time.time() >= promotion_min_remaining_seconds
        can_bootstrap_without_confirmation = (
            bool(cfg.get("bootstrap_promote_without_confirmation", True))
            and should_promote
            and not enough_time_for_confirmation
            and tier_best_score < -1e8
        )
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
            if _should_try_promotion(
                generation_best["record"],
                tier_best_score,
                promotion_margin,
                min_winrate=promotion_min_winrate,
            ):
                generation_best["record"]["promotion_reason"] = (
                    f"confirmed best score {promoted_score:.4f} with winrate {generation_best['record'].get('winrate', 0.0):.4f}"
                )
                tier_best_score, maybe_best_record = _promote_candidate(
                    generation_best["state"],
                    generation_best["record"],
                    promoted_score=promoted_score,
                    tier_name=tier_name,
                    tier_checkpoint_path=tier_checkpoint_path,
                    best_path=best_path,
                    global_best_score=global_best_score,
                    curriculum_state=curriculum_state,
                )
                if maybe_best_record:
                    global_best_score = promoted_score
                    best_record = maybe_best_record
                base_checkpoint = tier_checkpoint_path
            else:
                generation_best["record"]["checkpoint_promoted"] = False
                generation_best["record"]["promotion_reason"] = (
                    f"promotion rejected after confirmation "
                    f"score={generation_best['record']['score']:.4f} "
                    f"winrate={generation_best['record'].get('winrate', 0.0):.4f} "
                    f"required_score={tier_best_score + promotion_margin:.4f} "
                    f"required_winrate={promotion_min_winrate:.4f}"
                )
                fallback_base = best_path if best_path.exists() else latest_path
                base_checkpoint = _next_base_checkpoint(cfg, resume, tier_checkpoint_path, base_checkpoint, fallback_base)
        else:
            if can_bootstrap_without_confirmation:
                promoted_score = float(generation_best["record"]["score"])
                generation_best["record"]["eval_phase"] = "fast_bootstrap"
                generation_best["record"]["promotion_reason"] = (
                    "bootstrap best from fast eval: no previous tier best and not enough time for confirmation eval"
                )
                tier_best_score, maybe_best_record = _promote_candidate(
                    generation_best["state"],
                    generation_best["record"],
                    promoted_score=promoted_score,
                    tier_name=tier_name,
                    tier_checkpoint_path=tier_checkpoint_path,
                    best_path=best_path,
                    global_best_score=global_best_score,
                    curriculum_state=curriculum_state,
                )
                if maybe_best_record:
                    global_best_score = promoted_score
                    best_record = maybe_best_record
                base_checkpoint = tier_checkpoint_path
            elif should_promote:
                generation_best["record"]["promotion_reason"] = "promotion skipped: not enough time for confirmation eval"
                fallback_base = best_path if best_path.exists() else latest_path
                base_checkpoint = _next_base_checkpoint(cfg, resume, tier_checkpoint_path, base_checkpoint, fallback_base)
            else:
                if float(generation_best["record"].get("winrate", 0.0)) < promotion_min_winrate:
                    generation_best["record"]["promotion_reason"] = (
                        f"promotion gated by low winrate {generation_best['record'].get('winrate', 0.0):.4f} < {promotion_min_winrate:.4f}"
                    )
                fallback_base = best_path if best_path.exists() else latest_path
                base_checkpoint = _next_base_checkpoint(cfg, resume, tier_checkpoint_path, base_checkpoint, fallback_base)

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
                next_tier_checkpoint = _tier_best_checkpoint_path(cfg, checkpoint_dir, next_tier_name)
                if resume and bool(cfg.get("resume_from_tier_best", True)) and next_tier_checkpoint.exists():
                    base_checkpoint = next_tier_checkpoint
                else:
                    fallback_base = best_path if best_path.exists() else latest_path
                    base_checkpoint = _next_base_checkpoint(cfg, resume, tier_checkpoint_path, base_checkpoint, fallback_base)
        curriculum_state["total_generations"] = generation + 1
        generation_best["record"]["curriculum_advanced"] = advanced
        generation_best["record"]["curriculum_reason"] = advance_reason
        generation_best["record"]["next_curriculum_tier"] = current_tier.get("name")
        save_checkpoint(candidate_path, generation_best["state"], generation_best["record"])
        save_checkpoint(latest_path, generation_best["state"], generation_best["record"])

        for item in evaluated:
            if item is generation_best:
                item["record"].update(generation_best["record"])
            append_jsonl(log_path, item["record"])

        current_tier_checkpoint_path = _tier_best_checkpoint_path(cfg, checkpoint_dir, str(current_tier.get("name")))
        _write_curriculum_state(
            curriculum_state_path,
            curriculum_state,
            current_tier,
            pool,
            global_best_score,
            best_record,
            generation_best["record"],
            tier_best_checkpoint=str(current_tier_checkpoint_path),
        )

        print(
            f"generation {generation} candidate_score={generation_best['record']['score']:.3f} "
            f"winrate={generation_best['record'].get('winrate', 0.0):.3f} "
            f"rank={generation_best['record'].get('rank_mean', 4.0):.2f} "
            f"noop={generation_best['record'].get('eval_do_nothing_rate', 1.0):.2f} "
            f"promoted={int(bool(generation_best['record'].get('checkpoint_promoted', False)))} "
            f"global_best={global_best_score:.3f} tier_best={tier_best_score:.3f} "
            f"eval_tier={tier_name} next_tier={current_tier.get('name')} "
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
        "tier_checkpoint_dir": str(_tier_checkpoint_dir(cfg, checkpoint_dir)),
        "tier_best_checkpoint": str(_tier_best_checkpoint_path(cfg, checkpoint_dir, str(current_tier.get("name")))),
        "best_record": best_record,
    }
