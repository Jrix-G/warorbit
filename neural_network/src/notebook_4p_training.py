from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from SimGame import SimGame

from .encoder import encode_game_state
from .model import ModelConfig, NeuralNetworkModel, load_compatible_state_dict
from .orbit_wars_adapter import obs_to_game_dict
from .policy import build_action_candidates, choose_action, reconstruct_action
from .reward import compute_dense_reward
from .storage import append_jsonl, load_checkpoint, save_checkpoint
from .torch_compat import ensure_torch_dynamo_stub
from .trajectory import safe_plan_shot
from .utils import ensure_dir


ZOO: Dict[str, Callable] | None = None


def _obs_planets(obs: Any) -> List[list]:
    if isinstance(obs, dict):
        return list(obs.get("planets", []) or [])
    return list(getattr(obs, "planets", []) or [])


def _obs_player(obs: Any) -> int:
    if isinstance(obs, dict):
        return int(obs.get("player", 0) or 0)
    return int(getattr(obs, "player", 0) or 0)


def _passive_agent(obs, config=None):
    return []


def _random_agent(obs, config=None):
    planets = _obs_planets(obs)
    me = _obs_player(obs)
    moves = []
    for planet in planets:
        if int(planet[1]) == me and float(planet[5]) > 10:
            moves.append([int(planet[0]), random.uniform(0.0, 2.0 * math.pi), int(float(planet[5]) // 2)])
    return moves


def _greedy_agent(obs, config=None):
    planets = _obs_planets(obs)
    me = _obs_player(obs)
    my_planets = [p for p in planets if int(p[1]) == me]
    targets = [p for p in planets if int(p[1]) != me]
    moves = []
    for src in my_planets:
        if float(src[5]) < 10 or not targets:
            continue
        tgt = min(targets, key=lambda p: math.hypot(float(p[2]) - float(src[2]), float(p[3]) - float(src[3])))
        angle = math.atan2(float(tgt[3]) - float(src[3]), float(tgt[2]) - float(src[2]))
        moves.append([int(src[0]), angle, int(float(src[5]) // 2)])
    return moves


def _starter_agent(obs, config=None):
    planets = _obs_planets(obs)
    me = _obs_player(obs)
    static_targets = []
    for planet in planets:
        orbital_r = math.hypot(float(planet[2]) - 50.0, float(planet[3]) - 50.0)
        if orbital_r + float(planet[4]) >= 50.0 and int(planet[1]) != me:
            static_targets.append(planet)
    moves = []
    for src in planets:
        if int(src[1]) != me or float(src[5]) <= 0 or not static_targets:
            continue
        tgt = min(static_targets, key=lambda p: math.hypot(float(p[2]) - float(src[2]), float(p[3]) - float(src[3])))
        ships = int(float(src[5]) // 2)
        if ships >= 20:
            angle = math.atan2(float(tgt[3]) - float(src[3]), float(tgt[2]) - float(src[2]))
            moves.append([int(src[0]), angle, ships])
    return moves


def _segment_min_dist_to_sun(x1: float, y1: float, x2: float, y2: float) -> float:
    seg_dx = x2 - x1
    seg_dy = y2 - y1
    lsq = seg_dx * seg_dx + seg_dy * seg_dy
    if lsq < 1e-9:
        return math.hypot(x1 - 50.0, y1 - 50.0)
    t = max(0.0, min(1.0, ((50.0 - x1) * seg_dx + (50.0 - y1) * seg_dy) / lsq))
    return math.hypot(x1 + t * seg_dx - 50.0, y1 + t * seg_dy - 50.0)


def _distance_agent(obs, config=None):
    planets = _obs_planets(obs)
    me = _obs_player(obs)
    my_planets = [p for p in planets if int(p[1]) == me]
    moves = []
    for src in my_planets:
        if float(src[5]) < 12:
            continue
        candidates = []
        for tgt in planets:
            if int(tgt[1]) == me:
                continue
            distance = math.hypot(float(tgt[2]) - float(src[2]), float(tgt[3]) - float(src[3]))
            score = -distance - 0.5 * float(tgt[5]) + (8.0 if int(tgt[1]) == -1 else 0.0)
            candidates.append((score, tgt))
        if not candidates:
            continue
        _, tgt = max(candidates, key=lambda item: item[0])
        if float(src[5]) > float(tgt[5]) + 5.0:
            angle = math.atan2(float(tgt[3]) - float(src[3]), float(tgt[2]) - float(src[2]))
            moves.append([int(src[0]), angle, int(float(src[5]) * 0.6)])
    return moves


def _sun_dodge_agent(obs, config=None):
    planets = _obs_planets(obs)
    me = _obs_player(obs)
    my_planets = [p for p in planets if int(p[1]) == me]
    targets = [p for p in planets if int(p[1]) != me]
    moves = []
    for src in my_planets:
        if float(src[5]) < 10 or not targets:
            continue
        tgt = min(targets, key=lambda p: math.hypot(float(p[2]) - float(src[2]), float(p[3]) - float(src[3])))
        dx = float(tgt[2]) - float(src[2])
        dy = float(tgt[3]) - float(src[3])
        distance = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        if _segment_min_dist_to_sun(float(src[2]), float(src[3]), float(tgt[2]), float(tgt[3])) < 11.5:
            for delta in (0.3, -0.3, 0.6, -0.6, 0.9, -0.9):
                candidate = angle + delta
                end_x = float(src[2]) + math.cos(candidate) * distance
                end_y = float(src[3]) + math.sin(candidate) * distance
                if _segment_min_dist_to_sun(float(src[2]), float(src[3]), end_x, end_y) > 11.0:
                    angle = candidate
                    break
        moves.append([int(src[0]), angle, int(float(src[5]) // 2)])
    return moves


def _orbit_stars_agent(obs, config=None):
    moves = _sun_dodge_agent(obs, config)
    return moves if moves else _distance_agent(obs, config)


LOCAL_ZOO: Dict[str, Callable] = {
    "passive": _passive_agent,
    "random": _random_agent,
    "greedy": _greedy_agent,
    "starter": _starter_agent,
    "distance": _distance_agent,
    "sun_dodge": _sun_dodge_agent,
    "structured": _distance_agent,
    "orbit_stars": _orbit_stars_agent,
}


def _external_zoo() -> Dict[str, Callable]:
    global ZOO
    if ZOO is None:
        from opponents import ZOO as external_zoo

        ZOO = external_zoo
    return ZOO


def training_pool(limit: int = 15) -> List[str]:
    from opponents import training_pool as external_training_pool

    return external_training_pool(limit=limit)


def _notebook_names() -> List[str]:
    return [name for name in sorted(_external_zoo()) if name.startswith("notebook_")]


def _known_opponent_names(pool: Sequence[str]) -> List[str]:
    names = []
    external: Dict[str, Callable] | None = None
    for name in pool:
        if name in LOCAL_ZOO:
            names.append(name)
            continue
        if external is None:
            external = _external_zoo()
        if name in external:
            names.append(name)
    return names


def _agent_for_name(name: str) -> Callable:
    if name in LOCAL_ZOO:
        return LOCAL_ZOO[name]
    return _external_zoo().get(name, LOCAL_ZOO["random"])


def _linear_schedule(start: float, end: float, frac: float) -> float:
    return float(start + (end - start) * min(1.0, max(0.0, frac)))


def _candidate_move(game: Dict[str, Any], action: tuple[int, int, int]) -> list[list[int]]:
    src_id, tgt_id, ships = action
    if src_id < 0 or tgt_id < 0 or ships <= 0:
        return []
    src = next((p for p in game.get("planets", []) if int(p["id"]) == int(src_id)), None)
    tgt = next((p for p in game.get("planets", []) if int(p["id"]) == int(tgt_id)), None)
    if src is None or tgt is None:
        return []
    angle = safe_plan_shot(src, tgt, game)
    if angle is None:
        return []
    return [[int(src_id), float(angle), int(ships)]]


def _send_ratios(config: Dict[str, Any]) -> Tuple[float, ...]:
    values = config.get("send_ratios", [0.25, 0.5, 0.75])
    ratios = tuple(float(v) for v in values if 0.0 < float(v) < 1.0)
    return ratios or (0.25, 0.5, 0.75)


def run_match(
    agents,
    seed: int | None = None,
    n_players: int | None = None,
    neutral_pairs: int = 8,
    max_steps: int = 100,
    overage_time: float = 60.0,
    stop_player: int | None = None,
) -> Dict[str, Any]:
    """Run a local SimGame match and stop early if `stop_player` is eliminated."""
    players = int(n_players or len(agents))
    if len(agents) != players:
        raise ValueError(f"agent count ({len(agents)}) must match n_players ({players})")

    game = SimGame.random_game(
        seed=seed,
        n_players=players,
        neutral_pairs=neutral_pairs,
        max_steps=max_steps,
        overage_time=overage_time,
    )
    started = time.perf_counter()
    stop_player = None if stop_player is None else int(stop_player)
    initial_state = game.observation(stop_player if stop_player is not None else 0)

    while not game.is_terminal():
        actions = {}
        for player, agent in enumerate(agents):
            obs = game.observation(player)
            try:
                move = agent(obs, None)
            except TypeError:
                move = agent(obs)
            actions[player] = move if isinstance(move, list) else []
        game.step(actions)
        if stop_player is not None and stop_player not in game.alive_players():
            break

    elapsed = time.perf_counter() - started
    return {
        "winner": game.winner(),
        "scores": game.scores(),
        "steps": int(game.state.step),
        "seconds": elapsed,
        "steps_per_second": game.state.step / max(elapsed, 1e-9),
        "initial_state": initial_state,
        "final_state": game.observation(stop_player if stop_player is not None else 0),
    }


def _run_match_compat(
    agents,
    seed: int,
    n_players: int,
    max_steps: int,
    stop_player: int | None,
) -> Dict[str, Any]:
    try:
        return run_match(
            agents,
            seed=seed,
            n_players=n_players,
            max_steps=max_steps,
            stop_player=stop_player,
        )
    except TypeError as exc:
        if "stop_player" not in str(exc):
            raise
        return run_match(
            agents,
            seed=seed,
            n_players=n_players,
            max_steps=max_steps,
        )


def _make_our_agent(
    model: NeuralNetworkModel,
    config: Dict[str, Any],
    log_probs: List[torch.Tensor],
    action_records: List[Dict[str, Any]],
    temperature: float,
):
    def agent(obs, _config=None):
        game = obs_to_game_dict(obs)
        encoded = encode_game_state(game, config)
        ratios = _send_ratios(config)
        candidates = build_action_candidates(game, send_ratios=ratios)
        candidate_features = np.stack([c.score_features for c in candidates]).astype(np.float32)
        outputs = model(
            torch.tensor(encoded.features, dtype=torch.float32),
            torch.tensor(candidate_features, dtype=torch.float32),
        )
        cand, log_prob, entropy = choose_action(
            outputs,
            game,
            temperature=temperature,
            explore=True,
            return_entropy=True,
            send_ratios=ratios,
            prior_strength=float(config.get("policy_prior_strength", 0.8)),
        )
        log_probs.append(log_prob)
        action = reconstruct_action(cand, game)
        move = _candidate_move(game, action)
        executed_ships = int(move[0][2]) if move else 0
        action_records.append(
            {
                "mission": cand.mission if move else "do_nothing",
                "ships": executed_ships,
                "_value": outputs["value"].reshape(-1)[0],
                "_entropy": entropy,
            }
        )
        return move

    return agent


def _sample_opponents(pool: Sequence[str], seed: int, count: int = 3) -> List[str]:
    rng = random.Random(seed)
    names = _known_opponent_names(pool)
    if not names:
        names = ["random", "greedy", "starter"]
    rng.shuffle(names)
    return [names[i % len(names)] for i in range(max(1, count))]


def _build_agents(model: NeuralNetworkModel, config: Dict[str, Any], seed: int, our_index: int, temperature: float, pool: Sequence[str]):
    log_probs: List[torch.Tensor] = []
    action_records: List[Dict[str, Any]] = []
    opp_names = _sample_opponents(pool, seed, int(config.get("train_notebook_opponents", 1)))
    opp_iter = iter(opp_names)
    agents = []
    our_agent = _make_our_agent(model, config, log_probs, action_records, temperature)
    for slot in range(4):
        if slot == our_index:
            agents.append(our_agent)
        else:
            name = next(opp_iter, None)
            agents.append(_agent_for_name(name or "random"))
    return agents, log_probs, action_records, opp_names


def _episode_reward(result: Dict[str, Any], our_index: int) -> float:
    """
    Reward basée uniquement sur le rang relatif — somme nulle entre joueurs.
    Rang 1 → +1.0 | Rang 2 → +0.33 | Rang 3 → -0.33 | Rang 4 → -1.0
    Supprime tous les anciens signaux (terminal, rank_bonus, score_signal,
    score_floor_penalty) qui rendaient l'espérance structurellement négative.
    """
    scores = result.get("scores", [])
    ordered = sorted(
        ((float(score), idx) for idx, score in enumerate(scores)),
        reverse=True,
    )
    rank = next(
        (r for r, (_, idx) in enumerate(ordered, start=1) if idx == our_index),
        4,
    )
    n_players = 4
    return float((n_players - rank) / (n_players - 1) * 2 - 1)


def _action_summary(action_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    action_count = len(action_records)
    real_actions = [a for a in action_records if a.get("mission") != "do_nothing" and int(a.get("ships", 0)) > 0]
    ships_sent = [int(a.get("ships", 0)) for a in real_actions]
    return {
        "action_count": action_count,
        "real_action_count": len(real_actions),
        "do_nothing_rate": float(1.0 - (len(real_actions) / action_count)) if action_count else 1.0,
        "avg_ships_sent": float(np.mean(ships_sent) if ships_sent else 0.0),
    }


def _train_episode(
    model: NeuralNetworkModel,
    optimizer: torch.optim.Optimizer,
    log_probs: List[torch.Tensor],
    reward: float,
    baseline: float,
    entropy_coef: float = 0.01,
    action_records: Sequence[Dict[str, Any]] | None = None,
    value_coef: float = 0.25,
) -> Dict[str, float]:
    if not log_probs:
        return {"loss": 0.0, "grad_norm": 0.0, "policy_entropy": 0.0, "value_loss": 0.0, "policy_loss": 0.0}
    stacked = torch.stack(log_probs)
    advantage = torch.as_tensor(float(reward - baseline), dtype=stacked.dtype, device=stacked.device)
    policy_loss = -(stacked * advantage.detach()).mean()
    entropy_values = []
    value_values = []
    for record in action_records or []:
        entropy = record.get("_entropy")
        value = record.get("_value")
        if isinstance(entropy, torch.Tensor):
            entropy_values.append(entropy)
        if isinstance(value, torch.Tensor):
            value_values.append(value)
    entropy_tensor = torch.stack(entropy_values).mean() if entropy_values else torch.zeros((), dtype=stacked.dtype, device=stacked.device)
    if value_values:
        values = torch.stack(value_values).reshape(-1)
        target = torch.full_like(values, float(reward))
        value_loss = (values - target).pow(2).mean()
    else:
        value_loss = torch.zeros((), dtype=stacked.dtype, device=stacked.device)
    loss = policy_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy_tensor
    optimizer.zero_grad()
    loss.backward()
    grad_norm = float(clip_grad_norm_(model.parameters(), 5.0).item())
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "grad_norm": grad_norm,
        "policy_entropy": float(entropy_tensor.item()),
        "value_loss": float(value_loss.item()),
        "policy_loss": float(policy_loss.item()),
    }


def _eval_match(model: NeuralNetworkModel, config: Dict[str, Any], seed: int, our_index: int, pool: Sequence[str], temperature: float = 0.0) -> Dict[str, Any]:
    agents, log_probs, action_records, opp_names = _build_agents(model, config, seed, our_index, temperature, pool)
    result = _run_match_compat(
        agents,
        seed=seed,
        n_players=4,
        max_steps=int(config.get("max_turns", 100)),
        stop_player=our_index,
    )
    reward = _episode_reward(result, our_index)
    scores = result.get("scores", [])
    ordered = sorted(((float(score), idx) for idx, score in enumerate(scores)), reverse=True)
    rank = next((rank for rank, (_, idx) in enumerate(ordered, start=1) if idx == our_index), 4)
    return {
        "reward": reward,
        "winner": int(result.get("winner", -1)),
        "scores": result.get("scores", []),
        "steps": int(result.get("steps", 0)),
        "opponents": opp_names,
        "our_index": int(our_index),
        "rank": int(rank),
        **_action_summary(action_records),
    }


def evaluate_4p(model: NeuralNetworkModel, config: Dict[str, Any], pool: Sequence[str], episodes: int, seed_offset: int = 0) -> Dict[str, Any]:
    seeds = [seed_offset + i for i in range(episodes)]
    rows: List[Dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        our_index = i % 4
        rows.append(_eval_match(model, config, seed, our_index, pool, temperature=0.0))
    rewards = [r["reward"] for r in rows]
    wins = [1.0 if r["winner"] == r["our_index"] else 0.0 for r in rows]
    scores = [float(r["scores"][r["our_index"]]) for r in rows if len(r["scores"]) > r["our_index"]]
    by_position = {
        f"p{pos}": float(np.mean([1.0 if r["winner"] == r["our_index"] else 0.0 for r in rows if r["our_index"] == pos]) if any(r["our_index"] == pos for r in rows) else 0.0)
        for pos in range(4)
    }
    return {
        "eval_mean": float(np.mean(rewards) if rewards else 0.0),
        "eval_std": float(np.std(rewards) if rewards else 0.0),
        "winrate": float(np.mean(wins) if wins else 0.0),
        "winrate_by_position": by_position,
        "rank_mean": float(np.mean([r["rank"] for r in rows]) if rows else 4.0),
        "avg_score": float(np.mean(scores) if scores else 0.0),
        "avg_episode_length": float(np.mean([r["steps"] for r in rows]) if rows else 0.0),
        "eval_action_count": float(np.mean([r["action_count"] for r in rows]) if rows else 0.0),
        "eval_do_nothing_rate": float(np.mean([r["do_nothing_rate"] for r in rows]) if rows else 1.0),
        "eval_avg_ships_sent": float(np.mean([r["avg_ships_sent"] for r in rows]) if rows else 0.0),
        "seeds": seeds,
    }


def run_notebook_4p_training(config: Dict[str, Any], resume: bool = True) -> Dict[str, Any]:
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    ensure_torch_dynamo_stub()

    model = NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(config), hidden_dim=int(config.get("hidden_dim", 256))))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    checkpoint_dir = Path(config["checkpoint_dir"])
    latest = Path(config.get("latest_checkpoint", checkpoint_dir / "latest.npz"))
    best = Path(config.get("best_checkpoint", checkpoint_dir / "best.npz"))
    candidate = Path(config.get("candidate_checkpoint", checkpoint_dir / "candidate.npz"))
    log_path = Path(config["log_dir"]) / "notebook_4p_training.jsonl"
    ensure_dir(checkpoint_dir)
    ensure_dir(config["log_dir"])

    resume_path = Path(config.get("resume_checkpoint", str(latest)))
    if resume and resume_path.exists():
        state, _ = load_checkpoint(resume_path)
        load_compatible_state_dict(model, state)

    pool_limit = max(4, int(config.get("notebook_pool_limit", 4)))
    pool_limit_max = max(pool_limit, int(config.get("notebook_pool_limit_max", 15)))
    pool = training_pool(limit=pool_limit)
    if not pool:
        pool = _notebook_names()
    train_steps = int(config.get("train_steps", 50))
    eval_episodes = max(20, int(config.get("eval_episodes", 20)))
    eval_every = max(1, int(config.get("eval_every", max(1, train_steps // 10))))
    max_wall_seconds = float(config.get("max_wall_seconds", 0.0))
    deadline = time.time() + max_wall_seconds if max_wall_seconds > 0.0 else None
    baseline = 0.0
    best_score = -1e9
    best_record: Dict[str, Any] = {}

    for step in range(train_steps):
        if deadline is not None and time.time() >= deadline:
            break
        progress = step / max(1, train_steps - 1)
        temperature = _linear_schedule(float(config.get("temperature_start", 1.2)), float(config.get("temperature_end", 0.35)), progress)
        our_index = step % 4
        seed = int(config["seed"]) + step * 97
        agents, log_probs, action_records, opp_names = _build_agents(model, config, seed, our_index, temperature, pool)
        result = _run_match_compat(
            agents,
            seed=seed,
            n_players=4,
            max_steps=int(config.get("max_turns", 100)),
            stop_player=our_index,
        )
        # Dense curriculum reward (annealed, actif uniquement en début d'entraînement)
        # On utilise l'état final du match comme approximation de next_state terminal
        final_state = result.get("final_state", {})
        initial_state = result.get("initial_state", {})
        if final_state and initial_state:
            prev_game = obs_to_game_dict(initial_state)
            next_game = obs_to_game_dict(final_state)
            dense = compute_dense_reward(
                prev_state=prev_game,
                next_state=next_game,
                current_step=step,
                curriculum_steps=max(train_steps // 2, 1),
            )
        else:
            dense = 0.0
        reward = _episode_reward(result, our_index) + dense
        action_summary = _action_summary(action_records)
        baseline_rate = max(0.0, min(1.0, float(config.get("baseline_momentum", 0.05))))
        baseline = (1.0 - baseline_rate) * baseline + baseline_rate * reward
        entropy_coef = _linear_schedule(
            float(config.get("entropy_coef_start", 0.05)),
            float(config.get("entropy_coef_end", 0.005)),
            progress,
        )
        train_metrics = _train_episode(
            model,
            optimizer,
            log_probs,
            reward,
            baseline,
            entropy_coef,
            action_records=action_records,
            value_coef=float(config.get("value_loss_coef", 0.25)),
        )
        record = {
            "step": step,
            "reward": reward,
            "baseline": baseline,
            "temperature": temperature,
            "entropy_coef": entropy_coef,
            "grad_norm": train_metrics["grad_norm"],
            "loss": train_metrics["loss"],
            "policy_entropy": train_metrics.get("policy_entropy", 0.0),
            "value_loss": train_metrics.get("value_loss", 0.0),
            "policy_loss": train_metrics.get("policy_loss", 0.0),
            "winner": int(result.get("winner", -1)),
            "our_index": our_index,
            "opponents": opp_names,
            "episode_length": int(result.get("steps", 0)),
            **action_summary,
        }

        promoted = False
        promotion_reason = "no promotion"
        if (step + 1) % eval_every == 0 or step == train_steps - 1:
            eval_stats = evaluate_4p(model, config, pool, eval_episodes, seed_offset=int(config["seed"]) + 50000 + step * 1000)
            record.update(eval_stats)
            if eval_stats["winrate"] > 0.65 and pool_limit < pool_limit_max:
                pool_limit = pool_limit_max
                pool = training_pool(limit=pool_limit)
                if not pool:
                    pool = _notebook_names()
                record["curriculum_escalated"] = True
                record["curriculum_pool_limit"] = pool_limit
            else:
                record["curriculum_escalated"] = False
                record["curriculum_pool_limit"] = pool_limit
            if eval_stats["winrate"] > best_score:
                best_score = eval_stats["winrate"]
                promoted = True
                promotion_reason = f"winrate improved to {eval_stats['winrate']:.4f}"
                save_checkpoint(best, model.state_dict(), record)
                best_record = record.copy()
            record["checkpoint_promoted"] = promoted
            record["promotion_reason"] = promotion_reason
        else:
            record["checkpoint_promoted"] = False
            record["promotion_reason"] = promotion_reason
            record["curriculum_escalated"] = False
            record["curriculum_pool_limit"] = pool_limit

        append_jsonl(log_path, record)
        save_checkpoint(candidate, model.state_dict(), record)
        save_checkpoint(latest, model.state_dict(), record)
        if step == 0 or (step + 1) % max(1, eval_every // 2) == 0 or step == train_steps - 1:
            print(
                f"step {step+1}/{train_steps} "
                f"reward={reward:+.3f} "
                f"temp={temperature:.3f} "
                f"entropy={train_metrics.get('policy_entropy', 0.0):.4f} "
                f"baseline={baseline:+.3f} "
                f"ops={'/'.join(opp_names)}",
                flush=True,
            )

    return {
        "best_score": best_score,
        "best": str(best),
        "latest": str(latest),
        "candidate": str(candidate),
        "best_record": best_record,
        "pool": pool,
    }


def _infer_input_dim(config: Dict[str, Any]) -> int:
    from .self_play import make_synthetic_game

    sample = make_synthetic_game(seed=int(config["seed"]), four_player=True)
    encoded = encode_game_state(sample, config)
    return int(encoded.features.size)
