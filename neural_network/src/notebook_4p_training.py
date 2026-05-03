from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from opponents import ZOO, training_pool
from SimGame import SimGame

from .encoder import encode_game_state
from .model import ModelConfig, NeuralNetworkModel, load_compatible_state_dict
from .orbit_wars_adapter import obs_to_game_dict
from .policy import build_action_candidates, choose_action, reconstruct_action
from .reward import compute_dense_reward
from .storage import append_jsonl, load_checkpoint, save_checkpoint
from .torch_compat import ensure_torch_dynamo_stub
from .utils import ensure_dir


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
    angle = math.atan2(float(tgt["y"]) - float(src["y"]), float(tgt["x"]) - float(src["x"]))
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
        "final_state": game.observation(stop_player if stop_player is not None else 0),
    }


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
        cand, log_prob = choose_action(
            outputs,
            game,
            temperature=temperature,
            explore=True,
            send_ratios=ratios,
            prior_strength=float(config.get("policy_prior_strength", 0.8)),
        )
        log_probs.append(log_prob)
        action = reconstruct_action(cand, game)
        action_records.append({"mission": cand.mission, "ships": int(action[2])})
        return _candidate_move(game, action)

    return agent


def _sample_opponents(pool: Sequence[str], seed: int, count: int = 3) -> List[str]:
    rng = random.Random(seed)
    names = [name for name in pool if name in ZOO]
    if len(names) < count:
        names = list(ZOO.keys())
    rng.shuffle(names)
    return names[:count]


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
            agents.append(ZOO[name] if name is not None else ZOO["random"])
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
) -> Dict[str, float]:
    if not log_probs:
        return {"loss": 0.0, "grad_norm": 0.0, "policy_entropy": 0.0}
    stacked = torch.stack(log_probs)
    advantage = reward - baseline
    policy_loss = -stacked.sum() * float(advantage)
    # Bonus d'entropie : encourage l'exploration, évite le collapse
    entropy = -(stacked * stacked.exp()).sum()
    loss = policy_loss - entropy_coef * entropy
    optimizer.zero_grad()
    loss.backward()
    grad_norm = float(clip_grad_norm_(model.parameters(), 5.0).item())
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "grad_norm": grad_norm,
        "policy_entropy": float(entropy.item()),
    }


def _eval_match(model: NeuralNetworkModel, config: Dict[str, Any], seed: int, our_index: int, pool: Sequence[str], temperature: float = 0.0) -> Dict[str, Any]:
    agents, log_probs, action_records, opp_names = _build_agents(model, config, seed, our_index, temperature, pool)
    result = run_match(
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
        pool = [name for name in ZOO.keys() if name.startswith("notebook_")]
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
        result = run_match(
            agents,
            seed=seed,
            n_players=4,
            max_steps=int(config.get("max_turns", 100)),
            stop_player=our_index,
        )
        # Dense curriculum reward (annealed, actif uniquement en début d'entraînement)
        # On utilise l'état final du match comme approximation de next_state terminal
        final_state = result.get("final_state", {})
        if final_state:
            dense = compute_dense_reward(
                prev_state={},   # pas d'état précédent disponible au niveau épisode
                next_state=final_state,
                current_step=step,
                curriculum_steps=max(train_steps // 2, 1),
            )
        else:
            dense = 0.0
        reward = _episode_reward(result, our_index) + dense
        action_summary = _action_summary(action_records)
        baseline = 0.95 * baseline + 0.05 * reward
        entropy_coef = _linear_schedule(
            float(config.get("entropy_coef_start", 0.05)),
            float(config.get("entropy_coef_end", 0.005)),
            progress,
        )
        train_metrics = _train_episode(model, optimizer, log_probs, reward, baseline, entropy_coef)
        record = {
            "step": step,
            "reward": reward,
            "baseline": baseline,
            "temperature": temperature,
            "entropy_coef": entropy_coef,
            "grad_norm": train_metrics["grad_norm"],
            "loss": train_metrics["loss"],
            "policy_entropy": train_metrics.get("policy_entropy", 0.0),
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
                    pool = [name for name in ZOO.keys() if name.startswith("notebook_")]
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
