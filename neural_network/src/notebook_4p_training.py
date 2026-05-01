from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from opponents import ZOO, training_pool
from SimGame import run_match

from .encoder import encode_game_state
from .model import ModelConfig, NeuralNetworkModel
from .orbit_wars_adapter import obs_to_game_dict
from .policy import build_action_candidates, choose_action, reconstruct_action
from .storage import append_jsonl, load_checkpoint, save_checkpoint
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


def _make_our_agent(model: NeuralNetworkModel, config: Dict[str, Any], log_probs: List[torch.Tensor], temperature: float):
    def agent(obs, _config=None):
        game = obs_to_game_dict(obs)
        encoded = encode_game_state(game, config)
        candidates = build_action_candidates(game)
        candidate_features = np.stack([c.score_features for c in candidates]).astype(np.float32)
        outputs = model(
            torch.tensor(encoded.features, dtype=torch.float32),
            torch.tensor(candidate_features, dtype=torch.float32),
        )
        cand, log_prob = choose_action(outputs, game, temperature=temperature, explore=True)
        log_probs.append(log_prob)
        return _candidate_move(game, reconstruct_action(cand, game))

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
    opp_names = _sample_opponents(pool, seed, 3)
    opp_iter = iter(opp_names)
    agents = []
    our_agent = _make_our_agent(model, config, log_probs, temperature)
    for slot in range(4):
        if slot == our_index:
            agents.append(our_agent)
        else:
            name = next(opp_iter, None)
            agents.append(ZOO[name] if name is not None else ZOO["random"])
    return agents, log_probs, opp_names


def _episode_reward(result: Dict[str, Any], our_index: int) -> float:
    winner = int(result.get("winner", -1))
    scores = result.get("scores", [])
    our_score = float(scores[our_index]) if len(scores) > our_index else 0.0
    others = [float(s) for i, s in enumerate(scores) if i != our_index]
    best_other = max(others) if others else 0.0
    score_margin = (our_score - best_other) / max(1.0, abs(best_other) + 1.0)
    terminal = 1.0 if winner == our_index else (-1.0 if winner != -1 else 0.0)
    return float(terminal + 0.2 * score_margin)


def _train_episode(model: NeuralNetworkModel, optimizer: torch.optim.Optimizer, log_probs: List[torch.Tensor], reward: float, baseline: float) -> Dict[str, float]:
    if not log_probs:
        return {"loss": 0.0, "grad_norm": 0.0}
    advantage = reward - baseline
    loss = -torch.stack(log_probs).sum() * float(advantage)
    optimizer.zero_grad()
    loss.backward()
    grad_norm = float(clip_grad_norm_(model.parameters(), 5.0).item())
    optimizer.step()
    return {"loss": float(loss.item()), "grad_norm": grad_norm}


def _eval_match(model: NeuralNetworkModel, config: Dict[str, Any], seed: int, our_index: int, pool: Sequence[str], temperature: float = 0.0) -> Dict[str, Any]:
    agents, log_probs, opp_names = _build_agents(model, config, seed, our_index, temperature, pool)
    result = run_match(agents, seed=seed, n_players=4, max_steps=int(config.get("max_turns", 100)))
    reward = _episode_reward(result, our_index)
    return {
        "reward": reward,
        "winner": int(result.get("winner", -1)),
        "scores": result.get("scores", []),
        "steps": int(result.get("steps", 0)),
        "opponents": opp_names,
        "our_index": int(our_index),
    }


def evaluate_4p(model: NeuralNetworkModel, config: Dict[str, Any], pool: Sequence[str], episodes: int, seed_offset: int = 0) -> Dict[str, Any]:
    seeds = [seed_offset + i for i in range(episodes)]
    rows: List[Dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        our_index = i % 4
        rows.append(_eval_match(model, config, seed, our_index, pool, temperature=0.0))
    rewards = [r["reward"] for r in rows]
    wins = [1.0 if r["winner"] == r["our_index"] else 0.0 for r in rows]
    by_position = {
        f"p{pos}": float(np.mean([1.0 if r["winner"] == r["our_index"] else 0.0 for r in rows if r["our_index"] == pos]) if any(r["our_index"] == pos for r in rows) else 0.0)
        for pos in range(4)
    }
    return {
        "eval_mean": float(np.mean(rewards) if rewards else 0.0),
        "eval_std": float(np.std(rewards) if rewards else 0.0),
        "winrate": float(np.mean(wins) if wins else 0.0),
        "winrate_by_position": by_position,
        "avg_episode_length": float(np.mean([r["steps"] for r in rows]) if rows else 0.0),
        "seeds": seeds,
    }


def run_notebook_4p_training(config: Dict[str, Any], resume: bool = True) -> Dict[str, Any]:
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    random.seed(int(config["seed"]))

    model = NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(config), hidden_dim=int(config.get("hidden_dim", 128))))
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
        model.load_state_dict(state)

    pool = training_pool(limit=int(config.get("notebook_pool_limit", 15)))
    if not pool:
        pool = [name for name in ZOO.keys() if name.startswith("notebook_")]
    train_steps = int(config.get("train_steps", 50))
    eval_episodes = max(1, int(config.get("eval_episodes", 20)))
    eval_every = max(1, int(config.get("eval_every", max(1, train_steps // 5))))
    baseline = 0.0
    best_score = -1e9
    best_record: Dict[str, Any] = {}

    for step in range(train_steps):
        progress = step / max(1, train_steps - 1)
        temperature = _linear_schedule(float(config.get("temperature_start", 1.2)), float(config.get("temperature_end", 0.35)), progress)
        our_index = step % 4
        seed = int(config["seed"]) + step * 97
        agents, log_probs, opp_names = _build_agents(model, config, seed, our_index, temperature, pool)
        result = run_match(agents, seed=seed, n_players=4, max_steps=int(config.get("max_turns", 100)))
        reward = _episode_reward(result, our_index)
        baseline = 0.95 * baseline + 0.05 * reward
        train_metrics = _train_episode(model, optimizer, log_probs, reward, baseline)
        record = {
            "step": step,
            "reward": reward,
            "baseline": baseline,
            "temperature": temperature,
            "grad_norm": train_metrics["grad_norm"],
            "loss": train_metrics["loss"],
            "winner": int(result.get("winner", -1)),
            "our_index": our_index,
            "opponents": opp_names,
            "episode_length": int(result.get("steps", 0)),
        }

        promoted = False
        promotion_reason = "no promotion"
        if (step + 1) % eval_every == 0 or step == train_steps - 1:
            eval_stats = evaluate_4p(model, config, pool, eval_episodes, seed_offset=int(config["seed"]) + 50000 + step * 1000)
            record.update(eval_stats)
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

        append_jsonl(log_path, record)
        save_checkpoint(candidate, model.state_dict(), record)
        save_checkpoint(latest, model.state_dict(), record)

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
