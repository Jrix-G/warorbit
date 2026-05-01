from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .benchmark import benchmark_matchups
from .encoder import encode_game_state
from .model import NeuralNetworkModel, ModelConfig
from .self_play import play_episode, make_synthetic_game
from .storage import append_jsonl, load_checkpoint, save_checkpoint


def _returns(rewards: List[float], gamma: float) -> torch.Tensor:
    out = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        out.append(running)
    returns = torch.tensor(list(reversed(out)), dtype=torch.float32)
    if returns.numel() > 1:
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
    return returns


def train_step(model: NeuralNetworkModel, episode: List[Dict[str, Any]], optimizer: torch.optim.Optimizer, gamma: float) -> Dict[str, float]:
    log_probs = torch.stack([step["log_prob"] for step in episode])
    rewards = [float(step["reward"]) for step in episode]
    returns = _returns(rewards, gamma)
    loss = -(log_probs * returns).sum()
    optimizer.zero_grad()
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item())
    optimizer.step()
    return {"loss": float(loss.item()), "grad_norm": grad_norm, "reward_total": float(sum(rewards))}


def run_training(config: Dict[str, Any], resume: bool = True) -> Dict[str, Any]:
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    model = NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(config), hidden_dim=int(config.get("hidden_dim", 128))))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    latest = Path(config["latest_checkpoint"])
    best = Path(config["best_checkpoint"])
    log_path = Path(config["log_dir"]) / "training.jsonl"
    if resume and latest.exists():
        state, meta = load_checkpoint(latest)
        model.load_state_dict(state)
    best_score = -1e9
    for step in range(int(config["train_steps"])):
        episode = play_episode(model, config, seed=int(config["seed"]) + step)
        metrics = train_step(model, episode, optimizer, float(config["gamma"]))
        benchmark = benchmark_matchups(model, config, episodes=max(1, int(config["benchmark_games"]) // 2))
        record = {"step": step, **metrics, **benchmark, "episode_length": len(episode)}
        append_jsonl(log_path, record)
        save_checkpoint(latest, model.state_dict(), record)
        if metrics["reward_total"] > best_score:
            best_score = metrics["reward_total"]
            save_checkpoint(best, model.state_dict(), record)
    return {"best_score": best_score, "latest": str(latest), "best": str(best)}


def _infer_input_dim(config: Dict[str, Any]) -> int:
    sample = make_synthetic_game(seed=int(config["seed"]))
    encoded = encode_game_state(sample, config)
    return int(encoded.features.size)
