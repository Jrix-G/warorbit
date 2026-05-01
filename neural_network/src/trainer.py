from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json
import numpy as np

from .model import NeuralNetworkModel, ModelConfig
from .self_play import play_episode, make_synthetic_game
from .storage import save_checkpoint, append_jsonl, load_checkpoint
from .benchmark import benchmark_model
from .encoder import encode_game_state


def train_step(model: NeuralNetworkModel, batch: np.ndarray, lr: float) -> float:
    out = model.forward(batch)
    loss = float(np.mean(out["value"] ** 2))
    for p in model.parameters():
        p -= lr * np.sign(p) * 0.001
    return loss


def run_training(config: Dict[str, Any], resume: bool = True) -> Dict[str, Any]:
    rng = np.random.default_rng(int(config["seed"]))
    model = NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(config)), rng=rng)
    latest = Path(config["latest_checkpoint"])
    best = Path(config["best_checkpoint"])
    log_path = Path(config["log_dir"]) / "training.jsonl"
    if resume and latest.exists():
        state, meta = load_checkpoint(latest)
        model.load_state_dict(state)

    best_score = -1e9
    for step in range(int(config["train_steps"])):
        game = play_episode(model, config, seed=step)[-1]["next_state"]
        encoded = encode_game_state(game, config)
        loss = train_step(model, encoded.features[None, :], float(config["learning_rate"]))
        if step % 5 == 0:
            metrics = benchmark_model(model, config, games=max(1, int(config["benchmark_games"]) // 2))
            record = {"step": step, "loss": loss, **metrics}
            append_jsonl(log_path, record)
            save_checkpoint(latest, model.state_dict(), record)
            if metrics["avg_reward"] > best_score:
                best_score = metrics["avg_reward"]
                save_checkpoint(best, model.state_dict(), record)
    return {"best_score": best_score, "latest": str(latest), "best": str(best)}


def _infer_input_dim(config: Dict[str, Any]) -> int:
    sample = make_synthetic_game(seed=int(config["seed"]))
    encoded = encode_game_state(sample, config)
    return int(encoded.features.size)
