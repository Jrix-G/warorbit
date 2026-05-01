from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .benchmark import benchmark_matchups
from .encoder import encode_game_state
from .model import NeuralNetworkModel, ModelConfig
from .self_play import make_synthetic_game, play_episode
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


def _curriculum_ratios(config: Dict[str, Any], step: int) -> tuple[float, float]:
    if not bool(config.get("curriculum_enabled", False)):
        ratio4 = float(config.get("four_player_ratio", 0.5))
        return 1.0 - ratio4, ratio4
    early = float(config.get("curriculum_early_4p_ratio", 0.3))
    mid = float(config.get("curriculum_mid_4p_ratio", 0.6))
    late = float(config.get("curriculum_late_4p_ratio", 1.0))
    phase1 = int(config.get("curriculum_phase1_steps", 10))
    phase2 = int(config.get("curriculum_phase2_steps", 20))
    if step < phase1:
        ratio4 = early
    elif step < phase2:
        ratio4 = mid
    else:
        ratio4 = late
    return 1.0 - ratio4, ratio4


def _stable_eval(model: NeuralNetworkModel, config: Dict[str, Any], seeds: List[int], four_player_ratio: float) -> Dict[str, float]:
    local_cfg = dict(config)
    local_cfg["four_player_ratio"] = four_player_ratio
    local_cfg["eval_four_player_ratio"] = four_player_ratio
    scores: List[float] = []
    for seed in seeds:
        four_player = bool(np.random.random() < four_player_ratio)
        episode = play_episode(model, local_cfg, seed=seed, four_player=four_player)
        scores.append(float(sum(step["reward"] for step in episode)))
    bench = benchmark_matchups(model, local_cfg, episodes=max(1, int(config.get("benchmark_games", 8))), seed_offset=seeds[0] if seeds else 0)
    return {
        "eval_mean": float(np.mean(scores) if scores else 0.0),
        "eval_std": float(np.std(scores) if scores else 0.0),
        "eval_min": float(np.min(scores) if scores else 0.0),
        "eval_max": float(np.max(scores) if scores else 0.0),
        **bench,
    }


def run_training(config: Dict[str, Any], resume: bool = True) -> Dict[str, Any]:
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    model = NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(config), hidden_dim=int(config.get("hidden_dim", 128))))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    latest = Path(config["latest_checkpoint"])
    best = Path(config["best_checkpoint"])
    candidate = Path(config.get("candidate_checkpoint", str(best.parent / "candidate.npz")))
    log_path = Path(config["log_dir"]) / "training.jsonl"
    if resume and latest.exists():
        state, _meta = load_checkpoint(latest)
        model.load_state_dict(_state_dict_to_torch(state))
    best_score = -1e9
    best_record: Dict[str, Any] = {}
    eval_episodes = max(1, int(config.get("eval_episodes", config.get("benchmark_games", 8))))
    eval_seeds = [int(config["seed"]) + 50000 + i for i in range(eval_episodes)]
    for step in range(int(config["train_steps"])):
        _, ratio4 = _curriculum_ratios(config, step)
        episode = play_episode(model, config, seed=int(config["seed"]) + step, four_player=(np.random.random() < ratio4))
        metrics = train_step(model, episode, optimizer, float(config["gamma"]))
        benchmark = _stable_eval(
            model,
            config,
            eval_seeds,
            float(config.get("eval_four_player_ratio", config.get("four_player_ratio", 0.5))),
        )
        record = {
            "step": step,
            **metrics,
            **benchmark,
            "episode_length": len(episode),
            "train_ratio_4p": ratio4,
            "train_mode": "4p" if ratio4 >= 0.5 else "2p",
        }
        append_jsonl(log_path, record)
        save_checkpoint(candidate, model.state_dict(), record)
        save_checkpoint(latest, model.state_dict(), record)

        margin = float(config.get("promotion_margin", 0.05))
        min_std = float(config.get("promotion_min_eval_std", 0.25))
        if benchmark["eval_mean"] > best_score + margin or (benchmark["eval_mean"] > best_score and benchmark["eval_std"] <= min_std):
            best_score = benchmark["eval_mean"]
            best_record = record
            save_checkpoint(best, model.state_dict(), record)

        record["best_score"] = best_score
    return {
        "best_score": best_score,
        "latest": str(latest),
        "best": str(best),
        "candidate": str(candidate),
        "best_record": best_record,
    }


def _infer_input_dim(config: Dict[str, Any]) -> int:
    sample = make_synthetic_game(seed=int(config["seed"]), four_player=bool(config.get("train_four_player", False)))
    encoded = encode_game_state(sample, config)
    return int(encoded.features.size)


def _state_dict_to_torch(state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v) for k, v in state.items()}
