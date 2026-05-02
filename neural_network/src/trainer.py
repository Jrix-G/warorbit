from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .encoder import encode_game_state
from .diagnostics import should_promote_checkpoint
from .model import NeuralNetworkModel, ModelConfig, load_compatible_state_dict
from .self_play import make_synthetic_game, play_episode
from .storage import append_jsonl, load_checkpoint, save_checkpoint


def _returns(rewards: List[float], gamma: float) -> torch.Tensor:
    out = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        out.append(running)
    return torch.tensor(list(reversed(out)), dtype=torch.float32)


def train_step(model: NeuralNetworkModel, episode: List[Dict[str, Any]], optimizer: torch.optim.Optimizer, gamma: float) -> Dict[str, float]:
    metrics = _train_batch(model, [episode], optimizer, gamma, baseline=0.0, entropy_coef=0.0)
    metrics["reward_total"] = float(sum(float(step["reward"]) for step in episode))
    return metrics


def _linear_schedule(start: float, end: float, frac: float) -> float:
    return float(start + (end - start) * min(1.0, max(0.0, frac)))


def _episode_total_return(episode: List[Dict[str, Any]], gamma: float) -> float:
    rewards = [float(step["reward"]) for step in episode]
    returns = _returns(rewards, gamma)
    return float(returns[0].item()) if returns.numel() else 0.0


def _train_batch(
    model: NeuralNetworkModel,
    episodes: List[List[Dict[str, Any]]],
    optimizer: torch.optim.Optimizer,
    gamma: float,
    baseline: float,
    entropy_coef: float,
) -> Dict[str, float]:
    if not episodes:
        return {"loss": 0.0, "grad_norm": 0.0, "policy_loss": 0.0, "entropy_bonus": 0.0, "batch_return": 0.0}

    log_probs_list: List[torch.Tensor] = []
    entropy_list: List[torch.Tensor] = []
    advantages_list: List[torch.Tensor] = []
    episode_returns: List[float] = []

    for episode in episodes:
        if not episode:
            continue
        rewards = [float(step["reward"]) for step in episode]
        returns = _returns(rewards, gamma)
        if returns.numel() == 0:
            continue
        episode_returns.append(float(returns[0].item()))
        advantages = returns - float(baseline)
        advantages_list.append(advantages)
        log_probs_list.append(torch.stack([step["log_prob"] for step in episode]))
        entropies = torch.stack([
            step.get("entropy", torch.tensor(0.0, dtype=log_probs_list[-1].dtype, device=log_probs_list[-1].device))
            for step in episode
        ])
        entropy_list.append(entropies)

    if not log_probs_list:
        return {"loss": 0.0, "grad_norm": 0.0, "policy_loss": 0.0, "entropy_bonus": 0.0, "batch_return": 0.0}

    advantages = torch.cat(advantages_list).detach()
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    log_probs = torch.cat(log_probs_list)
    entropies = torch.cat(entropy_list)

    policy_loss = -(log_probs * advantages).mean()
    entropy_bonus = -float(entropy_coef) * entropies.mean()
    loss = policy_loss + entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item())
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "entropy_bonus": float(entropy_bonus.item()),
        "grad_norm": grad_norm,
        "batch_return": float(np.mean(episode_returns) if episode_returns else 0.0),
        "batch_return_std": float(np.std(episode_returns) if episode_returns else 0.0),
        "batch_episodes": float(len(episode_returns)),
    }


def _episode_won(episode: List[Dict[str, Any]]) -> float:
    if not episode:
        return 0.0
    final_state = episode[-1]["next_state"]
    return 1.0 if final_state.get("winner") == final_state.get("my_id", 0) else 0.0


def _curriculum_ratios(config: Dict[str, Any], level: int) -> tuple[float, float]:
    if not bool(config.get("curriculum_enabled", False)):
        ratio4 = float(config.get("four_player_ratio", 0.5))
        return 1.0 - ratio4, ratio4
    early = float(config.get("curriculum_early_4p_ratio", 0.3))
    mid = float(config.get("curriculum_mid_4p_ratio", 0.6))
    late = float(config.get("curriculum_late_4p_ratio", 1.0))
    if level <= 0:
        ratio4 = early
    elif level == 1:
        ratio4 = mid
    else:
        ratio4 = late
    return 1.0 - ratio4, ratio4


def _stable_eval(model: NeuralNetworkModel, config: Dict[str, Any], seeds: List[int], four_player_ratio: float) -> Dict[str, Any]:
    local_cfg = dict(config)
    local_cfg["four_player_ratio"] = four_player_ratio
    local_cfg["eval_four_player_ratio"] = four_player_ratio
    scores: List[float] = []
    for seed in seeds:
        four_player = bool(np.random.random() < four_player_ratio)
        episode = play_episode(model, local_cfg, seed=seed, four_player=four_player)
        scores.append(float(sum(step["reward"] for step in episode)))
    return {
        "eval_mean": float(np.mean(scores) if scores else 0.0),
        "eval_std": float(np.std(scores) if scores else 0.0),
        "eval_min": float(np.min(scores) if scores else 0.0),
        "eval_max": float(np.max(scores) if scores else 0.0),
        "eval_scores": scores,
    }


def run_training(config: Dict[str, Any], resume: bool = True) -> Dict[str, Any]:
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    model = NeuralNetworkModel(ModelConfig(input_dim=_infer_input_dim(config), hidden_dim=int(config.get("hidden_dim", 256))))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    latest = Path(config["latest_checkpoint"])
    best = Path(config["best_checkpoint"])
    candidate = Path(config.get("candidate_checkpoint", str(best.parent / "candidate.npz")))
    log_path = Path(config["log_dir"]) / "training.jsonl"
    resume_path = Path(config.get("resume_checkpoint", str(latest)))
    if resume and resume_path.exists():
        state, _meta = load_checkpoint(resume_path)
        load_compatible_state_dict(model, _state_dict_to_torch(state))
    best_score = -1e9
    best_record: Dict[str, Any] = {}
    eval_episodes = max(20, int(config.get("eval_episodes", config.get("benchmark_games", 20))))
    eval_seeds = [int(config["seed"]) + 50000 + i for i in range(eval_episodes)]
    batch_size = max(16, int(config.get("batch_size", config.get("self_play_episodes", 16))))
    eval_every_updates = max(1, int(config.get("eval_every_updates", 4)))
    baseline = float(config.get("moving_average_baseline", 0.0))
    baseline_momentum = float(config.get("baseline_momentum", 0.05))
    entropy_start = float(config.get("entropy_coef_start", config.get("entropy_coef", 0.01)))
    entropy_end = float(config.get("entropy_coef_end", entropy_start))
    curriculum_level = 0
    win_window = deque(maxlen=max(1, int(config.get("curriculum_window_size", 100))))
    curriculum_min_episodes = max(1, int(config.get("curriculum_min_episodes", min(20, int(config["train_steps"])))))
    curriculum_thresholds = list(config.get("curriculum_winrate_thresholds", [0.55, 0.65]))
    if not curriculum_thresholds:
        curriculum_thresholds = [0.55, 0.65]
    train_updates = int(config["train_steps"])
    for step in range(train_updates):
        if bool(config.get("curriculum_enabled", False)) and len(win_window) >= curriculum_min_episodes:
            recent_winrate = float(np.mean(list(win_window)))
            if curriculum_level == 0 and recent_winrate >= float(curriculum_thresholds[0]):
                curriculum_level = 1
            if curriculum_level == 1 and len(curriculum_thresholds) > 1 and recent_winrate >= float(curriculum_thresholds[1]):
                curriculum_level = 2
        else:
            recent_winrate = float(np.mean(list(win_window))) if win_window else 0.0
        _, ratio4 = _curriculum_ratios(config, curriculum_level)

        batch_episodes: List[List[Dict[str, Any]]] = []
        for episode_idx in range(batch_size):
            seed = int(config["seed"]) + step * batch_size + episode_idx
            progress = step / max(1, train_updates - 1)
            episode = play_episode(
                model,
                config,
                seed=seed,
                progress=progress,
                four_player=(np.random.random() < ratio4),
            )
            if not episode:
                continue
            batch_episodes.append(episode)
            win_window.append(_episode_won(episode))
        recent_winrate = float(np.mean(list(win_window))) if win_window else 0.0
        entropy_coef = _linear_schedule(entropy_start, entropy_end, step / max(1, train_updates - 1))
        batch_returns = [_episode_total_return(ep, float(config["gamma"])) for ep in batch_episodes]
        if batch_returns:
            batch_mean_return = float(np.mean(batch_returns))
            baseline = (1.0 - baseline_momentum) * baseline + baseline_momentum * batch_mean_return
        metrics = _train_batch(model, batch_episodes, optimizer, float(config["gamma"]), baseline=baseline, entropy_coef=entropy_coef)

        benchmark: Dict[str, Any] = {}
        should_eval = ((step + 1) % eval_every_updates == 0) or step == train_updates - 1
        if should_eval:
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
            "batch_episodes": len(batch_episodes),
            "batch_size": batch_size,
            "train_ratio_4p": ratio4,
            "train_mode": "4p" if ratio4 >= 0.5 else "2p",
            "curriculum_level": curriculum_level,
            "recent_winrate": recent_winrate,
            "moving_average_baseline": baseline,
            "entropy_coef": entropy_coef,
        }
        promoted = False
        promotion_reason = "no promotion"
        if should_eval and benchmark:
            eval_scores = benchmark.get("eval_scores", [])
            if should_promote_checkpoint(eval_scores, best_score, min_episodes=20):
                best_score = float(np.mean(eval_scores))
                best_record = record.copy()
                save_checkpoint(best, model.state_dict(), record)
                promoted = True
                promotion_reason = f"mean eval score improved to {best_score:.4f} over {len(eval_scores)} episodes"
        record["checkpoint_promoted"] = promoted
        record["promotion_reason"] = promotion_reason
        record["best_score"] = best_score if best_score > -1e9 else float(benchmark.get("eval_mean", 0.0))
        append_jsonl(log_path, record)
        save_checkpoint(candidate, model.state_dict(), record)
        save_checkpoint(latest, model.state_dict(), record)
    final_best_score = best_score if best_score > -1e9 else float(best_record.get("eval_mean", 0.0))
    return {
        "best_score": final_best_score,
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
