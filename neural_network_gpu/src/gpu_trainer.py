from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_network.src.model import NeuralNetworkModel
from neural_network.src.population_4p_training import _activity_shaping_reward
from neural_network.src.notebook_4p_training import _action_summary


def train_on_episodes(
    model: NeuralNetworkModel,
    optimizer: torch.optim.Optimizer,
    episodes: List[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
    baseline: float,
    baseline_momentum: float,
) -> Tuple[float, Dict[str, float]]:
    if not episodes:
        return baseline, {}

    entropy_coef = float(config.get("entropy_coef_start", 0.065))
    value_coef = float(config.get("value_loss_coef", 0.25))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))

    rewards = []
    all_policy_losses = []
    all_value_losses = []
    all_entropies = []

    model.train()

    for ep in episodes:
        trajectory = ep["trajectory"]
        if not trajectory:
            continue

        activity_reward = _activity_shaping_reward(ep.get("action_metrics", {}), config)
        reward = float(ep["terminal_reward"] + ep.get("dense_reward", 0.0) + activity_reward)
        reward = max(-2.0, min(2.0, reward))
        rewards.append(reward)

        advantage = reward - baseline

        ep_policy_losses = []
        ep_value_losses = []
        ep_entropies = []

        for step in trajectory:
            state_t = torch.as_tensor(step["state"], dtype=torch.float32, device=device).unsqueeze(0)
            cand_t = torch.as_tensor(step["candidates"], dtype=torch.float32, device=device).unsqueeze(0)

            outputs = model(state_t, cand_t)
            logits = outputs["policy_logits"].squeeze(0)
            value = outputs["value"].squeeze()

            log_probs = torch.log_softmax(logits, dim=0)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum()

            action_idx = int(step["action_idx"])
            if 0 <= action_idx < logits.shape[0]:
                log_prob = log_probs[action_idx]
            else:
                log_prob = torch.zeros(1, device=device).squeeze()

            ep_policy_losses.append(-log_prob * advantage)
            ep_value_losses.append((value - torch.tensor(reward, device=device)) ** 2)
            ep_entropies.append(entropy)

        if not ep_policy_losses:
            continue

        loss = (
            torch.stack(ep_policy_losses).mean()
            + value_coef * torch.stack(ep_value_losses).mean()
            - entropy_coef * torch.stack(ep_entropies).mean()
        )

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        all_policy_losses.append(torch.stack(ep_policy_losses).mean().item())
        all_value_losses.append(torch.stack(ep_value_losses).mean().item())
        all_entropies.append(torch.stack(ep_entropies).mean().item())

    model.eval()

    # Update baseline
    for r in rewards:
        baseline = (1.0 - baseline_momentum) * baseline + baseline_momentum * r

    metrics = {
        "policy_loss": float(np.mean(all_policy_losses)) if all_policy_losses else 0.0,
        "value_loss": float(np.mean(all_value_losses)) if all_value_losses else 0.0,
        "entropy": float(np.mean(all_entropies)) if all_entropies else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_win": float(np.mean([ep["win"] for ep in episodes])),
    }
    return baseline, metrics
