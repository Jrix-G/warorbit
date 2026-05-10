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
    ppo_clip_eps = float(config.get("ppo_clip_eps", 0.2))
    ppo_epochs = int(config.get("ppo_epochs", 3))
    advantage_eps = 1e-6

    # Build per-step dataset across all episodes
    dataset: List[Dict[str, Any]] = []
    rewards: List[float] = []
    terminal_rewards: List[float] = []
    dense_rewards: List[float] = []
    activity_rewards: List[float] = []
    skipped_missing_old_log_prob = 0
    trajectory_lengths: List[int] = []

    for ep in episodes:
        trajectory = ep["trajectory"]
        if not trajectory:
            continue
        activity_reward = _activity_shaping_reward(ep.get("action_metrics", {}), config)
        terminal_reward = float(ep["terminal_reward"])
        dense_reward = float(ep.get("dense_reward", 0.0))
        reward = float(terminal_reward + 0.15 * dense_reward + 0.05 * activity_reward)
        reward = max(-1.2, min(1.2, reward))
        rewards.append(reward)
        terminal_rewards.append(terminal_reward)
        dense_rewards.append(dense_reward)
        activity_rewards.append(float(activity_reward))
        valid_steps = [
            step for step in trajectory
            if step.get("old_log_prob") is not None and int(step.get("action_idx", -1)) >= 0
        ]
        skipped_missing_old_log_prob += len(trajectory) - len(valid_steps)
        if not valid_steps:
            continue
        step_weight = 1.0 / float(len(valid_steps))
        trajectory_lengths.append(len(valid_steps))
        for step in valid_steps:
            dataset.append({
                "state": step["state"],
                "candidates": step["candidates"],
                "action_idx": int(step["action_idx"]),
                "old_log_prob": float(step["old_log_prob"]),
                "sample_entropy": float(step.get("entropy") or 0.0),
                "temperature": float(step.get("temperature") or 0.0),
                "policy_version": int(step.get("policy_version") or 0),
                "reward": reward,
                "step_weight": step_weight,
            })

    if not dataset:
        return baseline, {"skipped_missing_old_log_prob": float(skipped_missing_old_log_prob)}

    rewards_arr = np.array([d["reward"] for d in dataset], dtype=np.float32)
    advantages_arr = rewards_arr - float(baseline)
    adv_std = float(advantages_arr.std()) + advantage_eps
    advantages_arr = (advantages_arr - advantages_arr.mean()) / adv_std

    all_policy_losses: List[float] = []
    all_value_losses: List[float] = []
    all_entropies: List[float] = []
    all_clip_fracs: List[float] = []
    all_total_losses: List[float] = []
    all_approx_kls: List[float] = []
    all_grad_norms: List[float] = []
    all_ratios: List[float] = []

    model.train()

    for _epoch in range(max(1, ppo_epochs)):
        indices = np.random.permutation(len(dataset))
        ep_policy_losses = []
        ep_value_losses = []
        ep_entropies = []
        clipped = 0

        for i in indices:
            d = dataset[int(i)]
            state_t = torch.as_tensor(d["state"], dtype=torch.float32, device=device).unsqueeze(0)
            cand_t = torch.as_tensor(d["candidates"], dtype=torch.float32, device=device).unsqueeze(0)
            outputs = model(state_t, cand_t)
            logits = outputs["policy_logits"].squeeze(0)
            value = outputs["value"].squeeze()

            log_probs_all = torch.log_softmax(logits, dim=0)
            probs = log_probs_all.exp()
            entropy = -(probs * log_probs_all).sum()

            action_idx = d["action_idx"]
            if 0 <= action_idx < logits.shape[0]:
                new_log_prob = log_probs_all[action_idx]
            else:
                new_log_prob = torch.zeros((), device=device)

            old_lp = float(d["old_log_prob"])
            adv = float(advantages_arr[int(i)])
            reward = float(d["reward"])
            step_weight = float(d["step_weight"])

            ratio = torch.exp(new_log_prob - old_lp)
            unclipped = ratio * adv
            clipped_obj = torch.clamp(ratio, 1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps) * adv
            policy_loss = -torch.min(unclipped, clipped_obj) * step_weight
            value_loss = (value - torch.tensor(reward, device=device)) ** 2
            value_loss = value_loss * step_weight

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy * step_weight

            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            ep_policy_losses.append(float(policy_loss.item()))
            ep_value_losses.append(float(value_loss.item()))
            ep_entropies.append(float(entropy.item()))
            all_total_losses.append(float(loss.item()))
            all_grad_norms.append(float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm))
            log_ratio = new_log_prob.detach() - torch.tensor(old_lp, device=device)
            approx_kl = (torch.exp(log_ratio) - 1.0 - log_ratio).detach()
            all_approx_kls.append(float(approx_kl.item()))
            all_ratios.append(float(ratio.item()))
            if abs(float(ratio.item()) - 1.0) > ppo_clip_eps:
                clipped += 1

        if ep_policy_losses:
            all_policy_losses.append(float(np.mean(ep_policy_losses)))
            all_value_losses.append(float(np.mean(ep_value_losses)))
            all_entropies.append(float(np.mean(ep_entropies)))
            all_clip_fracs.append(clipped / max(1, len(indices)))

    model.eval()

    # Update baseline (running mean of episode rewards)
    for r in rewards:
        baseline = (1.0 - baseline_momentum) * baseline + baseline_momentum * r

    metrics = {
        "policy_loss": float(np.mean(all_policy_losses)) if all_policy_losses else 0.0,
        "value_loss": float(np.mean(all_value_losses)) if all_value_losses else 0.0,
        "total_loss": float(np.mean(all_total_losses)) if all_total_losses else 0.0,
        "entropy": float(np.mean(all_entropies)) if all_entropies else 0.0,
        "clip_frac": float(np.mean(all_clip_fracs)) if all_clip_fracs else 0.0,
        "approx_kl": float(np.mean(all_approx_kls)) if all_approx_kls else 0.0,
        "grad_norm": float(np.mean(all_grad_norms)) if all_grad_norms else 0.0,
        "ratio_mean": float(np.mean(all_ratios)) if all_ratios else 0.0,
        "ratio_std": float(np.std(all_ratios)) if all_ratios else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "median_reward": float(np.median(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "terminal_reward_mean": float(np.mean(terminal_rewards)) if terminal_rewards else 0.0,
        "dense_reward_mean": float(np.mean(dense_rewards)) if dense_rewards else 0.0,
        "activity_reward_mean": float(np.mean(activity_rewards)) if activity_rewards else 0.0,
        "mean_win": float(np.mean([ep["win"] for ep in episodes])),
        "skipped_missing_old_log_prob": float(skipped_missing_old_log_prob),
        "mean_trajectory_len": float(np.mean(trajectory_lengths)) if trajectory_lengths else 0.0,
        "mean_sample_temperature": float(np.mean([d["temperature"] for d in dataset])) if dataset else 0.0,
        "mean_sample_entropy": float(np.mean([d["sample_entropy"] for d in dataset])) if dataset else 0.0,
        "policy_version": float(max(d["policy_version"] for d in dataset)) if dataset else 0.0,
    }
    return baseline, metrics
