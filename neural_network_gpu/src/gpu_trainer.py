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
    minibatch_size = max(1, int(config.get("ppo_minibatch_size", 512)))
    advantage_eps = 1e-6

    # Build per-step dataset across all episodes
    dataset: List[Dict[str, Any]] = []
    rewards: List[float] = []
    terminal_rewards: List[float] = []
    dense_rewards: List[float] = []
    activity_rewards: List[float] = []
    passivity_penalties: List[float] = []
    do_nothing_rates: List[float] = []
    skipped_missing_old_log_prob = 0
    trajectory_lengths: List[int] = []

    for ep in episodes:
        trajectory = ep["trajectory"]
        if not trajectory:
            continue
        action_metrics = ep.get("action_metrics", {})
        activity_reward = _activity_shaping_reward(action_metrics, config)
        do_nothing_rate = max(0.0, min(1.0, float(action_metrics.get("do_nothing_rate", 1.0))))
        passive_limit = max(0.0, min(1.0, float(config.get("train_target_do_nothing_rate", 0.45))))
        passive_coef = max(0.0, float(config.get("train_passivity_penalty_coef", 0.55)))
        passive_excess = max(0.0, do_nothing_rate - passive_limit) / max(1e-6, 1.0 - passive_limit)
        passivity_penalty = passive_coef * passive_excess
        terminal_reward = float(ep["terminal_reward"])
        dense_reward = float(ep.get("dense_reward", 0.0))
        reward = float(terminal_reward + 0.15 * dense_reward + 0.05 * activity_reward - passivity_penalty)
        reward = max(-1.2, min(1.2, reward))
        rewards.append(reward)
        terminal_rewards.append(terminal_reward)
        dense_rewards.append(dense_reward)
        activity_rewards.append(float(activity_reward))
        passivity_penalties.append(float(passivity_penalty))
        do_nothing_rates.append(float(do_nothing_rate))
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
    all_log_ratio_abs: List[float] = []
    minibatch_updates = 0

    model.train()
    before_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    for _epoch in range(max(1, ppo_epochs)):
        indices = np.random.permutation(len(dataset))
        ep_policy_losses: List[float] = []
        ep_value_losses: List[float] = []
        ep_entropies: List[float] = []
        clipped = 0

        for start in range(0, len(indices), minibatch_size):
            batch_indices = indices[start:start + minibatch_size]
            batch = [dataset[int(i)] for i in batch_indices]
            n_cands = [int(d["candidates"].shape[0]) for d in batch]
            max_n = max(n_cands)
            cand_dim = int(batch[0]["candidates"].shape[-1])

            states = np.stack([d["state"] for d in batch]).astype(np.float32, copy=False)
            cands_padded = np.zeros((len(batch), max_n, cand_dim), dtype=np.float32)
            mask = np.zeros((len(batch), max_n), dtype=bool)
            for row, (d, n) in enumerate(zip(batch, n_cands)):
                cands_padded[row, :n] = d["candidates"]
                mask[row, :n] = True

            state_t = torch.as_tensor(states, dtype=torch.float32, device=device)
            cand_t = torch.as_tensor(cands_padded, dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
            action_t = torch.as_tensor([d["action_idx"] for d in batch], dtype=torch.long, device=device)
            old_lp_t = torch.as_tensor([d["old_log_prob"] for d in batch], dtype=torch.float32, device=device)
            adv_t = torch.as_tensor(advantages_arr[batch_indices], dtype=torch.float32, device=device)
            reward_t = torch.as_tensor([d["reward"] for d in batch], dtype=torch.float32, device=device)
            weight_t = torch.as_tensor([d["step_weight"] for d in batch], dtype=torch.float32, device=device)
            temp_t = torch.as_tensor(
                [float(d.get("temperature") or 1.0) if float(d.get("temperature") or 0.0) > 0.0 else 1.0 for d in batch],
                dtype=torch.float32,
                device=device,
            )

            outputs = model(state_t, cand_t)
            logits = outputs["policy_logits"]
            prior_strength = float(config.get("policy_prior_strength", 0.0))
            if prior_strength:
                logits = logits + prior_strength * cand_t[..., -1] * 3.0
            logits = logits.masked_fill(~mask_t, float("-inf"))
            value = outputs["value"]

            action_logits = logits / temp_t.unsqueeze(-1)
            log_probs_all = torch.log_softmax(action_logits, dim=-1)
            probs = log_probs_all.exp()
            safe_log_probs = log_probs_all.masked_fill(~mask_t, 0.0)
            entropy_terms = probs * safe_log_probs
            entropy = -entropy_terms.sum(dim=-1)

            new_log_prob = log_probs_all.gather(1, action_t.unsqueeze(-1)).squeeze(-1)
            ratio = torch.exp(new_log_prob - old_lp_t)
            unclipped = ratio * adv_t
            clipped_obj = torch.clamp(ratio, 1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps) * adv_t
            policy_loss_vec = -torch.min(unclipped, clipped_obj) * weight_t
            value_loss_vec = (value - reward_t).pow(2) * weight_t
            entropy_loss_vec = entropy * weight_t

            policy_loss = policy_loss_vec.sum()
            value_loss = value_loss_vec.sum()
            entropy_loss = entropy_loss_vec.sum()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            minibatch_updates += 1

            ep_policy_losses.append(float(policy_loss.item()))
            ep_value_losses.append(float(value_loss.item()))
            ep_entropies.append(float(entropy.mean().item()))
            all_total_losses.append(float(loss.item()))
            all_grad_norms.append(float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm))
            log_ratio = new_log_prob.detach() - old_lp_t
            approx_kl = (torch.exp(log_ratio) - 1.0 - log_ratio).detach()
            all_approx_kls.extend(approx_kl.cpu().numpy().astype(float).tolist())
            all_ratios.extend(ratio.detach().cpu().numpy().astype(float).tolist())
            all_log_ratio_abs.extend(log_ratio.abs().cpu().numpy().astype(float).tolist())
            clipped += int((ratio.detach() - 1.0).abs().gt(ppo_clip_eps).sum().item())

        if ep_policy_losses:
            all_policy_losses.append(float(np.mean(ep_policy_losses)))
            all_value_losses.append(float(np.mean(ep_value_losses)))
            all_entropies.append(float(np.mean(ep_entropies)))
            all_clip_fracs.append(clipped / max(1, len(indices)))

    model.eval()

    delta_sq = 0.0
    param_sq = 0.0
    param_count = 0
    with torch.no_grad():
        for before, after in zip(before_params, (p for p in model.parameters() if p.requires_grad)):
            before_cpu = before.detach().float().cpu()
            after_cpu = after.detach().float().cpu()
            diff = after_cpu - before_cpu
            delta_sq += float(diff.pow(2).sum().item())
            param_sq += float(before_cpu.pow(2).sum().item())
            param_count += int(diff.numel())
    param_delta_rms = float(np.sqrt(delta_sq / max(1, param_count)))
    param_relative_delta = float(np.sqrt(delta_sq / max(1e-12, param_sq)))

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
        "ratio_min": float(np.min(all_ratios)) if all_ratios else 0.0,
        "ratio_max": float(np.max(all_ratios)) if all_ratios else 0.0,
        "log_ratio_abs_max": float(np.max(all_log_ratio_abs)) if all_log_ratio_abs else 0.0,
        "param_delta_rms": param_delta_rms,
        "param_relative_delta": param_relative_delta,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "median_reward": float(np.median(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "terminal_reward_mean": float(np.mean(terminal_rewards)) if terminal_rewards else 0.0,
        "dense_reward_mean": float(np.mean(dense_rewards)) if dense_rewards else 0.0,
        "activity_reward_mean": float(np.mean(activity_rewards)) if activity_rewards else 0.0,
        "passivity_penalty_mean": float(np.mean(passivity_penalties)) if passivity_penalties else 0.0,
        "do_nothing_rate_mean": float(np.mean(do_nothing_rates)) if do_nothing_rates else 1.0,
        "mean_win": float(np.mean([ep["win"] for ep in episodes])),
        "skipped_missing_old_log_prob": float(skipped_missing_old_log_prob),
        "mean_trajectory_len": float(np.mean(trajectory_lengths)) if trajectory_lengths else 0.0,
        "mean_sample_temperature": float(np.mean([d["temperature"] for d in dataset])) if dataset else 0.0,
        "mean_sample_entropy": float(np.mean([d["sample_entropy"] for d in dataset])) if dataset else 0.0,
        "policy_version": float(max(d["policy_version"] for d in dataset)) if dataset else 0.0,
        "train_samples": float(len(dataset)),
        "train_minibatches": float(minibatch_updates),
        "ppo_minibatch_size": float(minibatch_size),
    }
    return baseline, metrics
