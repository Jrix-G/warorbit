from __future__ import annotations

from typing import Sequence

import torch


def action_slot_noop_caps(config: dict, max_actions: int) -> list[float]:
    raw_caps = config.get("do_nothing_prob_caps_by_slot")
    if isinstance(raw_caps, Sequence) and not isinstance(raw_caps, (str, bytes)) and raw_caps:
        caps = [float(value) for value in raw_caps]
    else:
        caps = [float(config.get("do_nothing_prob_cap", 1.0))]
    if len(caps) < max_actions:
        caps.extend([caps[-1]] * (max_actions - len(caps)))
    return [max(0.0, min(1.0, cap)) for cap in caps[:max_actions]]


def caps_for_action_slots(config: dict, action_slots: torch.Tensor) -> torch.Tensor:
    if action_slots.numel() == 0:
        return torch.empty_like(action_slots, dtype=torch.float32)
    max_slot = int(action_slots.detach().max().item()) + 1
    caps = action_slot_noop_caps(config, max(1, max_slot))
    cap_t = torch.tensor(caps, dtype=torch.float32, device=action_slots.device)
    return cap_t[action_slots.clamp(min=0, max=len(caps) - 1)]


def cap_do_nothing_probability(
    probs: torch.Tensor,
    valid_mask: torch.Tensor,
    max_noop_prob: float | torch.Tensor,
) -> torch.Tensor:
    capped, _info = cap_do_nothing_probability_with_info(probs, valid_mask, max_noop_prob)
    return capped


def cap_do_nothing_probability_with_info(
    probs: torch.Tensor,
    valid_mask: torch.Tensor,
    max_noop_prob: float | torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = probs.size(0)
    device = probs.device
    dtype = probs.dtype
    empty_bool = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    empty_float = torch.ones((batch_size,), dtype=dtype, device=device)
    if probs.size(-1) <= 1:
        return probs, {
            "cap": empty_float,
            "has_real_candidate": empty_bool,
            "should_cap": empty_bool,
        }
    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask.to(dtype=torch.bool)
    if torch.is_tensor(max_noop_prob):
        cap = max_noop_prob.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    else:
        cap = torch.full((batch_size,), float(max_noop_prob), dtype=dtype, device=device).clamp(0.0, 1.0)
    active_cap = (cap > 0.0) & (cap < 1.0)

    has_real_candidate = valid_mask[:, 1:].any(dim=-1)
    noop_valid = valid_mask[:, 0]
    noop_prob = probs[:, 0]
    should_cap = active_cap & has_real_candidate & noop_valid & (noop_prob > cap)
    info = {
        "cap": cap,
        "has_real_candidate": has_real_candidate,
        "should_cap": should_cap,
    }
    if not bool(should_cap.any()):
        return probs, info

    capped = probs.clone()
    real_mask = valid_mask[:, 1:].to(dtype=probs.dtype)
    real_mass = (probs[:, 1:] * real_mask).sum(dim=-1).clamp_min(1e-12)
    scale = (1.0 - cap) / real_mass
    capped[should_cap, 0] = cap[should_cap]
    capped[should_cap, 1:] = probs[should_cap, 1:] * real_mask[should_cap] * scale[should_cap].unsqueeze(-1)
    return capped, info
