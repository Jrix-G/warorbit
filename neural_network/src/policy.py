from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from .trajectory import safe_plan_shot


@dataclass
class ActionCandidate:
    source_id: int
    target_id: int
    amount: int
    mission: str
    score_features: np.ndarray
    valid: bool = True


def _planet_lookup(game: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(p["id"]): p for p in game.get("planets", [])}


def _candidate_prior(candidate: ActionCandidate, game: Dict[str, Any]) -> float:
    if candidate.mission == "do_nothing":
        has_real_action = any(
            p["owner"] == game.get("my_id", 0) and float(p.get("ships", 0.0)) >= 2.0
            for p in game.get("planets", [])
        )
        return -2.5 if has_real_action else 0.0

    planets = _planet_lookup(game)
    src = planets.get(candidate.source_id)
    tgt = planets.get(candidate.target_id)
    if src is None or tgt is None:
        return -2.0

    src_ships = float(src.get("ships", 0.0))
    tgt_ships = float(tgt.get("ships", 0.0))
    distance = float(np.hypot(float(src.get("x", 0.0)) - float(tgt.get("x", 0.0)), float(src.get("y", 0.0)) - float(tgt.get("y", 0.0))))
    production = float(tgt.get("production", 0.0))
    sent_ratio = float(candidate.amount) / max(1.0, src_ships)
    safety_margin = float(candidate.amount) - tgt_ships
    useful_mass = min(25.0, float(candidate.amount)) / 25.0

    prior = -0.025 * distance
    prior += 0.20 * min(10.0, production)
    prior += 0.35 * sent_ratio
    prior += 0.35 * useful_mass
    prior += 0.015 * max(-100.0, min(100.0, safety_margin))
    if candidate.mission == "expand":
        prior += 0.75
        if safety_margin >= 1.0:
            prior += 0.75
        else:
            prior -= min(1.25, 0.18 * abs(safety_margin - 1.0))
    elif candidate.mission == "attack":
        prior += 0.25
        if safety_margin >= 3.0:
            prior += 0.80
        else:
            prior -= 0.80 + min(1.25, 0.16 * abs(safety_margin - 3.0))
    elif candidate.mission == "support":
        prior -= 0.15
    if candidate.mission in {"expand", "attack"} and candidate.amount < 4:
        prior -= 0.60
    if src_ships - candidate.amount < max(2.0, src_ships * 0.20):
        prior -= 0.65
    return float(max(-3.0, min(3.0, prior)))


def build_action_candidates(
    game: Dict[str, Any],
    send_ratios: Sequence[float] | None = None,
    min_expand_attack_ships: int = 1,
    allow_support: bool = True,
) -> List[ActionCandidate]:
    planets = game.get("planets", [])
    my_id = game.get("my_id", 0)
    do_nothing_features = np.zeros(16, dtype=np.float32)
    do_nothing_cand = ActionCandidate(-1, -1, 0, "do_nothing", do_nothing_features)
    do_nothing_cand.score_features[-1] = _candidate_prior(do_nothing_cand, game) / 3.0
    candidates: List[ActionCandidate] = [do_nothing_cand]
    my_planets = [p for p in planets if p["owner"] == my_id and float(p.get("ships", 0.0)) >= 2.0]
    ratios = tuple(float(r) for r in (send_ratios or (0.25, 0.35, 0.5, 0.65, 0.8, 0.95)) if 0.0 < float(r) < 1.0)
    seen: set[tuple[int, int, int]] = set()
    for src in my_planets:
        for tgt in planets:
            if tgt["id"] == src["id"]:
                continue
            distance = float(np.hypot(float(src.get("x", 0.0)) - float(tgt.get("x", 0.0)), float(src.get("y", 0.0)) - float(tgt.get("y", 0.0))))
            mission = "do_nothing"
            if tgt["owner"] == -1:
                mission = "expand"
            elif tgt["owner"] == my_id:
                mission = "support"
            else:
                mission = "attack"
            if mission == "support" and not allow_support:
                continue
            for ratio in ratios:
                amount = max(1, min(int(float(src.get("ships", 0.0)) * ratio), int(float(src.get("ships", 0.0))) - 1))
                if mission in {"expand", "attack"} and amount < int(min_expand_attack_ships):
                    continue
                if safe_plan_shot(src, tgt, game, ships=amount) is None:
                    continue
                key = (int(src["id"]), int(tgt["id"]), int(amount))
                if key in seen:
                    continue
                seen.add(key)
                score_features = np.asarray([
                    float(src["id"]) / max(1.0, float(len(planets))),
                    float(tgt["id"]) / max(1.0, float(len(planets))),
                    float(amount) / max(1.0, float(src.get("ships", 1.0))),
                    0.0 if mission == "do_nothing" else 1.0,
                    distance / 100.0,
                    float(src.get("production", 0.0)) / 10.0,
                    float(tgt.get("production", 0.0)) / 10.0,
                    float(src.get("ships", 0.0)) / 100.0,
                    float(tgt.get("ships", 0.0)) / 100.0,
                    1.0 if tgt["owner"] == -1 else 0.0,
                    1.0 if tgt["owner"] == my_id else 0.0,
                    1.0 if tgt["owner"] not in (-1, my_id) else 0.0,
                    float(src.get("ships", 0.0) - tgt.get("ships", 0.0)) / 100.0,
                    float(src.get("production", 0.0) - tgt.get("production", 0.0)) / 10.0,
                    float(len(my_planets)) / max(1.0, float(len(planets))),
                    0.0,
                ], dtype=np.float32)
                candidate = ActionCandidate(int(src["id"]), int(tgt["id"]), amount, mission, score_features)
                candidate.score_features[-1] = _candidate_prior(candidate, game) / 3.0
                candidates.append(candidate)
    return candidates


def is_valid_action(candidate: ActionCandidate, game: Dict[str, Any], check_trajectory: bool = True) -> bool:
    if candidate.mission == "do_nothing":
        return True
    if not candidate.valid:
        return False
    planets = _planet_lookup(game)
    src = planets.get(candidate.source_id)
    tgt = planets.get(candidate.target_id)
    my_id = game.get("my_id", 0)
    if src is None or tgt is None:
        return False
    if src["owner"] != my_id:
        return False
    if candidate.amount <= 0 or candidate.amount >= int(float(src.get("ships", 0.0))):
        return False
    if candidate.source_id == candidate.target_id:
        return False
    return (not check_trajectory) or safe_plan_shot(src, tgt, game, ships=int(candidate.amount)) is not None


def candidate_valid_mask(
    candidates: Sequence[ActionCandidate],
    game: Dict[str, Any],
    check_trajectory: bool = False,
) -> torch.Tensor:
    return torch.tensor(
        [is_valid_action(candidate, game, check_trajectory=check_trajectory) for candidate in candidates],
        dtype=torch.bool,
    )


def apply_action_mask(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask.to(dtype=torch.bool)
    if valid_mask.dim() != logits.dim():
        raise ValueError("valid_mask must have the same rank as logits")
    if logits.shape != valid_mask.shape:
        raise ValueError("valid_mask must match logits shape")
    masked = logits.clone()
    min_value = torch.finfo(masked.dtype).min if masked.dtype.is_floating_point else -1e9
    masked = masked.masked_fill(~valid_mask, min_value)
    return masked


def cap_do_nothing_probability(
    probs: torch.Tensor,
    valid_mask: torch.Tensor,
    max_noop_prob: float,
) -> torch.Tensor:
    if max_noop_prob <= 0.0 or max_noop_prob >= 1.0 or probs.size(-1) <= 1:
        return probs
    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask.to(dtype=torch.bool)

    has_real_candidate = valid_mask[1:].any()
    if not bool(has_real_candidate and valid_mask[0] and probs[0] > float(max_noop_prob)):
        return probs

    capped = probs.clone()
    real_mask = valid_mask[1:].to(dtype=probs.dtype)
    real_mass = (probs[1:] * real_mask).sum().clamp_min(1e-12)
    capped[0] = float(max_noop_prob)
    capped[1:] = probs[1:] * real_mask * ((1.0 - float(max_noop_prob)) / real_mass)
    return capped


def _noop_prob_cap_for_slot(
    default_cap: float,
    caps_by_slot: Sequence[float] | None,
    action_slot: int,
) -> float:
    if caps_by_slot:
        idx = max(0, min(int(action_slot), len(caps_by_slot) - 1))
        return float(caps_by_slot[idx])
    return float(default_cap)


def _select_deterministic_index(
    probs: torch.Tensor,
    candidates: Sequence[ActionCandidate],
    valid_mask: torch.Tensor,
    noop_cap: float,
    avoid_noop_if_real: bool,
) -> torch.Tensor:
    if not avoid_noop_if_real or float(noop_cap) >= 1.0 or probs.numel() <= 1 or not bool(valid_mask[0]):
        return torch.argmax(probs)
    real_indices = [
        idx for idx, candidate in enumerate(candidates)
        if idx > 0 and candidate.mission != "do_nothing" and bool(valid_mask[idx])
    ]
    if not real_indices:
        return torch.argmax(probs)
    best_idx = torch.argmax(probs)
    if int(best_idx.item()) != 0:
        return best_idx
    real_idx_tensor = torch.tensor(real_indices, dtype=torch.long, device=probs.device)
    real_probs = probs.index_select(0, real_idx_tensor)
    return real_idx_tensor[torch.argmax(real_probs)]


def choose_action(
    outputs: Dict[str, torch.Tensor],
    game: Dict[str, Any],
    temperature: float = 1.0,
    explore: bool = False,
    return_entropy: bool = False,
    send_ratios: Sequence[float] | None = None,
    min_expand_attack_ships: int = 1,
    prior_strength: float = 0.0,
    do_nothing_logit_penalty: float = 0.0,
    do_nothing_prob_cap: float = 1.0,
    do_nothing_prob_caps_by_slot: Sequence[float] | None = None,
    action_slot: int = 0,
    allow_support: bool = True,
    deterministic_avoid_noop_if_real: bool = True,
) -> Tuple[ActionCandidate, torch.Tensor] | Tuple[ActionCandidate, torch.Tensor, torch.Tensor]:
    candidates = build_action_candidates(
        game,
        send_ratios=send_ratios,
        min_expand_attack_ships=min_expand_attack_ships,
        allow_support=allow_support,
    )
    return choose_action_from_candidates(
        outputs,
        game,
        candidates,
        temperature=temperature,
        explore=explore,
        return_entropy=return_entropy,
        prior_strength=prior_strength,
        do_nothing_logit_penalty=do_nothing_logit_penalty,
        do_nothing_prob_cap=do_nothing_prob_cap,
        do_nothing_prob_caps_by_slot=do_nothing_prob_caps_by_slot,
        action_slot=action_slot,
        deterministic_avoid_noop_if_real=deterministic_avoid_noop_if_real,
    )


def choose_action_from_candidates(
    outputs: Dict[str, torch.Tensor],
    game: Dict[str, Any],
    candidates: Sequence[ActionCandidate],
    temperature: float = 1.0,
    explore: bool = False,
    return_entropy: bool = False,
    prior_strength: float = 0.0,
    do_nothing_logit_penalty: float = 0.0,
    do_nothing_prob_cap: float = 1.0,
    do_nothing_prob_caps_by_slot: Sequence[float] | None = None,
    action_slot: int = 0,
    deterministic_avoid_noop_if_real: bool = True,
) -> Tuple[ActionCandidate, torch.Tensor] | Tuple[ActionCandidate, torch.Tensor, torch.Tensor]:
    logits = outputs["policy_logits"]
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if logits.size(-1) != len(candidates):
        raise ValueError("policy_logits must match number of candidates")
    masked_logits = logits[0] / max(temperature, 1e-6)
    if prior_strength:
        priors = torch.tensor([_candidate_prior(c, game) for c in candidates], dtype=torch.float32, device=masked_logits.device)
        masked_logits = masked_logits + float(prior_strength) * priors
    valid_mask = candidate_valid_mask(candidates, game, check_trajectory=False).to(device=masked_logits.device)
    if not bool(valid_mask.any()):
        fallback_idx = next((idx for idx, candidate in enumerate(candidates) if candidate.mission == "do_nothing"), 0)
        fallback = candidates[int(fallback_idx)]
        log_prob = torch.zeros((), dtype=masked_logits.dtype, device=masked_logits.device)
        entropy = torch.zeros((), dtype=masked_logits.dtype, device=masked_logits.device)
        if return_entropy:
            return fallback, log_prob, entropy
        return fallback, log_prob
    real_valid_exists = any(
        candidate.mission != "do_nothing" and bool(valid_mask[idx])
        for idx, candidate in enumerate(candidates)
    )
    if do_nothing_logit_penalty > 0.0 and real_valid_exists:
        penalty = float(do_nothing_logit_penalty)
        for idx, candidate in enumerate(candidates):
            if candidate.mission == "do_nothing" and bool(valid_mask[idx]):
                masked_logits[idx] = masked_logits[idx] - penalty
    masked_logits = apply_action_mask(masked_logits, valid_mask)
    probs = torch.softmax(masked_logits, dim=-1)
    cap = _noop_prob_cap_for_slot(do_nothing_prob_cap, do_nothing_prob_caps_by_slot, action_slot)
    probs = cap_do_nothing_probability(probs, valid_mask, cap)
    dist = Categorical(probs=probs)
    idx = dist.sample() if explore else _select_deterministic_index(
        probs,
        candidates,
        valid_mask,
        noop_cap=cap,
        avoid_noop_if_real=deterministic_avoid_noop_if_real,
    )
    log_prob = dist.log_prob(idx)
    entropy = dist.entropy()
    if return_entropy:
        return candidates[int(idx.item())], log_prob, entropy
    return candidates[int(idx.item())], log_prob


def reconstruct_action(candidate: ActionCandidate, game: Dict[str, Any], min_ratio: float = 0.0) -> Tuple[int, int, int]:
    if candidate.mission == "do_nothing":
        return (-1, -1, 0)
    src = next(p for p in game.get("planets", []) if p["id"] == candidate.source_id)
    requested = int(candidate.amount)
    if min_ratio > 0.0:
        requested = max(requested, int(float(src.get("ships", 0.0)) * min_ratio))
    amount = max(1, min(requested, int(float(src.get("ships", 0.0))) - 1))
    return candidate.source_id, candidate.target_id, amount
