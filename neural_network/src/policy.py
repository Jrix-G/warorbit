from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.distributions import Categorical


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


def build_action_candidates(game: Dict[str, Any]) -> List[ActionCandidate]:
    planets = game.get("planets", [])
    my_id = game.get("my_id", 0)
    lookup = _planet_lookup(game)
    candidates: List[ActionCandidate] = [ActionCandidate(-1, -1, 0, "do_nothing", np.zeros(16, dtype=np.float32))]
    my_planets = [p for p in planets if p["owner"] == my_id and float(p.get("ships", 0.0)) >= 2.0]
    for src in my_planets:
        for tgt in planets:
            if tgt["id"] == src["id"]:
                continue
            distance = float(np.hypot(float(src.get("x", 0.0)) - float(tgt.get("x", 0.0)), float(src.get("y", 0.0)) - float(tgt.get("y", 0.0))))
            amount = max(1, int(float(src.get("ships", 0.0)) * 0.5))
            mission = "do_nothing"
            if tgt["owner"] == -1:
                mission = "expand"
            elif tgt["owner"] == my_id:
                mission = "support"
            else:
                mission = "attack"
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
                1.0,
            ], dtype=np.float32)
            candidates.append(ActionCandidate(int(src["id"]), int(tgt["id"]), amount, mission, score_features))
    return candidates


def is_valid_action(candidate: ActionCandidate, game: Dict[str, Any]) -> bool:
    if candidate.mission == "do_nothing":
        return True
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
    return candidate.source_id != candidate.target_id


def choose_action(
    outputs: Dict[str, torch.Tensor],
    game: Dict[str, Any],
    temperature: float = 1.0,
    explore: bool = False,
    return_entropy: bool = False,
) -> Tuple[ActionCandidate, torch.Tensor] | Tuple[ActionCandidate, torch.Tensor, torch.Tensor]:
    candidates = build_action_candidates(game)
    features = torch.tensor(np.stack([c.score_features for c in candidates]), dtype=torch.float32, device=outputs["latent"].device)
    latent = outputs["latent"][:1]
    logits = outputs["policy_logits"]
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if logits.size(-1) != len(candidates):
        raise ValueError("policy_logits must match number of candidates")
    masked_logits = logits[0] / max(temperature, 1e-6)
    valid_mask = torch.tensor([is_valid_action(c, game) for c in candidates], dtype=torch.bool, device=masked_logits.device)
    masked_logits = masked_logits.masked_fill(~valid_mask, -1e9)
    probs = torch.softmax(masked_logits, dim=-1)
    dist = Categorical(probs=probs)
    idx = dist.sample() if explore else torch.argmax(probs)
    log_prob = dist.log_prob(idx)
    entropy = dist.entropy()
    if return_entropy:
        return candidates[int(idx.item())], log_prob, entropy
    return candidates[int(idx.item())], log_prob


def reconstruct_action(candidate: ActionCandidate, game: Dict[str, Any], min_ratio: float = 0.5) -> Tuple[int, int, int]:
    if candidate.mission == "do_nothing":
        return (-1, -1, 0)
    src = next(p for p in game.get("planets", []) if p["id"] == candidate.source_id)
    amount = max(1, min(int(float(src.get("ships", 0.0)) * min_ratio), int(float(src.get("ships", 0.0))) - 1))
    return candidate.source_id, candidate.target_id, amount
