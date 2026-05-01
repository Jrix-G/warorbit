from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np

from .utils import softmax, sigmoid


@dataclass
class ActionCandidate:
    source_id: int
    target_id: int
    amount: int
    mission: str
    valid: bool = True


MISSION_TYPES = ("do_nothing", "expand", "attack", "defend", "support", "opportunistic")


def build_action_candidates(game: Dict[str, Any]) -> List[ActionCandidate]:
    candidates: List[ActionCandidate] = [ActionCandidate(-1, -1, 0, "do_nothing")]
    my_planets = [p for p in game.get("planets", []) if p["owner"] == game.get("my_id", 0)]
    for src in my_planets:
        for tgt in game.get("planets", []):
            if tgt["id"] == src["id"]:
                continue
            mission = "expand" if tgt["owner"] == -1 else ("attack" if tgt["owner"] != game.get("my_id", 0) else "support")
            candidates.append(ActionCandidate(src["id"], tgt["id"], max(1, int(src["ships"] * 0.2)), mission))
    return candidates


def is_valid_action(candidate: ActionCandidate, game: Dict[str, Any]) -> bool:
    if candidate.mission == "do_nothing":
        return True
    planet = next((p for p in game.get("planets", []) if p["id"] == candidate.source_id), None)
    if planet is None or planet["owner"] != game.get("my_id", 0):
        return False
    if candidate.amount <= 0 or candidate.amount >= planet["ships"]:
        return False
    return candidate.target_id != candidate.source_id


def choose_action(outputs: Dict[str, np.ndarray], game: Dict[str, Any], temperature: float = 1.0, explore: bool = False) -> ActionCandidate:
    candidates = build_action_candidates(game)
    scores = np.asarray(outputs["policy_logits"], dtype=np.float32).reshape(-1)
    if scores.size < len(candidates):
        pad = np.full((len(candidates) - scores.size,), -1e9, dtype=np.float32)
        scores = np.concatenate([scores, pad])
    scores = scores[:len(candidates)]
    mask = np.array([1.0 if is_valid_action(c, game) else 0.0 for c in candidates], dtype=np.float32)
    masked_scores = np.where(mask > 0, scores / max(temperature, 1e-6), -1e9)
    if explore:
        probs = softmax(masked_scores)
        idx = int(np.random.choice(len(candidates), p=probs / probs.sum()))
    else:
        idx = int(np.argmax(masked_scores))
    return candidates[idx]


def reconstruct_action(candidate: ActionCandidate, game: Dict[str, Any], min_ratio: float = 0.1) -> Tuple[int, int, int]:
    if candidate.mission == "do_nothing":
        return (-1, -1, 0)
    planet = next(p for p in game.get("planets", []) if p["id"] == candidate.source_id)
    amount = max(1, min(int(planet["ships"] * max(min_ratio, candidate.amount / max(1, planet["ships"]))), int(planet["ships"]) - 1))
    return candidate.source_id, candidate.target_id, amount

