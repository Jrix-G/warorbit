from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from .utils import softmax, sigmoid


@dataclass
class ModelConfig:
    input_dim: int
    hidden_dim: int = 64
    source_dim: int = 64
    target_dim: int = 64
    mission_dim: int = 5
    amount_dim: int = 6


class NeuralNetworkModel:
    def __init__(self, cfg: ModelConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(42)
        h = cfg.hidden_dim
        self.w1 = self.rng.normal(0, 0.1, size=(cfg.input_dim, h)).astype(np.float32)
        self.b1 = np.zeros((h,), dtype=np.float32)
        self.w2 = self.rng.normal(0, 0.1, size=(h, h)).astype(np.float32)
        self.b2 = np.zeros((h,), dtype=np.float32)
        self.w_policy = self.rng.normal(0, 0.1, size=(h, 1 + cfg.mission_dim + cfg.amount_dim)).astype(np.float32)
        self.b_policy = np.zeros((1 + cfg.mission_dim + cfg.amount_dim,), dtype=np.float32)
        self.w_value = self.rng.normal(0, 0.1, size=(h, 1)).astype(np.float32)
        self.b_value = np.zeros((1,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        if x.ndim == 1:
            x = x[None, :]
        z1 = np.tanh(x @ self.w1 + self.b1)
        z2 = np.tanh(z1 @ self.w2 + self.b2)
        policy = z2 @ self.w_policy + self.b_policy
        value = z2 @ self.w_value + self.b_value
        return {"policy_logits": policy, "value": value.squeeze(-1), "latent": z2}

    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w_policy, self.b_policy, self.w_value, self.b_value]

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in {
            "w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2,
            "w_policy": self.w_policy, "b_policy": self.b_policy,
            "w_value": self.w_value, "b_value": self.b_value,
        }.items()}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        for k, v in state.items():
            setattr(self, k, np.asarray(v, dtype=np.float32))

