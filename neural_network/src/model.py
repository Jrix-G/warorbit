from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import nn


@dataclass
class ModelConfig:
    input_dim: int
    hidden_dim: int = 128


class NeuralNetworkModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.candidate_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + 16, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

    def forward(self, state_features: torch.Tensor, candidate_features: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        if state_features.dim() == 1:
            state_features = state_features.unsqueeze(0)
        latent = self.encoder(state_features.float())
        value = self.value_head(latent).squeeze(-1)
        result = {"value": value, "latent": latent}
        if candidate_features is not None:
            if candidate_features.dim() == 2:
                candidate_features = candidate_features.unsqueeze(0)
            if candidate_features.size(0) != latent.size(0):
                raise ValueError("candidate_features batch size must match state batch size")
            latent_expanded = latent.unsqueeze(1).expand(-1, candidate_features.size(1), -1)
            score_input = torch.cat([latent_expanded, candidate_features.float()], dim=-1)
            policy_logits = self.candidate_head(score_input).squeeze(-1)
            result["policy_logits"] = policy_logits
        return result

    def save_state_dict(self) -> Dict[str, Any]:
        return self.state_dict()

    def load_state(self, state: Dict[str, Any]) -> None:
        self.load_state_dict(state)
