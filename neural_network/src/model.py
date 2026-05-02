from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import nn


@dataclass
class ModelConfig:
    input_dim: int
    hidden_dim: int = 256


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


def load_compatible_state_dict(module: nn.Module, state: Dict[str, Any]) -> Dict[str, Any]:
    current = module.state_dict()
    compatible = {}
    skipped = {}
    for key, value in state.items():
        if key not in current:
            skipped[key] = "missing"
            continue
        tensor = torch.as_tensor(value, dtype=current[key].dtype, device=current[key].device)
        if tuple(tensor.shape) != tuple(current[key].shape):
            skipped[key] = f"shape_mismatch:{tuple(tensor.shape)}!= {tuple(current[key].shape)}"
            continue
        compatible[key] = tensor
    current.update(compatible)
    module.load_state_dict(current)
    return {"loaded": list(compatible.keys()), "skipped": skipped}


class NeuralNetworkModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.encoder_blocks = nn.ModuleList([ResidualBlock(cfg.hidden_dim) for _ in range(3)])
        self.candidate_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + 16, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, max(128, cfg.hidden_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(128, cfg.hidden_dim // 2), 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, max(128, cfg.hidden_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(128, cfg.hidden_dim // 2), 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        last_candidate = self.candidate_head[-1]
        if isinstance(last_candidate, nn.Linear):
            nn.init.orthogonal_(last_candidate.weight, gain=0.01)
        last_value = self.value_head[-1]
        if isinstance(last_value, nn.Linear):
            nn.init.orthogonal_(last_value.weight, gain=1.0)

    def forward(self, state_features: torch.Tensor, candidate_features: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        if state_features.dim() == 1:
            state_features = state_features.unsqueeze(0)
        latent = self.input_proj(state_features.float())
        for block in self.encoder_blocks:
            latent = block(latent)
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
        load_compatible_state_dict(self, state)
