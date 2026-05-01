import numpy as np
import torch
from neural_network.src.model import NeuralNetworkModel, ModelConfig


def test_overfit_simple_action():
    torch.manual_seed(0)
    model = NeuralNetworkModel(ModelConfig(input_dim=4))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    state = torch.zeros(1, 4)
    candidates = torch.zeros(1, 2, 16)
    candidates[0, 1, 0] = 1.0
    target_idx = 1
    before = None
    for _ in range(100):
        out = model(state, candidates)
        probs = torch.softmax(out["policy_logits"], dim=-1)
        before = float(probs[0, target_idx].item())
        loss = -torch.log(probs[0, target_idx] + 1e-8)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    out = model(state, candidates)
    after = float(torch.softmax(out["policy_logits"], dim=-1)[0, target_idx].item())
    assert after > before
