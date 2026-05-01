import torch
from neural_network.src.model import NeuralNetworkModel, ModelConfig


def test_forward_runs():
    model = NeuralNetworkModel(ModelConfig(input_dim=10))
    state = torch.zeros(2, 10)
    candidates = torch.zeros(2, 3, 16)
    out = model(state, candidates)
    assert out["policy_logits"].shape == (2, 3)
    assert out["value"].shape == (2,)
