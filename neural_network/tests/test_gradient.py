import torch
from neural_network.src.model import NeuralNetworkModel, ModelConfig


def test_backward_and_grad_norm():
    model = NeuralNetworkModel(ModelConfig(input_dim=8))
    state = torch.randn(4, 8)
    candidates = torch.randn(4, 5, 16)
    out = model(state, candidates)
    loss = out["policy_logits"].sum() + out["value"].sum()
    loss.backward()
    grad_norm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None))
    assert float(grad_norm.item()) > 0
