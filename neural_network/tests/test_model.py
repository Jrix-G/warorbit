import numpy as np
from neural_network.src.model import NeuralNetworkModel, ModelConfig


def test_forward_runs():
    model = NeuralNetworkModel(ModelConfig(input_dim=10))
    out = model.forward(np.zeros((2, 10), dtype=np.float32))
    assert out["policy_logits"].shape[0] == 2
    assert out["value"].shape == (2,)

