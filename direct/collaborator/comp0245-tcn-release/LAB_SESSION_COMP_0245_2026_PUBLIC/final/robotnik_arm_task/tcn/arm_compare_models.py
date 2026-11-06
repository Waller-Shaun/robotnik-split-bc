"""Arm-only MLP and causal dilated CNN with matched pretrained functions.

The TCN is initialized by putting each pretrained linear matrix at the
current-time tap of a causal convolution and zeroing historical taps.
At initialization both policies are the same current-state function. After
training the TCN can use history. This is transfer learning, not training a
TCN from scratch. Parameter counts and compute are intentionally reported,
not claimed to be matched.
"""
import torch
from torch import nn
from tcn_model import CausalConv1d


class ArmMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(26, 256), nn.ReLU(),
                                     nn.Linear(256, 256), nn.ReLU(),
                                     nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 7))

    def forward(self, x):
        return self.network(x[:, -1] if x.ndim == 3 else x)


class ArmTCN(nn.Module):
    receptive_field = 15

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(CausalConv1d(26, 256, 3, 1), nn.ReLU(),
                                     CausalConv1d(256, 256, 3, 2), nn.ReLU(),
                                     CausalConv1d(256, 128, 3, 4), nn.ReLU(),
                                     nn.Conv1d(128, 7, 1))

    def forward_sequence(self, x):
        return self.network(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        return self.forward_sequence(x)[:, -1]


CLASSES = {"mlp": ArmMLP, "tcn": ArmTCN}


def initialize_from_released(model, state):
    with torch.no_grad():
        for index in (0, 2, 4, 6):
            weight = state[f"network.{index}.weight"]
            bias = state[f"network.{index}.bias"]
            if index == 6:
                weight, bias = weight[4:11], bias[4:11]
            layer = model.network[index]
            if isinstance(layer, CausalConv1d):
                layer.conv.weight.zero_()
                layer.conv.weight[:, :, -1].copy_(weight)
                layer.conv.bias.copy_(bias)
            elif isinstance(layer, nn.Conv1d):
                layer.weight.copy_(weight.unsqueeze(-1))
                layer.bias.copy_(bias)
            else:
                layer.weight.copy_(weight)
                layer.bias.copy_(bias)


def verify_transfer(state):
    torch.manual_seed(42)
    mlp, tcn = ArmMLP().eval(), ArmTCN().eval()
    initialize_from_released(mlp, state)
    initialize_from_released(tcn, state)
    x = torch.randn(8, 20, 26)
    with torch.no_grad():
        torch.testing.assert_close(mlp(x), tcn(x), rtol=1e-5, atol=2e-5)
        changed = x.clone()
        changed[:, 10:] += 4
        torch.testing.assert_close(tcn.forward_sequence(x)[:, :10],
                                   tcn.forward_sequence(changed)[:, :10])
    # Historical coefficients must be trainable, even though initialized zero.
    tcn(x).square().mean().backward()
    assert tcn.network[0].conv.weight.grad[:, :, :-1].abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in tcn.parameters())
    return {"initial_functions_match": True, "causality_passed": True,
            "historical_taps_receive_gradient": True,
            "mlp_parameters": sum(p.numel() for p in mlp.parameters()),
            "tcn_parameters": sum(p.numel() for p in tcn.parameters())}
