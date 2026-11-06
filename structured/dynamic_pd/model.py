"""Bounded state-dependent gains; no gain labels, online IK or dynamics calls."""
import torch
from torch import nn


class GainNetwork(nn.Module):
    def __init__(self, mean=None, std=None):
        super().__init__()
        self.register_buffer('mean', torch.zeros(128) if mean is None else mean.clone())
        self.register_buffer('std', torch.ones(128) if std is None else std.clone())
        self.net = nn.Sequential(nn.Linear(128, 32), nn.SiLU(), nn.Linear(32, 14))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('center', torch.tensor([100.] * 7 + [30.] * 7))
        self.register_buffer('span', torch.tensor([20.] * 7 + [6.] * 7))

    def forward(self, hidden):
        delta = torch.tanh(self.net((hidden - self.mean) / self.std))
        return self.center + self.span * delta, delta


def supervised_torque(gains, q_error, predicted_velocity, measured_velocity, predicted_gravity):
    """Differentiable structured final-torque output for expert-action loss."""
    return (gains[:, :7] * q_error + gains[:, 7:] *
            (predicted_velocity - measured_velocity) + predicted_gravity).clamp(-30., 30.)
