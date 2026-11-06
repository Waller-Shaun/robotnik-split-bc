"""Current-state MLP control baseline for the shared BC training pipeline.

Accepts [B,T,26] for the shared loader, but uses ONLY the last state. Hidden
widths (256,256,176) make its parameter count close to the TCN's: 153,083 vs
152,075. The wheel/arm head widths and dropout probability match the TCN.
This comparison changes both temporal context and encoder family; it is not
an architecture-only comparison at equal history length.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MLPPolicy(nn.Module):
    def __init__(self, input_dim: int = 26, hidden_dims: Sequence[int] = (256, 256, 176), dropout: float = .1):
        super().__init__()
        hidden_dims = tuple(hidden_dims)
        if input_dim < 1 or not hidden_dims or any(n < 1 for n in hidden_dims):
            raise ValueError("Input/hidden dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0,1)")
        self.input_dim = input_dim
        layers = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)))
            previous = width
        self.encoder = nn.Sequential(*layers)
        self.wheel_head = nn.Sequential(nn.Linear(previous, 64), nn.ReLU(), nn.Linear(64, 4))
        self.arm_head = nn.Sequential(nn.Linear(previous, 128), nn.ReLU(), nn.Linear(128, 7))

    def forward(self, observation):
        if observation.ndim == 3:
            if observation.shape[1] < 1:
                raise ValueError("History must contain at least one observation")
            observation = observation[:, -1]
        if observation.ndim != 2 or observation.shape[-1] != self.input_dim:
            raise ValueError(f"Expected [B,{self.input_dim}] or [B,T,{self.input_dim}]")
        encoded = self.encoder(observation)
        return torch.cat((self.wheel_head(encoded), self.arm_head(encoded)), dim=-1)
