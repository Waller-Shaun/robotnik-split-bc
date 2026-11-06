"""Causal temporal convolution policy for Robotnik expert-action regression.

Public input: [batch, time, 26], oldest observation first.
Public output: [batch, 11], wheels first (4), then arm torques (7).
The trainer must normalize inputs/targets using training-set statistics, and
inference must undo action normalization before applying physical limits.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    """Left-only padding preserves time length without looking into the future."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=0
        )

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class TemporalResidualBlock(nn.Module):
    """Two causal convolutions at the same dilation, with a residual path.

    No batch/time normalization is used: intermediate features at time t
    must not depend on later timesteps through normalization statistics.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.skip = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.layers(x) + self.skip(x))


class TCNPolicy(nn.Module):
    """Shared temporal encoder with separate wheel and arm output heads."""

    def __init__(
        self,
        input_dim: int = 26,
        channels: Sequence[int] = (64, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        channels = tuple(channels)
        if input_dim < 1 or not channels or any(c < 1 for c in channels):
            raise ValueError("input_dim and all channel counts must be positive")
        if kernel_size < 2:
            raise ValueError("kernel_size must be >= 2 for temporal modeling")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.input_dim = input_dim
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.dilations = tuple(2 ** i for i in range(len(channels)))
        # Two convolutions per block. Finite input windows may be shorter
        # than this nominal receptive field; unavailable positions are padded.
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(self.dilations)

        blocks = []
        previous = input_dim
        for width, dilation in zip(channels, self.dilations):
            blocks.append(TemporalResidualBlock(
                previous, width, kernel_size, dilation, dropout
            ))
            previous = width
        self.encoder = nn.Sequential(*blocks)
        self.wheel_head = nn.Sequential(nn.Linear(previous, 64), nn.ReLU(), nn.Linear(64, 4))
        self.arm_head = nn.Sequential(nn.Linear(previous, 128), nn.ReLU(), nn.Linear(128, 7))

    def _encode(self, history):
        if history.ndim != 3 or history.shape[-1] != self.input_dim:
            raise ValueError(f"Expected [batch, time, {self.input_dim}], got {tuple(history.shape)}")
        if history.shape[1] < 1:
            raise ValueError("History must contain at least one observation")
        return self.encoder(history.transpose(1, 2)).transpose(1, 2)

    def _heads(self, encoded):
        return torch.cat((self.wheel_head(encoded), self.arm_head(encoded)), dim=-1)

    def forward(self, history):
        """Predict the action aligned with the LAST observation of the window."""
        return self._heads(self._encode(history)[:, -1])

    def forward_sequence(self, history):
        """Per-time predictions [B,T,11], used to verify causal computation."""
        return self._heads(self._encode(history))


def main():
    parser = argparse.ArgumentParser(description="TCN architecture checks; no dataset training")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    torch.manual_seed(42)
    model = TCNPolicy()
    model.eval()
    x = torch.randn(32, 20, 26)

    with torch.no_grad():
        prediction = model(x)
        assert prediction.shape == (32, 11)
        original = model.forward_sequence(x)
        changed = x.clone()
        changed[:, 10:] += 10 * torch.randn_like(changed[:, 10:])
        perturbed = model.forward_sequence(changed)
        torch.testing.assert_close(original[:, :10], perturbed[:, :10])
        torch.testing.assert_close(prediction, original[:, -1])
        assert model(x[:, :1]).shape == (32, 11)

    # Check that older observations can actually affect the current action.
    historical_input = x[:2].clone().requires_grad_()
    model(historical_input).square().mean().backward()
    assert historical_input.grad[:, :-1].abs().sum().item() > 0
    print(f"Input batch: {tuple(x.shape)}; output batch: {tuple(prediction.shape)}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Channels: {model.channels}; dilations: {model.dilations}")
    print(f"Nominal receptive field: {model.receptive_field} samples; supplied history: 20")
    print("PASS: output shape, causal prefix, history dependence, single-frame input")

    if args.device == "auto":
        selected = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        selected = args.device
    device = torch.device(selected)
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    output = model(x.to(device))
    loss = F.mse_loss(output, torch.randn(32, 11, device=device))
    assert torch.isfinite(loss).item()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
    optimizer.step()
    assert all(torch.isfinite(p).all().item() for p in model.parameters())
    print(f"Device: {device}")
    print("PASS: finite forward, backward, and optimizer step on synthetic data")
    print("Architecture check only: no expert-data training and no checkpoint saved.")


if __name__ == "__main__":
    main()
