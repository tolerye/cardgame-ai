"""Tiny MLP with policy + value heads. PyTorch is loaded lazily so the
rest of the project runs without it installed."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

from .encoder import FEATURE_DIM


def build_model(hidden: int = 128):
    import torch.nn as nn

    class PolicyValueNet(nn.Module):
        def __init__(self, in_dim: int = FEATURE_DIM, h: int = hidden) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, h), nn.ReLU(),
                nn.Linear(h, h), nn.ReLU(),
                nn.Linear(h, h), nn.ReLU(),
            )
            self.policy_head = nn.Linear(h, 2)  # [draw, fold] logits
            self.value_head = nn.Sequential(nn.Linear(h, 1), nn.Tanh())

        def forward(self, x):
            z = self.trunk(x)
            return self.policy_head(z), self.value_head(z).squeeze(-1)

    return PolicyValueNet()


def save(model, path: str) -> None:
    import torch
    torch.save(model.state_dict(), path)


def load(path: str, hidden: int = 128):
    import torch
    model = build_model(hidden)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
