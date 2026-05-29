"""Policy+value MLP with policy + value heads. PyTorch is loaded lazily so
the rest of the project runs without it installed.

支持可配置网络容量：hidden 和 n_layers。默认 128/3，扩大版 256/5。
推理时可从权重 state_dict 自动推断尺寸（向后兼容旧 checkpoint）。
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

from .encoder import FEATURE_DIM


def build_model(hidden: int = 128, n_layers: int = 3):
    """构造一个 (in → hidden)^n_layers → policy(2) + value(1) 的 MLP。"""
    import torch.nn as nn

    class PolicyValueNet(nn.Module):
        def __init__(self, in_dim: int = FEATURE_DIM,
                     h: int = hidden, layers: int = n_layers) -> None:
            super().__init__()
            mods = []
            mods += [nn.Linear(in_dim, h), nn.ReLU()]
            for _ in range(layers - 1):
                mods += [nn.Linear(h, h), nn.ReLU()]
            self.trunk = nn.Sequential(*mods)
            self.policy_head = nn.Linear(h, 2)
            self.value_head = nn.Sequential(nn.Linear(h, 1), nn.Tanh())
            self.h = h
            self.layers = layers

        def forward(self, x):
            z = self.trunk(x)
            return self.policy_head(z), self.value_head(z).squeeze(-1)

    return PolicyValueNet()


def infer_arch(state_dict) -> tuple[int, int]:
    """从 state_dict 反推 (hidden, n_layers)。

    trunk 是 Linear,ReLU,Linear,ReLU,...，索引 0/2/4/... 是 Linear。
    第一层 weight 形状 [hidden, FEATURE_DIM]。
    """
    h = state_dict["trunk.0.weight"].shape[0]
    # 数 trunk 里有多少 Linear
    layers = 0
    while f"trunk.{layers * 2}.weight" in state_dict:
        layers += 1
    return h, layers


def save(model, path: str) -> None:
    import torch
    torch.save(model.state_dict(), path)


def load(path: str, hidden: Optional[int] = None, n_layers: Optional[int] = None):
    """加载 checkpoint。如果不指定尺寸，从 state_dict 自动反推。"""
    import torch
    state = torch.load(path, map_location="cpu")
    if hidden is None or n_layers is None:
        h, n = infer_arch(state)
        hidden = hidden or h
        n_layers = n_layers or n
    model = build_model(hidden, n_layers)
    model.load_state_dict(state)
    model.eval()
    return model
