"""Export trained PyTorch model to a compact JSON the web client can load.

Network is a small MLP (62 → 128 → 128 → 128 → 2 policy + 1 value),
~30k parameters, ~120KB JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train.network import build_model


def _round_list(x, prec=5):
    """递归地把列表里所有浮点数四舍五入到 prec 位（缩小 JSON 体积）。"""
    if isinstance(x, list):
        return [_round_list(v, prec) for v in x]
    return round(float(x), prec)


def export(model_path: str, out_path: str) -> None:
    model = build_model()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    sd = model.state_dict()
    out = {
        "arch": {
            "in_dim": 62,
            "hidden": 128,
            "out_policy": 2,
            "trunk_layers": 3,
            "tanh_value": True,
        },
        "trunk": [
            {
                "W": _round_list(sd[f"trunk.{idx}.weight"].tolist()),
                "b": _round_list(sd[f"trunk.{idx}.bias"].tolist()),
            }
            for idx in [0, 2, 4]
        ],
        "policy_head": {
            "W": _round_list(sd["policy_head.weight"].tolist()),
            "b": _round_list(sd["policy_head.bias"].tolist()),
        },
        "value_head": {
            "W": _round_list(sd["value_head.0.weight"].tolist()),
            "b": _round_list(sd["value_head.0.bias"].tolist()),
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    size = Path(out_path).stat().st_size
    print(f"导出: {model_path} → {out_path} ({size/1024:.1f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/model_best.pt")
    ap.add_argument("--out", default="web/model.json")
    args = ap.parse_args()
    export(args.model, args.out)


if __name__ == "__main__":
    main()
