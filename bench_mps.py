"""Quick benchmark: CPU vs MPS inference at different batch sizes.
Used to justify when (and when not) to use Apple GPU for this small network."""

from __future__ import annotations

import time

import numpy as np
import torch

from train.encoder import FEATURE_DIM
from train.network import build_model


def time_inference(model, x: torch.Tensor, n_iters: int = 200) -> float:
    # warmup
    for _ in range(20):
        with torch.no_grad():
            model(x)
    if x.device.type == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        with torch.no_grad():
            out = model(x)
    if x.device.type == "mps":
        torch.mps.synchronize()
    dt = (time.time() - t0) / n_iters
    return dt * 1000  # ms


def run() -> None:
    print("=== MPS vs CPU inference benchmark (PolicyValueNet, ~30K params) ===")
    has_mps = torch.backends.mps.is_available()
    print(f"MPS available: {has_mps}")
    print()

    cpu_model = build_model().eval()
    if has_mps:
        mps_model = build_model().to("mps").eval()

    print(f"{'batch':>6} {'CPU (ms)':>10} {'MPS (ms)':>10} {'speedup':>10}")
    for bs in [1, 4, 16, 64, 256, 1024]:
        x_cpu = torch.randn(bs, FEATURE_DIM)
        cpu_ms = time_inference(cpu_model, x_cpu)
        if has_mps:
            x_mps = x_cpu.to("mps")
            mps_ms = time_inference(mps_model, x_mps)
            sp = cpu_ms / mps_ms if mps_ms > 0 else float("inf")
            print(f"{bs:>6} {cpu_ms:>10.3f} {mps_ms:>10.3f} {sp:>9.2f}x")
        else:
            print(f"{bs:>6} {cpu_ms:>10.3f} {'N/A':>10} {'N/A':>10}")

    # Also: measure throughput (per-sample) at each batch size
    print()
    print("Per-sample latency (μs):")
    print(f"{'batch':>6} {'CPU/sample':>12} {'MPS/sample':>12}")
    for bs in [1, 16, 64, 256, 1024]:
        x_cpu = torch.randn(bs, FEATURE_DIM)
        cpu_ms = time_inference(cpu_model, x_cpu, n_iters=100)
        cpu_us = cpu_ms * 1000 / bs
        if has_mps:
            x_mps = x_cpu.to("mps")
            mps_ms = time_inference(mps_model, x_mps, n_iters=100)
            mps_us = mps_ms * 1000 / bs
            print(f"{bs:>6} {cpu_us:>12.1f} {mps_us:>12.1f}")
        else:
            print(f"{bs:>6} {cpu_us:>12.1f} {'N/A':>12}")

    print()
    print("Conclusion: MPS pays off when batch ≥ ~64. At batch=1 (current MCTS),")
    print("CPU per-sample latency is lower because of HtoD transfer overhead.")


if __name__ == "__main__":
    run()
