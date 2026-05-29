"""Quick benchmark: vectorized self-play on CPU vs GPU.

Usage:
    python3 bench_vectorized.py                # default: cpu
    python3 bench_vectorized.py --device cuda  # NVIDIA GPU
    python3 bench_vectorized.py --device mps   # Apple Silicon GPU
"""

from __future__ import annotations

import argparse
import time

from train.network import build_model
from train.vectorized_selfplay import vectorized_selfplay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', help='cpu / cuda / mps')
    parser.add_argument('--concurrent', type=int, default=32, help='并发 game 数')
    parser.add_argument('--n-sims', type=int, default=80)
    parser.add_argument('--n-games', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=32, help='单 game MCTS batch')
    args = parser.parse_args()

    print(f"Building model on {args.device}...")
    model = build_model()
    if args.device != 'cpu':
        try:
            import torch
            model = model.to(args.device)
            print(f"Model moved to {args.device}")
        except Exception as e:
            print(f"⚠ failed: {e}")
            return

    print(f"\nVectorized self-play:")
    print(f"  concurrent games  : {args.concurrent}")
    print(f"  MCTS sims/decision: {args.n_sims}")
    print(f"  total games       : {args.n_games}")
    print(f"  inner batch_size  : {args.batch_size}")
    print(f"  effective batch   : ~{args.concurrent * args.batch_size} (peak)")
    print()

    t0 = time.time()
    examples = vectorized_selfplay(
        model,
        n_concurrent=args.concurrent,
        n_sims=args.n_sims,
        n_games_total=args.n_games,
        batch_size=args.batch_size,
        device=args.device,
        verbose=True,
    )
    dt = time.time() - t0

    print()
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  total time     : {dt:.1f}s")
    print(f"  games          : {args.n_games}")
    print(f"  game/sec       : {args.n_games / dt:.2f}")
    print(f"  game/hour      : {args.n_games / dt * 3600:.0f}")
    print(f"  examples total : {len(examples)}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == '__main__':
    main()
