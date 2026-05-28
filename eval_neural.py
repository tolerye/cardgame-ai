"""Evaluate the trained NeuralAgent against the strong baselines, in parallel.

Each worker plays one full 4-agent match. Workers receive serialized model
weights (bytes) and rebuild the model locally — necessary because PyTorch
models can't be pickled across spawn boundaries reliably."""

from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import time
from typing import Tuple

from agents import EVAgent, ExpectimaxAgent, GreedyAgent
from agents.neural_agent import NeuralAgent
from game import GameConfig, GameEngine


AGENT_NAMES = ("neural", "expectimax", "ev", "greedy")


def _build_agents(weights_bytes: bytes, use_mcts: bool, n_sims: int):
    import torch
    from train.network import build_model
    model = build_model()
    model.load_state_dict(torch.load(io.BytesIO(weights_bytes), map_location="cpu"))
    model.eval()
    return [
        NeuralAgent(model=model, use_mcts=use_mcts, n_simulations=n_sims),
        ExpectimaxAgent(),
        EVAgent(),
        GreedyAgent(),
    ]


def _worker_play(args: Tuple[bytes, bool, int, int]) -> Tuple[int, list]:
    """Play one match, return (winner_idx, [total_scores])."""
    weights_bytes, use_mcts, n_sims, seed = args
    import torch
    torch.set_num_threads(1)
    agents = _build_agents(weights_bytes, use_mcts, n_sims)
    cfg = GameConfig(num_players=4, seed=seed)
    e = GameEngine(cfg, agents)
    winner = e.play_match()
    return winner, [p.total_score for p in e.state.players]


def run_tournament(model_path: str, n_matches: int, use_mcts: bool,
                   n_sims: int, workers: int) -> None:
    import torch
    # Load once in parent, send bytes to workers
    state_dict = torch.load(model_path, map_location="cpu")
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    weights_bytes = buf.getvalue()

    label = "neural_mcts" if use_mcts else "neural"
    names = (label,) + AGENT_NAMES[1:]

    wins = [0] * 4
    totals = [0] * 4
    t0 = time.time()

    if workers <= 1:
        for s in range(n_matches):
            winner, t = _worker_play((weights_bytes, use_mcts, n_sims, s))
            wins[winner] += 1
            for j, v in enumerate(t):
                totals[j] += v
    else:
        ctx = mp.get_context("spawn")
        tasks = [(weights_bytes, use_mcts, n_sims, s) for s in range(n_matches)]
        with ctx.Pool(processes=workers) as pool:
            for winner, t in pool.imap_unordered(_worker_play, tasks, chunksize=1):
                wins[winner] += 1
                for j, v in enumerate(t):
                    totals[j] += v

    elapsed = time.time() - t0
    print(f"\n=== Final Evaluation ({n_matches} matches, {elapsed:.1f}s, "
          f"workers={workers}) ===")
    print(f"{'idx':>3} {'agent':<14} {'wins':>6} {'win%':>7} {'avg total':>10}")
    for j, name in enumerate(names):
        wp = wins[j] / n_matches * 100
        avg = totals[j] / n_matches
        print(f"{j:>3} {name:<14} {wins[j]:>6} {wp:>6.1f}% {avg:>10.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("-n", "--n-matches", type=int, default=200)
    parser.add_argument("--mcts", action="store_true",
                        help="use neural-MCTS at inference (slower, stronger)")
    parser.add_argument("--n-sims", type=int, default=80)
    parser.add_argument("--workers", type=int, default=6,
                        help="parallel match workers; 1 = serial")
    args = parser.parse_args()
    run_tournament(args.model, args.n_matches, args.mcts, args.n_sims, args.workers)


if __name__ == "__main__":
    main()
