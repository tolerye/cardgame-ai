"""Headless tournament runner. Plays many matches between four agents and
prints win rates and average totals."""

from __future__ import annotations

import argparse
import copy
import time
from typing import Callable, List

from agents import EVAgent, ExpectimaxAgent, GreedyAgent, MCTSAgent, RandomAgent
from agents.base import BaseAgent
from game import GameConfig, GameEngine


AgentFactory = Callable[[], BaseAgent]


REGISTRY: dict[str, AgentFactory] = {
    "random": lambda: RandomAgent(),
    "greedy": lambda: GreedyAgent(),
    "greedy20": lambda: GreedyAgent(fold_at=20),
    "greedy35": lambda: GreedyAgent(fold_at=35),
    "ev": lambda: EVAgent(),
    "ev_aggressive": lambda: EVAgent(risk_curve=1.2),
    "exmax2": lambda: ExpectimaxAgent(depth=2),
    "exmax3": lambda: ExpectimaxAgent(depth=3),
    "exmax4": lambda: ExpectimaxAgent(depth=4),
    "mcts": lambda: MCTSAgent(n_simulations=100),
    "mcts_strong": lambda: MCTSAgent(n_simulations=400),
    "mcts_fast": lambda: MCTSAgent(n_simulations=60,
                                    rollout_factory=lambda: EVAgent()),
}


def make_agents(names: List[str]) -> List[BaseAgent]:
    return [REGISTRY[name]() for name in names]


def run_match(names: List[str], seed: int) -> tuple[int, list[int]]:
    cfg = GameConfig(num_players=len(names), seed=seed)
    agents = make_agents(names)
    engine = GameEngine(cfg, agents)
    winner = engine.play_match()
    totals = [p.total_score for p in engine.state.players]
    return winner, totals


def tournament(names: List[str], n_matches: int, base_seed: int = 1) -> None:
    wins = [0] * len(names)
    sum_totals = [0] * len(names)
    busts_per_match = []
    t0 = time.time()
    for i in range(n_matches):
        winner, totals = run_match(names, seed=base_seed + i)
        wins[winner] += 1
        for j, t in enumerate(totals):
            sum_totals[j] += t
    elapsed = time.time() - t0
    print(f"\n{n_matches} matches | {elapsed:.1f}s | {n_matches/elapsed:.0f} matches/s")
    print(f"{'idx':>3} {'agent':<14} {'wins':>6} {'win%':>7} {'avg total':>10}")
    for j, name in enumerate(names):
        wp = wins[j] / n_matches * 100
        avg = sum_totals[j] / n_matches
        print(f"{j:>3} {name:<14} {wins[j]:>6} {wp:>6.1f}% {avg:>10.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs=4, default=["ev", "greedy", "greedy20", "random"])
    parser.add_argument("-n", "--n-matches", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    tournament(args.agents, args.n_matches, args.seed)


if __name__ == "__main__":
    main()
