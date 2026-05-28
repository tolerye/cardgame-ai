from __future__ import annotations

import random

from game.cards import CardKind
from game.state import GameState

from .base import BaseAgent


class RandomAgent(BaseAgent):
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose_action(self, state: GameState, my_idx: int) -> str:
        return self.rng.choice(["draw", "fold"])

    def choose_skill_target(self, state: GameState, my_idx: int, kind: CardKind) -> int:
        candidates = [p.index for p in state.players if p.is_active and p.index != my_idx]
        if not candidates:
            return my_idx
        return self.rng.choice(candidates)


class GreedyAgent(BaseAgent):
    """Folds at a fixed score threshold. Useful sanity baseline."""

    name = "greedy"

    def __init__(self, fold_at: int = 28) -> None:
        self.fold_at = fold_at

    def choose_action(self, state: GameState, my_idx: int) -> str:
        me = state.players[my_idx]
        score = sum(me.hand_numbers) + me.bonus_flat_total
        if score >= self.fold_at:
            return "fold"
        return "draw"

    def choose_skill_target(self, state: GameState, my_idx: int, kind: CardKind) -> int:
        # target the leader
        candidates = [p for p in state.players if p.is_active and p.index != my_idx]
        if not candidates:
            return my_idx
        leader = max(candidates, key=lambda p: (p.total_score + p.current_round_score()))
        return leader.index
