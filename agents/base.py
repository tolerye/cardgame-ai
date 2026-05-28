from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

from game.cards import CardKind
from game.state import GameState


class BaseAgent(ABC):
    """All agents subclass this. State passed in is the live GameState — agents
    must not mutate it."""

    name: str = "base"

    @abstractmethod
    def choose_action(self, state: GameState, my_idx: int) -> str: ...

    def choose_skill_target(self, state: GameState, my_idx: int, kind: CardKind) -> int:
        # default: any other active player
        for p in state.players:
            if p.index != my_idx and p.is_active:
                return p.index
        return my_idx
