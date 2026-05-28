from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set

from .cards import Card, DeckCounts


class PlayerStatus(Enum):
    ACTIVE = "active"
    FOLDED = "folded"
    EXILED = "exiled"
    BUSTED = "busted"


@dataclass
class PlayerState:
    index: int
    total_score: int = 0  # carries across rounds

    # round-local fields, reset every round
    hand_numbers: List[int] = field(default_factory=list)
    bonus_flat_total: int = 0
    has_insurance: bool = False
    locked_round_score: int = 0
    status: PlayerStatus = PlayerStatus.ACTIVE

    @property
    def is_active(self) -> bool:
        return self.status == PlayerStatus.ACTIVE

    @property
    def unique_numbers(self) -> Set[int]:
        return set(self.hand_numbers)

    def current_round_score(self) -> int:
        """Live score if locked right now (for active players)."""
        if self.status == PlayerStatus.BUSTED:
            return 0
        if self.status in (PlayerStatus.FOLDED, PlayerStatus.EXILED):
            return self.locked_round_score
        return sum(self.hand_numbers) + self.bonus_flat_total

    def reset_round(self) -> None:
        self.hand_numbers = []
        self.bonus_flat_total = 0
        self.has_insurance = False
        self.locked_round_score = 0
        self.status = PlayerStatus.ACTIVE


@dataclass
class GameConfig:
    num_players: int = 4
    target_score: int = 200
    six_burst_bonus: int = 15  # extra reward on "6翻了"
    seed: Optional[int] = None


@dataclass
class GameState:
    """Full mutable state of an in-progress match. Engine mutates this; agents
    read it via copy()."""

    config: GameConfig
    players: List[PlayerState]
    remaining: DeckCounts  # multiset of cards still in the deck
    current_player: int = 0
    starter: int = 0  # who starts the current round
    last_actor: Optional[int] = None  # most recent active actor (for next round's starter)
    round_number: int = 0
    round_over: bool = False
    game_over: bool = False
    winner: Optional[int] = None
    log: List[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.config.num_players

    def active_players(self) -> List[PlayerState]:
        return [p for p in self.players if p.is_active]

    def others_active(self, idx: int) -> List[PlayerState]:
        return [p for p in self.players if p.is_active and p.index != idx]

    def add_log(self, msg: str) -> None:
        self.log.append(f"R{self.round_number}|P{self.current_player}: {msg}")
