from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class CardKind(Enum):
    NUMBER = "number"
    BONUS_FLAT = "bonus_flat"      # +10 to round score
    BONUS_DOUBLE = "bonus_double"  # double sum of current numeric hand
    INSURANCE = "insurance"
    EXILE = "exile"
    TRIPLE = "triple"


@dataclass(frozen=True)
class Card:
    kind: CardKind
    value: int = 0  # only meaningful for NUMBER

    def __repr__(self) -> str:
        if self.kind == CardKind.NUMBER:
            return f"N{self.value}"
        return self.kind.value


BONUS_FLAT_VALUES = [2, 4, 6, 8, 10]  # 加分牌：5 张不同面值，各 1 张
BONUS_FLAT_COUNT = len(BONUS_FLAT_VALUES)  # 5
BONUS_FLAT_AVG = sum(BONUS_FLAT_VALUES) / len(BONUS_FLAT_VALUES)  # 6.0，给 agent EV 估算

# Counts per the spec
NUMBER_COUNTS = {0: 1, 1: 1, **{n: n for n in range(2, 13)}}  # total 79
BONUS_DOUBLE_COUNT = 3
SKILL_PER_KIND = 3  # insurance / exile / triple each


def build_full_deck() -> List[Card]:
    deck: List[Card] = []
    for n, count in NUMBER_COUNTS.items():
        deck.extend(Card(CardKind.NUMBER, n) for _ in range(count))
    for v in BONUS_FLAT_VALUES:
        deck.append(Card(CardKind.BONUS_FLAT, v))  # value 是加分点数
    deck.extend(Card(CardKind.BONUS_DOUBLE) for _ in range(BONUS_DOUBLE_COUNT))
    for kind in (CardKind.INSURANCE, CardKind.EXILE, CardKind.TRIPLE):
        deck.extend(Card(kind) for _ in range(SKILL_PER_KIND))
    assert len(deck) == 96, f"expected 96 cards, got {len(deck)}"
    return deck


@dataclass
class Deck:
    """Stack of remaining cards. Maintains both an order (for actual draws) and
    a multiset view for agents to compute exact probabilities."""

    cards: List[Card] = field(default_factory=list)

    @classmethod
    def shuffled(cls, rng: Optional[random.Random] = None) -> "Deck":
        rng = rng or random.Random()
        cards = build_full_deck()
        rng.shuffle(cards)
        return cls(cards=cards)

    def draw(self) -> Card:
        return self.cards.pop()

    def __len__(self) -> int:
        return len(self.cards)

    # --- multiset view for agents -------------------------------------------------
    def counts(self) -> "DeckCounts":
        c = DeckCounts()
        for card in self.cards:
            c.add(card)
        return c


@dataclass
class DeckCounts:
    """Remaining-card multiset; agents use this to compute exact probabilities."""

    numbers: dict = field(default_factory=lambda: {n: 0 for n in range(13)})
    bonus_flat: int = 0
    bonus_double: int = 0
    insurance: int = 0
    exile: int = 0
    triple: int = 0

    @classmethod
    def full(cls) -> "DeckCounts":
        c = cls()
        for n, count in NUMBER_COUNTS.items():
            c.numbers[n] = count
        c.bonus_flat = BONUS_FLAT_COUNT
        c.bonus_double = BONUS_DOUBLE_COUNT
        c.insurance = SKILL_PER_KIND
        c.exile = SKILL_PER_KIND
        c.triple = SKILL_PER_KIND
        return c

    def add(self, card: Card) -> None:
        if card.kind == CardKind.NUMBER:
            self.numbers[card.value] += 1
        elif card.kind == CardKind.BONUS_FLAT:
            self.bonus_flat += 1
        elif card.kind == CardKind.BONUS_DOUBLE:
            self.bonus_double += 1
        elif card.kind == CardKind.INSURANCE:
            self.insurance += 1
        elif card.kind == CardKind.EXILE:
            self.exile += 1
        elif card.kind == CardKind.TRIPLE:
            self.triple += 1

    def remove(self, card: Card) -> None:
        if card.kind == CardKind.NUMBER:
            self.numbers[card.value] -= 1
        elif card.kind == CardKind.BONUS_FLAT:
            self.bonus_flat -= 1
        elif card.kind == CardKind.BONUS_DOUBLE:
            self.bonus_double -= 1
        elif card.kind == CardKind.INSURANCE:
            self.insurance -= 1
        elif card.kind == CardKind.EXILE:
            self.exile -= 1
        elif card.kind == CardKind.TRIPLE:
            self.triple -= 1

    def total(self) -> int:
        return (sum(self.numbers.values()) + self.bonus_flat + self.bonus_double
                + self.insurance + self.exile + self.triple)

    def copy(self) -> "DeckCounts":
        c = DeckCounts()
        c.numbers = dict(self.numbers)
        c.bonus_flat = self.bonus_flat
        c.bonus_double = self.bonus_double
        c.insurance = self.insurance
        c.exile = self.exile
        c.triple = self.triple
        return c
