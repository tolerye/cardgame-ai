"""Sanity tests: deck composition, six-burst, bust + insurance, exile, triple,
starter inheritance. Run: python -m pytest tests/ or just `python tests/test_engine.py`."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game.cards import (BONUS_FLAT_AMOUNT, Card, CardKind, Deck, build_full_deck)
from game.engine import GameEngine
from game.state import GameConfig, PlayerStatus


# ----------------------------- deck composition ------------------------------
def test_deck_size():
    deck = build_full_deck()
    assert len(deck) == 94
    counts = {}
    for c in deck:
        if c.kind == CardKind.NUMBER:
            counts.setdefault(c.value, 0)
            counts[c.value] += 1
    assert counts[0] == 1
    assert counts[1] == 1
    for n in range(2, 13):
        assert counts[n] == n
    print("✓ deck composition")


# ----------------------------- scripted engine -------------------------------
class ScriptedAgent:
    """Drives the engine through a deterministic action sequence; falls back
    to fold when out of script."""

    def __init__(self, actions: list[str], targets: list[int] | None = None) -> None:
        self.actions = list(actions)
        self.targets = list(targets or [])

    def choose_action(self, state, my_idx):
        if self.actions:
            return self.actions.pop(0)
        return "fold"

    def choose_skill_target(self, state, my_idx, kind):
        if self.targets:
            return self.targets.pop(0)
        for p in state.players:
            if p.is_active and p.index != my_idx:
                return p.index
        return my_idx


def _engine_with_stacked_top(top_cards: list[Card], num_players: int = 4):
    """Build an engine whose deck top (last element) is `top_cards[-1]`, etc."""
    cfg = GameConfig(num_players=num_players, seed=42)
    agents = [ScriptedAgent([]) for _ in range(num_players)]
    engine = GameEngine(cfg, agents)
    # Begin a round to reset deck etc.
    engine.state.round_number += 1
    for p in engine.state.players:
        p.reset_round()
    deck = Deck(cards=list(top_cards))
    engine._deck = deck
    engine.state.remaining = deck.counts()
    engine.state.current_player = 0
    engine.state.starter = 0
    return engine, agents


def test_six_burst():
    cards = [
        Card(CardKind.NUMBER, 1),
        Card(CardKind.NUMBER, 2),
        Card(CardKind.NUMBER, 3),
        Card(CardKind.NUMBER, 4),
        Card(CardKind.NUMBER, 5),
        Card(CardKind.NUMBER, 6),
    ]
    cards = list(reversed(cards))  # so pop() yields 1, 2, 3...
    engine, agents = _engine_with_stacked_top(cards)
    agents[0].actions = ["draw"] * 6
    for _ in range(6):
        engine._take_turn(0)
    p0 = engine.state.players[0]
    assert engine.state.round_over, "six-burst should end round"
    assert p0.locked_round_score == 1 + 2 + 3 + 4 + 5 + 6 + 15
    assert engine.state.last_actor == 0
    print("✓ six-burst lock + last_actor")


def test_bust_no_insurance():
    cards = [Card(CardKind.NUMBER, 5), Card(CardKind.NUMBER, 5)]
    cards = list(reversed(cards))
    engine, agents = _engine_with_stacked_top(cards, num_players=2)
    agents[0].actions = ["draw", "draw"]
    engine._take_turn(0)
    engine._take_turn(0)
    p0 = engine.state.players[0]
    assert p0.status == PlayerStatus.BUSTED
    assert p0.locked_round_score == 0
    print("✓ bust without insurance")


def test_bust_with_insurance():
    cards = [Card(CardKind.NUMBER, 5),
             Card(CardKind.INSURANCE),
             Card(CardKind.NUMBER, 5)]
    cards = list(reversed(cards))
    engine, agents = _engine_with_stacked_top(cards, num_players=2)
    agents[0].actions = ["draw", "draw", "draw"]
    for _ in range(3):
        engine._take_turn(0)
    p0 = engine.state.players[0]
    assert p0.status == PlayerStatus.ACTIVE  # insurance saved us
    assert not p0.has_insurance
    assert p0.hand_numbers == [5]  # second 5 was discarded
    print("✓ bust with insurance consumed")


def test_exile():
    cards = [Card(CardKind.EXILE)]
    engine, agents = _engine_with_stacked_top(cards, num_players=3)
    # P1 has some round score
    engine.state.players[1].hand_numbers = [3, 4]
    agents[0].actions = ["draw"]
    agents[0].targets = [1]
    engine._take_turn(0)
    p1 = engine.state.players[1]
    assert p1.status == PlayerStatus.EXILED
    assert p1.locked_round_score == 7
    print("✓ exile locks target")


def test_triple_chain_busts():
    cards = [
        Card(CardKind.NUMBER, 7),  # 3rd forced draw → bust
        Card(CardKind.NUMBER, 4),  # 2nd
        Card(CardKind.NUMBER, 3),  # 1st (safe)
        Card(CardKind.TRIPLE),     # P0 draws this
    ]
    engine, agents = _engine_with_stacked_top(cards, num_players=2)
    engine.state.players[1].hand_numbers = [7]  # second triple-card 7 will bust
    agents[0].actions = ["draw"]
    agents[0].targets = [1]
    engine._take_turn(0)
    p1 = engine.state.players[1]
    assert p1.status == PlayerStatus.BUSTED, p1.status
    print("✓ triple chain busts target")


def run_all():
    test_deck_size()
    test_six_burst()
    test_bust_no_insurance()
    test_bust_with_insurance()
    test_exile()
    test_triple_chain_busts()
    print("\nAll engine tests passed.")


if __name__ == "__main__":
    run_all()
