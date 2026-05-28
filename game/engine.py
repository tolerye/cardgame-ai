from __future__ import annotations

import copy
import random
from typing import List, Optional, Protocol, Sequence

from .cards import (Card, CardKind, Deck, DeckCounts, NUMBER_COUNTS)
from .state import GameConfig, GameState, PlayerState, PlayerStatus


class Agent(Protocol):
    """All agents implement this interface. State passed in is the live
    GameState — agents must not mutate it."""

    def choose_action(self, state: GameState, my_idx: int) -> str:
        """Return 'draw' or 'fold'."""

    def choose_skill_target(self, state: GameState, my_idx: int, kind: CardKind) -> int:
        """Return target player index for EXILE / TRIPLE / forced INSURANCE gift."""


class GameEngine:
    def __init__(self, config: GameConfig, agents: Sequence[Agent]) -> None:
        assert len(agents) == config.num_players
        self.config = config
        self.agents = list(agents)
        self.rng = random.Random(config.seed)
        self.state = GameState(
            config=config,
            players=[PlayerState(index=i) for i in range(config.num_players)],
            remaining=Deck.shuffled(self.rng).counts(),  # filled in per-round
        )
        self._deck: Deck = Deck()  # reshuffled at the start of each round

    # --------------------------------------------------------------- public api
    def play_match(self) -> int:
        """Run rounds until someone hits target_score after a round closes."""
        while not self.state.game_over:
            self.play_round()
        assert self.state.winner is not None
        return self.state.winner

    # --------------------------------------------------------------- cloning
    def clone_for_simulation(self, agents: Sequence[Agent], rng_seed: Optional[int] = None) -> "GameEngine":
        """Deep-copy state and rebuild a deck consistent with `state.remaining`.
        Used by MCTS — caller supplies a (typically rollout-policy) agent set."""
        new = GameEngine.__new__(GameEngine)
        new.config = self.config
        new.agents = list(agents)
        new.rng = random.Random(rng_seed if rng_seed is not None else random.randint(0, 1 << 30))
        new.state = copy.deepcopy(self.state)
        new._deck = self._determinize_deck(new.state.remaining, new.rng)
        return new

    @staticmethod
    def _determinize_deck(counts: DeckCounts, rng: random.Random) -> Deck:
        cards: List[Card] = []
        for v, c in counts.numbers.items():
            cards.extend(Card(CardKind.NUMBER, v) for _ in range(c))
        cards.extend(Card(CardKind.BONUS_FLAT) for _ in range(counts.bonus_flat))
        cards.extend(Card(CardKind.BONUS_DOUBLE) for _ in range(counts.bonus_double))
        cards.extend(Card(CardKind.INSURANCE) for _ in range(counts.insurance))
        cards.extend(Card(CardKind.EXILE) for _ in range(counts.exile))
        cards.extend(Card(CardKind.TRIPLE) for _ in range(counts.triple))
        rng.shuffle(cards)
        return Deck(cards=cards)

    def play_round_to_completion(self) -> None:
        """Continue from the current mid-round state until the round ends.
        Used by MCTS rollouts after a forced first action."""
        st = self.state
        while not st.round_over and not st.game_over:
            if not st.players[st.current_player].is_active:
                self._advance_turn()
                if st.round_over:
                    break
                continue
            self._take_turn(st.current_player)
            if st.round_over:
                break
            self._advance_turn()
        if st.round_over:
            self._settle_round()

    def force_action(self, idx: int, action: str) -> None:
        """Apply a specific first action for player idx without advancing turn.
        Used by MCTS to commit the action being evaluated."""
        st = self.state
        st.current_player = idx
        self._take_turn(idx)

    def play_round(self) -> None:
        self._begin_round()
        while not self.state.round_over:
            self._take_turn(self.state.current_player)
            if self.state.round_over:
                break
            self._advance_turn()
        self._settle_round()

    # ----------------------------------------------------------------- rounds
    def _begin_round(self) -> None:
        st = self.state
        st.round_number += 1
        st.round_over = False
        for p in st.players:
            p.reset_round()
        self._deck = Deck.shuffled(self.rng)
        st.remaining = self._deck.counts()
        st.current_player = st.starter
        st.last_actor = None

    def _settle_round(self) -> None:
        st = self.state
        for p in st.players:
            if p.status == PlayerStatus.ACTIVE:
                # round ended by force (six-burst); active means not yet locked
                p.locked_round_score = sum(p.hand_numbers) + p.bonus_flat_total
            p.total_score += p.locked_round_score
        # next round's starter is the last actor
        if st.last_actor is not None:
            st.starter = st.last_actor
        # game-over check
        max_score = max(p.total_score for p in st.players)
        if max_score >= self.config.target_score:
            st.game_over = True
            # tiebreak: highest score; if tied, lower index wins (stable enough)
            best = max(st.players, key=lambda p: (p.total_score, -p.index))
            st.winner = best.index

    # --------------------------------------------------------------- turn flow
    def _advance_turn(self) -> None:
        st = self.state
        n = self.config.num_players
        nxt = (st.current_player + 1) % n
        # skip non-active players
        for _ in range(n):
            if st.players[nxt].is_active:
                st.current_player = nxt
                return
            nxt = (nxt + 1) % n
        # no active players left → round over
        st.round_over = True

    def _take_turn(self, idx: int) -> None:
        st = self.state
        player = st.players[idx]
        if not player.is_active:
            return
        action = self.agents[idx].choose_action(st, idx)
        st.last_actor = idx
        if action == "fold":
            player.locked_round_score = sum(player.hand_numbers) + player.bonus_flat_total
            player.status = PlayerStatus.FOLDED
            st.add_log(f"FOLD lock={player.locked_round_score}")
        elif action == "draw":
            self._draw_for(idx)
        else:
            raise ValueError(f"unknown action: {action}")
        # round may end if six-burst or no actives remain
        if not any(p.is_active for p in st.players):
            st.round_over = True

    # ----------------------------------------------------------- card handling
    def _draw_for(self, idx: int, force: bool = False) -> None:
        """Draw and resolve one card for player idx. `force` skips active-check
        (used by triple-draw chain)."""
        st = self.state
        player = st.players[idx]
        if not force and not player.is_active:
            return
        if len(self._deck) == 0:
            # extreme edge: deck empty mid-round; treat as forced fold
            player.locked_round_score = sum(player.hand_numbers) + player.bonus_flat_total
            player.status = PlayerStatus.FOLDED
            return
        card = self._deck.draw()
        st.remaining.remove(card)
        self._resolve(idx, card)

    def _resolve(self, idx: int, card: Card) -> None:
        st = self.state
        player = st.players[idx]
        kind = card.kind
        if kind == CardKind.NUMBER:
            self._resolve_number(idx, card.value)
        elif kind == CardKind.BONUS_FLAT:
            # card.value is the flat bonus amount (one of BONUS_FLAT_VALUES)
            player.bonus_flat_total += card.value
            st.add_log(f"BONUS+{card.value}")
        elif kind == CardKind.BONUS_DOUBLE:
            # double the current numeric hand sum (additive: x → 2x)
            cur_sum = sum(player.hand_numbers)
            player.bonus_flat_total += cur_sum  # adds another copy
            st.add_log(f"DOUBLE +{cur_sum}")
        elif kind == CardKind.INSURANCE:
            self._resolve_insurance(idx)
        elif kind == CardKind.EXILE:
            self._resolve_exile(idx)
        elif kind == CardKind.TRIPLE:
            self._resolve_triple(idx)

    def _resolve_number(self, idx: int, value: int) -> None:
        st = self.state
        player = st.players[idx]
        if value in player.unique_numbers:
            # bust check
            if player.has_insurance:
                player.has_insurance = False
                st.add_log(f"BUST_AVOIDED on {value}")
                return
            player.hand_numbers = []
            player.bonus_flat_total = 0
            player.locked_round_score = 0
            player.status = PlayerStatus.BUSTED
            st.add_log(f"BUST on {value}")
            return
        player.hand_numbers.append(value)
        st.add_log(f"DRAW {value}")
        if len(player.unique_numbers) >= 6:
            self._trigger_six_burst(idx)

    def _trigger_six_burst(self, idx: int) -> None:
        st = self.state
        player = st.players[idx]
        bonus = self.config.six_burst_bonus
        player.locked_round_score = sum(player.hand_numbers) + player.bonus_flat_total + bonus
        player.status = PlayerStatus.FOLDED  # treat as locked
        st.add_log(f"SIX-BURST! lock={player.locked_round_score}")
        # everyone else still active gets force-locked at current score
        for other in st.players:
            if other.index != idx and other.status == PlayerStatus.ACTIVE:
                other.locked_round_score = sum(other.hand_numbers) + other.bonus_flat_total
                other.status = PlayerStatus.FOLDED
        st.round_over = True
        st.last_actor = idx

    def _resolve_insurance(self, idx: int) -> None:
        st = self.state
        player = st.players[idx]
        if not player.has_insurance:
            player.has_insurance = True
            st.add_log("INSURANCE+")
            return
        # forced gift to another active player
        candidates = [p.index for p in st.others_active(idx) if not p.has_insurance]
        if not candidates:
            candidates = [p.index for p in st.others_active(idx)]
        if not candidates:
            st.add_log("INSURANCE wasted (no targets)")
            return
        target = self.agents[idx].choose_skill_target(st, idx, CardKind.INSURANCE)
        if target not in candidates:
            target = candidates[0]
        st.players[target].has_insurance = True
        st.add_log(f"INSURANCE -> P{target}")

    def _resolve_exile(self, idx: int) -> None:
        st = self.state
        # spec says "any player" — self-exile is allowed (acts like a forced fold)
        candidates = [p.index for p in st.players if p.is_active]
        if not candidates:
            st.add_log("EXILE wasted (no targets)")
            return
        target = self.agents[idx].choose_skill_target(st, idx, CardKind.EXILE)
        if target not in candidates:
            target = candidates[0]
        victim = st.players[target]
        victim.locked_round_score = sum(victim.hand_numbers) + victim.bonus_flat_total
        victim.status = PlayerStatus.EXILED
        st.add_log(f"EXILE -> P{target} lock={victim.locked_round_score}")

    def _resolve_triple(self, idx: int) -> None:
        st = self.state
        # spec says "any player" — self-targeting is allowed (e.g. force a six-burst attempt)
        candidates = [p.index for p in st.players if p.is_active]
        if not candidates:
            st.add_log("TRIPLE wasted (no targets)")
            return
        target = self.agents[idx].choose_skill_target(st, idx, CardKind.TRIPLE)
        if target not in candidates:
            target = candidates[0]
        st.add_log(f"TRIPLE -> P{target}")
        for _ in range(3):
            if not st.players[target].is_active:
                break
            self._draw_for(target, force=True)
            if st.round_over:
                break
