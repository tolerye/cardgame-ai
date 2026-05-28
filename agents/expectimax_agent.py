"""Expectimax agent: depth-limited recursive EV with optimal fold/draw choice
at each lookahead step. Strictly stronger than the single-step EVAgent because
it accounts for the value of subsequent decisions after a draw.

Recursion expands ~18 children per node (13 number values + 5 non-number kinds).
Default depth=3 → ~5800 leaves per decision, ~1ms in Python — fast enough."""

from __future__ import annotations

from typing import Optional

from game.cards import BONUS_FLAT_AMOUNT, CardKind, DeckCounts
from game.state import GameState

from .base import BaseAgent

NEG_INF = float("-inf")

# heuristic skill bonuses applied at draw time (positional value of holding the skill)
INSURANCE_GAIN_VALUE = 6.0
EXILE_DRAW_VALUE = 5.0
TRIPLE_DRAW_VALUE = 4.0


class ExpectimaxAgent(BaseAgent):
    name = "expectimax"

    def __init__(self, depth: int = 3, risk_curve: float = 1.0) -> None:
        self.depth = depth
        self.risk_curve = risk_curve

    # ------------------------------------------------------------ public API
    def choose_action(self, state: GameState, my_idx: int) -> str:
        me = state.players[my_idx]
        cur_score = sum(me.hand_numbers) + me.bonus_flat_total
        target = state.config.target_score
        if me.total_score + cur_score >= target:
            return "fold"

        hand_set = frozenset(me.hand_numbers)
        hand_sum = sum(me.hand_numbers)
        counts = state.remaining

        ev_draw = self._draw_ev(
            hand_set, hand_sum, me.bonus_flat_total, me.has_insurance,
            counts, self.depth,
        )
        ev_fold = float(cur_score)

        # risk multiplier: behind → push harder; ahead → take fewer risks
        ev_draw *= self._risk_multiplier(state, my_idx)

        return "draw" if ev_draw > ev_fold else "fold"

    def choose_skill_target(self, state: GameState, my_idx: int, kind: CardKind) -> int:
        # Reuse the EV heuristic — these aren't bottleneck decisions.
        from .ev_agent import EVAgent
        return EVAgent().choose_skill_target(state, my_idx, kind)

    # ----------------------------------------------------------- recursion
    def _best_value(self, hand_set: frozenset, hand_sum: int, bonus: int,
                    insurance: bool, counts: DeckCounts, depth: int) -> float:
        cur = hand_sum + bonus
        if depth == 0:
            return float(cur)
        ev_draw = self._draw_ev(hand_set, hand_sum, bonus, insurance, counts, depth)
        return max(float(cur), ev_draw)

    def _draw_ev(self, hand_set: frozenset, hand_sum: int, bonus: int,
                 insurance: bool, counts: DeckCounts, depth: int) -> float:
        total = counts.total()
        if total <= 0:
            return float(hand_sum + bonus)

        ev = 0.0
        # number cards
        for v in range(13):
            c = counts.numbers[v]
            if c == 0:
                continue
            p = c / total
            if v in hand_set:
                if insurance:
                    new_counts = counts.copy()
                    new_counts.numbers[v] -= 1
                    ev += p * self._best_value(
                        hand_set, hand_sum, bonus, False, new_counts, depth - 1,
                    )
                else:
                    ev += 0.0  # bust → 0 score
            else:
                new_counts = counts.copy()
                new_counts.numbers[v] -= 1
                new_set = hand_set | {v}
                if len(new_set) >= 6:
                    # round ends here with the bonus
                    ev += p * (hand_sum + v + bonus + 15)
                else:
                    ev += p * self._best_value(
                        new_set, hand_sum + v, bonus, insurance, new_counts, depth - 1,
                    )

        # +10 flat bonus
        if counts.bonus_flat > 0:
            p = counts.bonus_flat / total
            new_counts = counts.copy()
            new_counts.bonus_flat -= 1
            ev += p * self._best_value(
                hand_set, hand_sum, bonus + BONUS_FLAT_AMOUNT, insurance, new_counts, depth - 1,
            )

        # bonus double — adds another copy of hand_sum (numeric only)
        if counts.bonus_double > 0:
            p = counts.bonus_double / total
            new_counts = counts.copy()
            new_counts.bonus_double -= 1
            ev += p * self._best_value(
                hand_set, hand_sum, bonus + hand_sum, insurance, new_counts, depth - 1,
            )

        # insurance — gain insurance if not already, else heuristic positional value
        if counts.insurance > 0:
            p = counts.insurance / total
            new_counts = counts.copy()
            new_counts.insurance -= 1
            new_ins = True if not insurance else insurance
            base = self._best_value(hand_set, hand_sum, bonus, new_ins, new_counts, depth - 1)
            extra = INSURANCE_GAIN_VALUE if not insurance else 0.0
            ev += p * (base + extra)

        # exile/triple — modeled as "skill drawn, score unchanged, gain positional value"
        if counts.exile > 0:
            p = counts.exile / total
            new_counts = counts.copy()
            new_counts.exile -= 1
            base = self._best_value(hand_set, hand_sum, bonus, insurance, new_counts, depth - 1)
            ev += p * (base + EXILE_DRAW_VALUE)

        if counts.triple > 0:
            p = counts.triple / total
            new_counts = counts.copy()
            new_counts.triple -= 1
            base = self._best_value(hand_set, hand_sum, bonus, insurance, new_counts, depth - 1)
            ev += p * (base + TRIPLE_DRAW_VALUE)

        return ev

    # ----------------------------------------------------------- modifiers
    def _risk_multiplier(self, state: GameState, my_idx: int) -> float:
        me = state.players[my_idx]
        others = [p.total_score for p in state.players if p.index != my_idx]
        leader = max(others) if others else 0
        deficit = leader - me.total_score
        rel = deficit / max(state.config.target_score, 1)
        m = 1.0 + 0.5 * rel
        m = max(0.7, min(1.5, m))
        return m * self.risk_curve
