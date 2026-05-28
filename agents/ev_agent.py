"""EV-based agent: precise probability tracking + single-step expected value
with several handcrafted corrections (six-burst sprint, deficit pressure,
endgame guard, skill-target heuristics)."""

from __future__ import annotations

from typing import Optional

from game.cards import BONUS_FLAT_AMOUNT, CardKind, DeckCounts
from game.state import GameState, PlayerState

from .base import BaseAgent

# --- skill heuristic values (used as their EV contribution when drawn) -------
INSURANCE_SELF_VALUE = 8.0   # one extra safe draw, roughly
EXILE_VALUE = 6.0            # locking an opponent ≈ steal their next decision
TRIPLE_VALUE = 5.0


def _safe_draw_count(counts: DeckCounts, hand_set: set[int]) -> int:
    """Number of distinct-number cards left that won't bust me."""
    return sum(c for v, c in counts.numbers.items() if v not in hand_set)


def _bust_count(counts: DeckCounts, hand_set: set[int]) -> int:
    return sum(counts.numbers[v] for v in hand_set)


class EVAgent(BaseAgent):
    """Stage ① + ②: exact-probability EV plus dynamic fold threshold,
    six-burst sprint mode, and skill-target heuristics."""

    name = "ev"

    def __init__(
        self,
        *,
        risk_curve: float = 1.0,        # >1 → more aggressive draws
        skill_value_scale: float = 1.0,  # tune skill EV contribution
    ) -> None:
        self.risk_curve = risk_curve
        self.skill_value_scale = skill_value_scale

    # --------------------------------------------------------------- main API
    def choose_action(self, state: GameState, my_idx: int) -> str:
        me = state.players[my_idx]
        cur_score = sum(me.hand_numbers) + me.bonus_flat_total
        target = state.config.target_score

        # Endgame guard: folding now ends the match in our favour.
        if me.total_score + cur_score >= target:
            return "fold"

        ev_draw = self._ev_draw(state, my_idx)
        ev_fold = float(cur_score)

        # Risk modifier: scale draw EV by deficit / lead.
        ev_draw *= self._risk_multiplier(state, my_idx)

        # Sprint mode: with 5 distinct numbers, six-burst is huge — bias towards draw.
        if len(me.unique_numbers) == 5 and self._sprint_attractive(state, me):
            return "draw"

        return "draw" if ev_draw > ev_fold else "fold"

    # --------------------------------------------------------------- EV core
    def _ev_draw(self, state: GameState, my_idx: int) -> float:
        me = state.players[my_idx]
        counts = state.remaining
        total = counts.total()
        if total <= 0:
            return 0.0

        cur_score = sum(me.hand_numbers) + me.bonus_flat_total
        hand_sum_numeric = sum(me.hand_numbers)
        hand_set = me.unique_numbers
        has_ins = me.has_insurance
        n_unique = len(hand_set)
        ev = 0.0

        # numeric outcomes
        for v in range(13):
            c = counts.numbers[v]
            if c == 0:
                continue
            p = c / total
            if v in hand_set:
                # bust path
                if has_ins:
                    # insurance consumed, score unchanged, can still act later
                    ev += p * cur_score
                else:
                    ev += p * 0.0
            else:
                new_unique = n_unique + 1
                if new_unique >= 6:
                    lock = cur_score + v + state.config.six_burst_bonus
                    ev += p * lock
                else:
                    # post-draw value: assume we make a follow-up optimal choice next turn
                    # Approximation: EV after = max(new_score, EV_continue_from_next_state)
                    # Use new_score as a conservative lower bound; correction comes from
                    # risk_curve. Cheap and effective.
                    ev += p * (cur_score + v)

        # +10 flat bonus
        if counts.bonus_flat > 0:
            p = counts.bonus_flat / total
            ev += p * (cur_score + BONUS_FLAT_AMOUNT)

        # double current numeric hand sum (adds another copy of hand_sum_numeric)
        if counts.bonus_double > 0:
            p = counts.bonus_double / total
            ev += p * (cur_score + hand_sum_numeric)

        # skills: numeric score unchanged, plus heuristic positional value
        scale = self.skill_value_scale
        if counts.insurance > 0:
            p = counts.insurance / total
            ins_val = INSURANCE_SELF_VALUE if not has_ins else 2.0
            ev += p * (cur_score + ins_val * scale)
        if counts.exile > 0:
            p = counts.exile / total
            ev += p * (cur_score + EXILE_VALUE * scale)
        if counts.triple > 0:
            p = counts.triple / total
            ev += p * (cur_score + TRIPLE_VALUE * scale)

        return ev

    # ---------------------------------------------------------- modifiers
    def _risk_multiplier(self, state: GameState, my_idx: int) -> float:
        me = state.players[my_idx]
        others = [p.total_score for p in state.players if p.index != my_idx]
        leader = max(others) if others else 0
        deficit = leader - me.total_score  # positive means we're behind

        target = state.config.target_score
        # If we're far behind near endgame, push harder.
        rel = deficit / max(target, 1)
        m = 1.0 + 0.6 * rel  # behind by 50% of target → +30%
        # Clamp
        m = max(0.7, min(1.6, m))
        return m * self.risk_curve

    def _sprint_attractive(self, state: GameState, me: PlayerState) -> bool:
        counts = state.remaining
        total = counts.total()
        if total <= 0:
            return False
        hand_set = me.unique_numbers
        good = _safe_draw_count(counts, hand_set)
        bad = _bust_count(counts, hand_set)
        # Six-burst attempt: if P(good) > P(bad) (good leads to +sum+15, bad to bust),
        # almost always worth it — unless we're already winning by folding.
        cur_score = sum(me.hand_numbers) + me.bonus_flat_total
        if me.total_score + cur_score >= state.config.target_score:
            return False
        if me.has_insurance:
            return True  # net free shot
        return good >= bad

    # ------------------------------------------------------------ skill targeting
    def choose_skill_target(self, state: GameState, my_idx: int, kind: CardKind) -> int:
        candidates = [p for p in state.players if p.is_active and p.index != my_idx]
        if not candidates:
            return my_idx

        if kind == CardKind.EXILE:
            # take the leader's locked round score from them — pick the one whose
            # current round score is highest (we deny them upside).
            target = max(candidates, key=lambda p: (p.current_round_score(),
                                                     p.total_score))
            return target.index

        if kind == CardKind.TRIPLE:
            # prefer victims most likely to bust: lots of unique numbers, no insurance.
            counts = state.remaining
            total = max(counts.total(), 1)

            def bust_prob_3(p: PlayerState) -> float:
                # crude: P(at least one bust over 3 draws)
                hand_set = p.unique_numbers
                bad = _bust_count(counts, hand_set)
                p_bad = bad / total
                # without insurance one bust ends them
                p_safe_one = 1.0 - p_bad
                return 1.0 - p_safe_one ** 3

            return max(candidates, key=lambda p: (
                0 if p.has_insurance else 1,  # prefer no-insurance victims
                bust_prob_3(p),
                p.current_round_score(),
            )).index

        if kind == CardKind.INSURANCE:
            # forced gift — give to weakest active opponent (least threatening).
            return min(candidates, key=lambda p: (p.total_score + p.current_round_score(),
                                                   p.index)).index

        return candidates[0].index
