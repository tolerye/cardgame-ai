"""Fixed-length encoder for GameState → numpy feature vector.

Used by the neural network agent and self-play. Layout is documented in
FEATURE_LAYOUT for reproducibility."""

from __future__ import annotations

from typing import List

import numpy as np

from game.state import GameState, PlayerStatus


# Layout (4-player game; configurable via num_players):
#   per-self block      : 13 (hand counts) + 1 (bonus_flat) + 1 (insurance)
#                         + 1 (total/200)  + 1 (current round score / 200)
#                         + 1 (unique count / 6)         = 19
#   per-opponent block ×3: total/200, round_score/200, has_insurance,
#                         unique_count/6, status (4 one-hot)         = 8 each = 24
#   global              : remaining numbers (13) + bonus_flat + bonus_double
#                         + insurance + exile + triple
#                         + deck_size/94 + round_number/20            = 20
# total                 : 19 + 24 + 20 = 63

PER_SELF = 18
PER_OPP = 8
GLOBAL = 20


def encode_state(state: GameState, my_idx: int) -> np.ndarray:
    n = state.config.num_players
    target = state.config.target_score
    me = state.players[my_idx]

    feats: List[float] = []

    # --- self ---
    counts = [0] * 13
    for v in me.hand_numbers:
        counts[v] += 1
    feats.extend(counts)
    feats.append(me.bonus_flat_total / 50.0)
    feats.append(1.0 if me.has_insurance else 0.0)
    feats.append(me.total_score / target)
    feats.append((sum(me.hand_numbers) + me.bonus_flat_total) / target)
    feats.append(len(me.unique_numbers) / 6.0)

    # --- opponents (sorted by relative seat order so policy is symmetric over seat) ---
    others = [(p.index - my_idx) % n for p in state.players if p.index != my_idx]
    ordered = sorted(state.players, key=lambda p: (p.index - my_idx) % n)
    ordered = [p for p in ordered if p.index != my_idx]
    for p in ordered:
        feats.append(p.total_score / target)
        feats.append(p.current_round_score() / target)
        feats.append(1.0 if p.has_insurance else 0.0)
        feats.append(len(p.unique_numbers) / 6.0)
        for s in (PlayerStatus.ACTIVE, PlayerStatus.FOLDED,
                  PlayerStatus.EXILED, PlayerStatus.BUSTED):
            feats.append(1.0 if p.status == s else 0.0)

    # pad to 3 opponents (always; if num_players != 4 the model needs retraining)
    while len(ordered) < 3:
        feats.extend([0.0] * PER_OPP)
        ordered.append(None)

    # --- global ---
    rem = state.remaining
    deck_total = rem.total()
    for v in range(13):
        feats.append(rem.numbers[v] / max(deck_total, 1))
    feats.append(rem.bonus_flat / max(deck_total, 1))
    feats.append(rem.bonus_double / max(deck_total, 1))
    feats.append(rem.insurance / max(deck_total, 1))
    feats.append(rem.exile / max(deck_total, 1))
    feats.append(rem.triple / max(deck_total, 1))
    feats.append(deck_total / 94.0)
    feats.append(state.round_number / 20.0)

    return np.array(feats, dtype=np.float32)


FEATURE_DIM = PER_SELF + PER_OPP * 3 + GLOBAL  # = 63
