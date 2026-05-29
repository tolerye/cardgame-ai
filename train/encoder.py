"""Fixed-length encoder for GameState → numpy feature vector.

Used by the neural network agent and self-play. Layout is documented in
FEATURE_LAYOUT for reproducibility.

v2 layout (这次升级，FEATURE_DIM = 80):
- 原始 62 维不变
- 新增 18 维"手工领域特征"，把模型本来要从原始计数中学的东西提前算好
"""

from __future__ import annotations

from typing import List

import numpy as np

from game.state import GameState, PlayerStatus


# 原始布局：18 + 8*3 + 20 = 62
PER_SELF = 18
PER_OPP = 8
GLOBAL = 20

# 新增：18 维手工领域特征
HANDCRAFTED = 18

FEATURE_DIM = PER_SELF + PER_OPP * 3 + GLOBAL + HANDCRAFTED  # = 80


# 加分牌 +2/+4/+6/+8/+10 各 1 张，平均 6
BONUS_FLAT_AVG = 6.0


def encode_state(state: GameState, my_idx: int) -> np.ndarray:
    n = state.config.num_players
    target = state.config.target_score
    me = state.players[my_idx]

    feats: List[float] = []

    # ========== 原始 self (18) ==========
    counts = [0] * 13
    for v in me.hand_numbers:
        counts[v] += 1
    feats.extend(counts)
    feats.append(me.bonus_flat_total / 50.0)
    feats.append(1.0 if me.has_insurance else 0.0)
    feats.append(me.total_score / target)
    cur_score = sum(me.hand_numbers) + me.bonus_flat_total
    feats.append(cur_score / target)
    feats.append(len(me.unique_numbers) / 6.0)

    # ========== 原始 opponents (8*3=24) ==========
    ordered = state.players
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
    while len(ordered) < 3:
        feats.extend([0.0] * PER_OPP)
        ordered.append(None)

    # ========== 原始 global (20) ==========
    rem = state.remaining
    deck_total = rem.total()
    denom = max(deck_total, 1)
    for v in range(13):
        feats.append(rem.numbers[v] / denom)
    feats.append(rem.bonus_flat / denom)
    feats.append(rem.bonus_double / denom)
    feats.append(rem.insurance / denom)
    feats.append(rem.exile / denom)
    feats.append(rem.triple / denom)
    feats.append(deck_total / 94.0)
    feats.append(state.round_number / 20.0)

    # ========== 手工领域特征 (18) ==========
    hand_set = me.unique_numbers
    n_unique = len(hand_set)

    # --- 摸下一张的概率分解（5 个）---
    bust_count = sum(rem.numbers[v] for v in hand_set)
    safe_count = sum(c for v, c in rem.numbers.items() if v not in hand_set)
    p_bust = bust_count / denom
    p_safe = safe_count / denom
    p_bonus = (rem.bonus_flat + rem.bonus_double) / denom
    p_skill = (rem.insurance + rem.exile + rem.triple) / denom
    avg_safe = (sum(v * c for v, c in rem.numbers.items() if v not in hand_set)
                / safe_count if safe_count > 0 else 0.0)
    feats.append(p_bust)
    feats.append(p_safe)
    feats.append(p_bonus)
    feats.append(p_skill)
    feats.append(avg_safe / 12.0)

    # --- EV 启发式（3 个，跟 expectimax 思路一致的简化版）---
    hand_sum = sum(me.hand_numbers)
    ev_fold = cur_score / target
    bust_cost = 0 if me.has_insurance else cur_score
    ev_draw_raw = (
        p_bust * (cur_score - bust_cost)
        + (p_safe if n_unique < 5 else 0) * (cur_score + avg_safe)
        + (p_safe if n_unique == 5 else 0) * (cur_score + avg_safe + 15)
        + (rem.bonus_flat / denom) * (cur_score + BONUS_FLAT_AVG)
        + (rem.bonus_double / denom) * (cur_score + hand_sum)
        + ((rem.insurance + rem.exile + rem.triple) / denom) * (cur_score + 5)
    )
    ev_draw = ev_draw_raw / target
    ev_diff = ev_draw - ev_fold  # 正值倾向摸，负值倾向跑
    feats.append(ev_draw)
    feats.append(ev_fold)
    feats.append(ev_diff)

    # --- 6 翻冲刺（2 个）---
    feats.append(1.0 if n_unique == 5 else 0.0)
    p_six = p_safe if n_unique == 5 else 0.0
    feats.append(p_six)

    # --- 分数位置 / 威胁（5 个）---
    feats.append(min(1.0, (me.total_score + cur_score) / target))   # 离目标进度
    others_total = [p.total_score for p in state.players if p.index != my_idx]
    max_other = max(others_total) if others_total else 0
    min_other = min(others_total) if others_total else 0
    n_active_opp = sum(1 for p in state.players
                        if p.index != my_idx and p.is_active)
    # 我相对领先 / 落后
    lead = (me.total_score - max_other) / target  # 正：我领先；负：落后
    feats.append(max(-1.0, min(1.0, lead)))
    # 距离 target 还差多少（负值=已经超过）
    feats.append(max(-1.0, min(1.0, (target - me.total_score - cur_score) / target)))
    # 我现在排第几（0~3，归一化）
    rank = sum(1 for s in others_total if s > me.total_score) / 3.0
    feats.append(rank)
    # 还活跃的对手数
    feats.append(n_active_opp / 3.0)

    # --- 手牌结构（3 个）---
    if me.hand_numbers:
        feats.append(max(me.hand_numbers) / 12.0)
        feats.append(min(me.hand_numbers) / 12.0)
        spread = (max(me.hand_numbers) - min(me.hand_numbers)) / 12.0
        feats.append(spread)
    else:
        feats.extend([0.0, 0.0, 0.0])

    return np.array(feats, dtype=np.float32)
