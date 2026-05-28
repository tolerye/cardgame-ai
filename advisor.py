"""决策助手：输入当前局面，给出 draw / fold 建议 + 详细概率分析。

适用场景：你在线下/别的平台玩这个游戏，用这个工具实时辅助决策。

用法：
    python3 advisor.py
    python3 advisor.py --hand "3 5 8" --bonus 10 --total 45 --target 200
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

from agents.ev_agent import EVAgent
from agents.expectimax_agent import ExpectimaxAgent
from game import GameConfig
from game.cards import (BONUS_FLAT_COUNT, BONUS_DOUBLE_COUNT, NUMBER_COUNTS,
                        SKILL_PER_KIND, DeckCounts)
from game.state import GameState, PlayerState, PlayerStatus


USE_COLOR = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
def C(c, t):
    return f"\033[{c}m{t}\033[0m" if USE_COLOR else t
BOLD = lambda t: C("1", t)
DIM = lambda t: C("2", t)
GREEN = lambda t: C("32", t)
YELLOW = lambda t: C("33", t)
RED = lambda t: C("31", t)
CYAN = lambda t: C("36", t)
MAGENTA = lambda t: C("35", t)


# 中文别名 → 内部名 (+加分值)
# 加分牌：+2/+4/+6/+8/+10 各 1 张；翻倍牌单独 3 张
SPECIAL_ALIASES = {
    # +N 加分牌（具体面值，但 DeckCounts 只记总张数，所以只用 'bonus_flat'）
    "+2": "bonus_flat", "+4": "bonus_flat", "+6": "bonus_flat",
    "+8": "bonus_flat", "+10": "bonus_flat",
    "加分": "bonus_flat", "flat": "bonus_flat",
    "翻倍": "bonus_double", "double": "bonus_double", "x2": "bonus_double", "×2": "bonus_double",
    "保险": "insurance", "ins": "insurance", "shield": "insurance",
    "放逐": "exile", "exile": "exile",
    "三连": "triple", "triple": "triple",
}


def parse_seen(s: str) -> Tuple[Dict[int, int], Dict[str, int]]:
    """解析 '1 2 7 +10 保险' → ({1:1, 2:1, 7:1}, {bonus_flat:1, insurance:1})"""
    nums: Dict[int, int] = {}
    specials: Dict[str, int] = {}
    if not s.strip():
        return nums, specials
    for tok in s.replace(",", " ").split():
        tok = tok.strip()
        if not tok:
            continue
        if tok.lstrip("-").isdigit() and not tok.startswith("+"):
            v = int(tok)
            if 0 <= v <= 12:
                nums[v] = nums.get(v, 0) + 1
                continue
        key = SPECIAL_ALIASES.get(tok.lower())
        if key is None:
            print(RED(f"  ⚠ 未识别的牌：'{tok}'，跳过"))
            continue
        specials[key] = specials.get(key, 0) + 1
    return nums, specials


def parse_hand(s: str) -> List[int]:
    out = []
    for tok in s.replace(",", " ").split():
        try:
            v = int(tok)
            if 0 <= v <= 12:
                out.append(v)
        except ValueError:
            pass
    return out


def build_remaining(my_hand: List[int], my_bonus_flat_count: int,
                     my_bonus_double_count: int, my_insurance: bool,
                     seen_nums: Dict[int, int], seen_specials: Dict[str, int]) -> DeckCounts:
    rem = DeckCounts.full()
    # 减去自己手上的数字
    for v in my_hand:
        rem.numbers[v] -= 1
    rem.bonus_flat -= my_bonus_flat_count
    rem.bonus_double -= my_bonus_double_count
    if my_insurance:
        rem.insurance -= 1
    # 减去已见
    for v, c in seen_nums.items():
        rem.numbers[v] -= c
    for k, c in seen_specials.items():
        setattr(rem, k, getattr(rem, k) - c)
    # 防负值并提示
    issues = []
    for v in range(13):
        if rem.numbers[v] < 0:
            issues.append(f"数字{v}多算了 {-rem.numbers[v]} 张")
            rem.numbers[v] = 0
    for k in ("bonus_flat", "bonus_double", "insurance", "exile", "triple"):
        if getattr(rem, k) < 0:
            issues.append(f"{k} 多算了 {-getattr(rem, k)} 张")
            setattr(rem, k, 0)
    if issues:
        print(RED("  ⚠ 输入与牌库矛盾：" + "；".join(issues)))
    return rem


def analyze(state: GameState, my_idx: int) -> dict:
    me = state.players[my_idx]
    counts = state.remaining
    total = counts.total()
    cur_score = sum(me.hand_numbers) + me.bonus_flat_total
    hand_set = me.unique_numbers
    n_unique = len(hand_set)

    if total == 0:
        return {"error": "牌库已空"}

    bust_count = sum(counts.numbers[v] for v in hand_set)
    safe_count = sum(c for v, c in counts.numbers.items() if v not in hand_set)
    p_bust = bust_count / total
    p_safe = safe_count / total

    # 安全数字命中后期望加分
    if safe_count > 0:
        avg_safe_value = sum(v * c for v, c in counts.numbers.items() if v not in hand_set) / safe_count
    else:
        avg_safe_value = 0

    # 6 翻概率（仅当当前 5 张不同）
    p_six = p_safe if n_unique == 5 else 0
    p_continue_with_safe = p_safe if n_unique < 5 else 0

    p_flat = counts.bonus_flat / total
    p_double = counts.bonus_double / total
    p_ins = counts.insurance / total
    p_exile = counts.exile / total
    p_triple = counts.triple / total

    # EV 估算
    ev_fold = float(cur_score)
    if me.has_insurance:
        # 爆牌不归零（消耗保险）
        ev_bust_cost = 0  # 视为不变
    else:
        ev_bust_cost = -cur_score

    ev_draw = 0.0
    ev_draw += p_bust * (cur_score + ev_bust_cost)  # 爆牌路径
    ev_draw += p_continue_with_safe * (cur_score + avg_safe_value)  # 安全（未触六翻）
    ev_draw += p_six * (cur_score + avg_safe_value + 15)  # 6 翻终结
    ev_draw += p_flat * (cur_score + 10)
    ev_draw += p_double * (cur_score + sum(me.hand_numbers))  # 翻倍数字总和
    ev_draw += (p_ins + p_exile + p_triple) * (cur_score + 5)  # 技能粗估 +5

    # 调 expectimax 拿"最优应对后的 EV"
    expectimax = ExpectimaxAgent(depth=3)
    expectimax_decision = expectimax.choose_action(state, my_idx)

    return {
        "total_remaining": total,
        "n_unique": n_unique,
        "cur_score": cur_score,
        "p_bust": p_bust * 100,
        "p_safe": p_safe * 100,
        "p_six": p_six * 100,
        "avg_safe_value": avg_safe_value,
        "p_flat": p_flat * 100,
        "p_double": p_double * 100,
        "p_ins": p_ins * 100,
        "p_exile": p_exile * 100,
        "p_triple": p_triple * 100,
        "ev_draw": ev_draw,
        "ev_fold": ev_fold,
        "expectimax_decision": expectimax_decision,
    }


def render_analysis(a: dict, state: GameState, my_idx: int) -> None:
    if "error" in a:
        print(RED(f"\n  ⚠ {a['error']}"))
        return
    me = state.players[my_idx]
    target = state.config.target_score
    print()
    print(BOLD(CYAN("━━━━━━━━━━━━━━ 分析结果 ━━━━━━━━━━━━━━")))
    print(f"  剩余牌库：{a['total_remaining']} 张")
    print(f"  你的手牌：{sorted(me.hand_numbers)}（{a['n_unique']} 个不同数字）")
    print(f"  本局已得：{a['cur_score']} 分    保险：{'有' if me.has_insurance else '无'}")
    print(f"  你的总分：{me.total_score} / {target}")
    print()

    # 概率分解
    print(BOLD("  📊 抽下一张的概率分解："))
    bust_color = GREEN if a["p_bust"] < 15 else YELLOW if a["p_bust"] < 30 else RED
    bust_pct = bust_color(f"{a['p_bust']:>5.1f}%")
    bust_note = "有保险，免一次" if me.has_insurance else f"本局 {a['cur_score']} 分归零"
    print(f"    💥 爆牌：     {bust_pct}    （{bust_note}）")
    if a["n_unique"] < 5:
        safe_pct = GREEN(f"{a['p_safe']:>5.1f}%")
        print(f"    ✅ 安全数字： {safe_pct}    平均 +{a['avg_safe_value']:.1f} 分")
    elif a["n_unique"] == 5:
        six_pct_str = f"{a['p_six']:>5.1f}%"
        six_pct = BOLD(YELLOW(six_pct_str)) if a["p_six"] > 50 else YELLOW(six_pct_str)
        print(f"    🚀 6 翻终结：{six_pct}    +15 奖励 + 强制全场结算")
    print(f"    🟢 +10 加分： {a['p_flat']:>5.1f}%")
    print(f"    🟢 翻倍：     {a['p_double']:>5.1f}%    +{sum(me.hand_numbers)} 分")
    print(f"    🛡 保险：     {a['p_ins']:>5.1f}%")
    print(f"    🚷 放逐：     {a['p_exile']:>5.1f}%")
    print(f"    ⚡ 三连：     {a['p_triple']:>5.1f}%")
    print()

    # EV
    print(BOLD("  📈 期望值对比："))
    diff = a["ev_draw"] - a["ev_fold"]
    draw_label = f"EV(继续摸) = {a['ev_draw']:.1f}"
    fold_label = f"EV(跑路) = {a['ev_fold']:.1f}"
    if diff > 1:
        print(f"    {GREEN(draw_label)}   {DIM(fold_label)}")
        print(f"    差值：{GREEN(f'+{diff:.1f}')}")
    elif diff < -1:
        print(f"    {DIM(draw_label)}   {GREEN(fold_label)}")
        print(f"    差值：{RED(f'{diff:.1f}')}")
    else:
        print(f"    {YELLOW(draw_label)}   {YELLOW(fold_label)}")
        print(f"    差值：{YELLOW(f'{diff:+.1f}')}（接近，看风险偏好）")
    print()

    # 推荐
    decision = a["expectimax_decision"]
    if decision == "draw":
        rec = BOLD(GREEN("🎲 继续摸牌"))
    else:
        rec = BOLD(YELLOW("🔒 跑路（锁分）"))
    print(BOLD(f"  💡 Expectimax 建议：{rec}"))

    # 特殊提示
    if me.total_score + a["cur_score"] >= target:
        print(BOLD(GREEN(f"  🏆 警告：现在跑路即可获胜！锁定 {a['cur_score']} 分赢比赛。")))
    if a["n_unique"] == 5 and a["p_six"] > 50:
        print(BOLD(YELLOW(f"  🚀 5 张冲刺机会：6 翻概率 {a['p_six']:.1f}%，建议拼一把")))
    if a["p_bust"] > 35:
        print(BOLD(RED(f"  ⚠ 高风险：爆牌概率 {a['p_bust']:.1f}%，强烈建议跑路")))


def interactive() -> None:
    print(BOLD(CYAN("━━━━━━━━━━━━━━ 决策助手 ━━━━━━━━━━━━━━")))
    print(DIM("  输入当前局面，按回车跳过的字段使用默认值\n"))

    hand_str = input("  我的手牌（数字，空格分隔，如 '3 5 8'）: ").strip()
    my_hand = parse_hand(hand_str)

    bonus_str = input("  我的加分总分（默认 0）: ").strip() or "0"
    my_bonus = int(bonus_str)
    # 加分牌张数：用户给加分总分，从 +2/+4/+6/+8/+10 推算最少需要几张能凑出
    # （advisor 只关心剩余牌库，所以用合理估算即可）
    my_bonus_flat_count = 0
    remaining_bonus = my_bonus
    for v in [10, 8, 6, 4, 2]:
        if remaining_bonus >= v:
            my_bonus_flat_count += 1
            remaining_bonus -= v
    my_bonus_double_count = 0
    if input("  抽过翻倍牌？(y/n) [n]: ").strip().lower() in ("y", "yes"):
        my_bonus_double_count = 1
    my_insurance = input("  有保险？(y/n) [n]: ").strip().lower() in ("y", "yes")

    total_str = input("  我的总分（默认 0）: ").strip() or "0"
    my_total = int(total_str)

    target_str = input("  目标分（默认 200）: ").strip() or "200"
    target = int(target_str)

    n_str = input("  总玩家数（默认 4）: ").strip() or "4"
    n_players = int(n_str)

    # 简化版：不细问对手手牌，只问已见牌总览
    print(DIM("\n  已见过的其他牌（含其他玩家手牌+弃牌堆，可不输）"))
    print(DIM("  格式示例：'1 1 5 7 +10 保险 翻倍'"))
    seen_str = input("  > ").strip()
    seen_nums, seen_specials = parse_seen(seen_str)

    # 构建状态
    rem = build_remaining(my_hand, my_bonus_flat_count, my_bonus_double_count,
                           my_insurance, seen_nums, seen_specials)

    cfg = GameConfig(num_players=n_players, target_score=target)
    me = PlayerState(index=0, total_score=my_total)
    me.hand_numbers = list(my_hand)
    me.bonus_flat_total = my_bonus
    me.has_insurance = my_insurance
    me.status = PlayerStatus.ACTIVE
    others = [PlayerState(index=i, total_score=0, status=PlayerStatus.ACTIVE)
              for i in range(1, n_players)]
    state = GameState(config=cfg, players=[me] + others, remaining=rem)

    a = analyze(state, my_idx=0)
    render_analysis(a, state, my_idx=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="决策助手：建议 draw 或 fold")
    parser.add_argument("--hand", help="我的手牌，如 '3 5 8'")
    parser.add_argument("--bonus", type=int, default=0, help="我的加分总分")
    parser.add_argument("--double", action="store_true", help="抽过翻倍牌")
    parser.add_argument("--insurance", action="store_true", help="有保险")
    parser.add_argument("--total", type=int, default=0, help="我的总分")
    parser.add_argument("--target", type=int, default=200, help="目标分")
    parser.add_argument("--players", type=int, default=4, help="玩家数")
    parser.add_argument("--seen", default="", help="已见过的牌，如 '1 5 +10 保险'")
    args = parser.parse_args()

    if args.hand is None:
        interactive()
        return

    my_hand = parse_hand(args.hand)
    # 同 interactive 推算 flat 张数
    my_bonus_flat_count = 0
    remaining = args.bonus
    for v in [10, 8, 6, 4, 2]:
        if remaining >= v:
            my_bonus_flat_count += 1
            remaining -= v
    my_bonus_double_count = 1 if args.double else 0
    seen_nums, seen_specials = parse_seen(args.seen)
    rem = build_remaining(my_hand, my_bonus_flat_count, my_bonus_double_count,
                           args.insurance, seen_nums, seen_specials)
    cfg = GameConfig(num_players=args.players, target_score=args.target)
    me = PlayerState(index=0, total_score=args.total)
    me.hand_numbers = list(my_hand)
    me.bonus_flat_total = args.bonus
    me.has_insurance = args.insurance
    me.status = PlayerStatus.ACTIVE
    others = [PlayerState(index=i, total_score=0, status=PlayerStatus.ACTIVE)
              for i in range(1, args.players)]
    state = GameState(config=cfg, players=[me] + others, remaining=rem)
    a = analyze(state, my_idx=0)
    render_analysis(a, state, my_idx=0)


if __name__ == "__main__":
    main()
