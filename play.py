"""终端交互版游戏。

你是 P0，其余三个座位是 AI（默认 expectimax / ev / greedy）。

    python3 play.py
    python3 play.py --opponents expectimax,expectimax,expectimax   # 最强对手
    python3 play.py --opponents neural,expectimax,ev               # 含训练好的 NN
    python3 play.py --target 100                                   # 短局
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List

from agents import EVAgent, ExpectimaxAgent, GreedyAgent, RandomAgent
from agents.base import BaseAgent
from game import GameConfig, GameEngine
from game.cards import CardKind
from game.state import GameState, PlayerStatus


USE_COLOR = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"


def C(code: str, text: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


BOLD = lambda t: C("1", t)
DIM = lambda t: C("2", t)
GREEN = lambda t: C("32", t)
YELLOW = lambda t: C("33", t)
RED = lambda t: C("31", t)
CYAN = lambda t: C("36", t)
BLUE = lambda t: C("34", t)
MAGENTA = lambda t: C("35", t)


# 状态翻译
STATUS_CN = {
    PlayerStatus.ACTIVE: "进行中",
    PlayerStatus.FOLDED: "已跑路",
    PlayerStatus.BUSTED: "已爆牌",
    PlayerStatus.EXILED: "被放逐",
}

# 技能牌中文
KIND_CN = {
    CardKind.EXILE: "放逐",
    CardKind.TRIPLE: "三连",
    CardKind.INSURANCE: "保险",
}

# 引擎 log 行翻译（按关键字）
def translate_log(line: str) -> str:
    """Translate engine log strings to Chinese.
    Format: "R{n}|P{i}: EVENT details" → "R{n}|P{i}: 中文描述"."""
    # SIX-BURST! lock=NN
    m = re.match(r"^(R\d+\|P\d+): SIX-BURST! lock=(\d+)$", line)
    if m:
        return f"{m.group(1)}: 6翻了！锁分 {m.group(2)}"
    # FOLD lock=NN
    m = re.match(r"^(R\d+\|P\d+): FOLD lock=(\d+)$", line)
    if m:
        return f"{m.group(1)}: 跑路 锁分 {m.group(2)}"
    # DRAW N
    m = re.match(r"^(R\d+\|P\d+): DRAW (\d+)$", line)
    if m:
        return f"{m.group(1)}: 抽到数字 {m.group(2)}"
    # BUST_AVOIDED on N
    m = re.match(r"^(R\d+\|P\d+): BUST_AVOIDED on (\d+)$", line)
    if m:
        return f"{m.group(1)}: 重复 {m.group(2)} 但保险消耗（免爆）"
    # BUST on N
    m = re.match(r"^(R\d+\|P\d+): BUST on (\d+)$", line)
    if m:
        return f"{m.group(1)}: 爆牌！抽到重复的 {m.group(2)}"
    # BONUS+10
    m = re.match(r"^(R\d+\|P\d+): BONUS\+(\d+)$", line)
    if m:
        return f"{m.group(1)}: 抽到加分牌 +{m.group(2)}"
    # DOUBLE +N
    m = re.match(r"^(R\d+\|P\d+): DOUBLE \+(\d+)$", line)
    if m:
        return f"{m.group(1)}: 抽到翻倍牌 当前数字总和翻倍 +{m.group(2)}"
    # INSURANCE+
    m = re.match(r"^(R\d+\|P\d+): INSURANCE\+$", line)
    if m:
        return f"{m.group(1)}: 获得保险"
    # INSURANCE -> PN
    m = re.match(r"^(R\d+\|P\d+): INSURANCE -> P(\d+)$", line)
    if m:
        return f"{m.group(1)}: 强制赠送保险给 P{m.group(2)}"
    # INSURANCE wasted
    if "INSURANCE wasted" in line:
        return line.replace("INSURANCE wasted (no targets)", "保险作废（无可送对象）")
    # EXILE -> PN lock=NN
    m = re.match(r"^(R\d+\|P\d+): EXILE -> P(\d+) lock=(\d+)$", line)
    if m:
        return f"{m.group(1)}: 放逐 P{m.group(2)} 锁分 {m.group(3)}"
    if "EXILE wasted" in line:
        return line.replace("EXILE wasted (no targets)", "放逐作废（无可选目标）")
    # TRIPLE -> PN
    m = re.match(r"^(R\d+\|P\d+): TRIPLE -> P(\d+)$", line)
    if m:
        return f"{m.group(1)}: 三连 -> P{m.group(2)} 强制摸 3 张"
    if "TRIPLE wasted" in line:
        return line.replace("TRIPLE wasted (no targets)", "三连作废（无可选目标）")
    return line


# --------------------------------------------------------------- 人类玩家
class HumanAgent(BaseAgent):
    name = "human"

    def __init__(self) -> None:
        self._last_log_len = 0
        self._round_seen = 0
        self._my_idx = 0
        # 缓存本局所有事件（按玩家分类），跨回合可见
        self._round_events: List[str] = []
        self._round_events_round = 0

    def choose_action(self, state: GameState, my_idx: int) -> str:
        self._my_idx = my_idx
        self._catch_up_logs(state)
        self._render(state, my_idx)
        while True:
            try:
                raw = input(BOLD(f"\n  P{my_idx}（你）→ 选择 [d=要牌, f=跑路, q=退出]: ")).strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\n  退出。")
                sys.exit(0)
            if raw in ("d", "draw", "要牌", ""):
                return "draw"
            if raw in ("f", "fold", "跑路"):
                return "fold"
            if raw in ("q", "quit", "exit", "退出"):
                print("  再见。")
                sys.exit(0)
            print(RED("  ?  请输入 d 或 f"))

    def choose_skill_target(self, state: GameState, my_idx: int, kind: CardKind) -> int:
        self._my_idx = my_idx
        self._catch_up_logs(state)
        if kind == CardKind.INSURANCE:
            candidates = [p for p in state.players if p.is_active and p.index != my_idx]
        else:
            candidates = [p for p in state.players if p.is_active]
        if not candidates:
            return my_idx
        prompt_cn = {
            CardKind.EXILE: "选择强制跑路的对象（锁定本局得分）。可选自己（自我锁分）。",
            CardKind.TRIPLE: "选择强制摸 3 张的对象。可选自己（如已 5 张不同数字想冲 6 翻）。",
            CardKind.INSURANCE: "你已有保险，必须把这张转送给一名其他玩家",
        }.get(kind, "选择目标")
        print(MAGENTA(f"\n  ★ {KIND_CN.get(kind, kind.value)}：{prompt_cn}"))
        for i, p in enumerate(candidates):
            ins = " [保险]" if p.has_insurance else ""
            hand = " ".join(str(n) for n in sorted(p.hand_numbers))
            mark = BOLD(GREEN(" ← 你")) if p.index == my_idx else ""
            print(f"    {i}：P{p.index}{mark}  总分={p.total_score:>3}  本局={p.current_round_score():>3}  "
                  f"手牌=[{hand}]{ins}")
        while True:
            try:
                raw = input("    输入编号：").strip()
                idx = int(raw)
                if 0 <= idx < len(candidates):
                    return candidates[idx].index
            except (ValueError, KeyboardInterrupt, EOFError):
                pass
            print(RED("    无效编号"))

    def _catch_up_logs(self, state: GameState) -> None:
        new = state.log[self._last_log_len:]
        self._last_log_len = len(state.log)

        # 跨局了：上局的尾部事件（如三连结果、6翻、其他玩家最后行动）应该回放出来
        if state.round_number != self._round_seen:
            prev_round = self._round_seen
            if prev_round > 0:
                tail = [l for l in new if l.startswith(f"R{prev_round}|")]
                if tail:
                    print()
                    print(DIM(f"  📜 第 {prev_round} 局后续："))
                    for line in tail:
                        self._print_event(line, self._my_idx)
            # 重置本局
            self._round_seen = state.round_number
            self._round_events = []

        # 把属于当前局的事件加入历史
        cur_prefix = f"R{state.round_number}|"
        for line in new:
            if line.startswith(cur_prefix):
                self._round_events.append(line)

    def _render(self, state: GameState, my_idx: int) -> None:
        target = state.config.target_score
        print()
        print(BOLD(CYAN(f"━━━━━━━━━━━━━ 第 {state.round_number} 局 ━━━━━━━━━━━━━")))
        # 本局事件历史（最多 12 条，老的折叠）
        if self._round_events:
            print(DIM("  📜 本局已发生："))
            shown = self._round_events
            hidden = len(shown) - 12
            if hidden > 0:
                print(DIM(f"     … 上面省略 {hidden} 条早期事件 …"))
                shown = shown[-12:]
            for line in shown:
                self._print_event(line, my_idx)
            print()

        # 当前局面
        print(BOLD("  📊 当前局面：") + DIM(f"  目标 {target}    剩余牌库 {state.remaining.total()} 张"))
        for p in state.players:
            self._render_player(state, p, my_idx)
        print()

    def _print_event(self, line: str, my_idx: int) -> None:
        cn = translate_log(line)
        # 提取 P 编号
        m = re.match(r"R\d+\|P(\d+):\s*(.*)", cn)
        if m:
            pid = int(m.group(1))
            event = m.group(2)
            tag = BOLD(GREEN("★ 你 ")) if pid == my_idx else f"P{pid}    "
            # 给事件本身上色
            event_colored = self._colorize_event_text(event)
            print(f"     {tag} {event_colored}")
        else:
            print(f"     {cn}")

    @staticmethod
    def _colorize_event_text(text: str) -> str:
        if "爆牌" in text:
            return RED(text)
        if "6翻了" in text:
            return BOLD(YELLOW(text))
        if "放逐" in text or "三连" in text:
            return MAGENTA(text)
        if "保险" in text:
            return BLUE(text)
        if "加分" in text or "翻倍" in text:
            return GREEN(text)
        if "跑路" in text:
            return DIM(text)
        if "抽到数字" in text:
            return CYAN(text)
        return text

    def _render_player(self, state: GameState, p, my_idx: int) -> None:
        target = state.config.target_score
        is_me = p.index == my_idx
        emoji = self._status_emoji(p.status)
        marker = BOLD(GREEN(" ★ 你")) if is_me else "     "
        bar = self._bar(p.total_score, target, 16)

        if is_me:
            hand_str = ", ".join(str(n) for n in sorted(p.hand_numbers)) or DIM("(空)")
            ins = GREEN(" [保险]") if p.has_insurance else ""
            bonus = f" 加分+{p.bonus_flat_total}" if p.bonus_flat_total else ""
            cur = sum(p.hand_numbers) + p.bonus_flat_total
            print(f"  {emoji} P{p.index}{marker}  {self._status_str(p.status)}  总分 {BOLD(str(p.total_score)):>3} {bar}")
            print(f"           手牌 [{BOLD(hand_str)}]{bonus}{ins}    本局 +{BOLD(str(cur))}")
            self._render_my_hint(state, p)
        else:
            hand_str = ",".join(str(n) for n in sorted(p.hand_numbers))
            ins = GREEN(" [保险]") if p.has_insurance else ""
            line2 = f"手牌[{hand_str}]{ins}" if hand_str else DIM("(空手)")
            print(f"  {emoji} P{p.index}{marker}  {self._status_str(p.status)}  总分 {p.total_score:>3} {bar}  本局 +{p.current_round_score():>2}  {line2}")

    @staticmethod
    def _status_emoji(status: PlayerStatus) -> str:
        return {
            PlayerStatus.ACTIVE: "🎲",
            PlayerStatus.FOLDED: "🔒",
            PlayerStatus.BUSTED: "💥",
            PlayerStatus.EXILED: "🚷",
        }.get(status, "  ")

    @staticmethod
    def _status_str(status: PlayerStatus) -> str:
        s = STATUS_CN.get(status, status.value)
        if status == PlayerStatus.ACTIVE:
            return GREEN(f"{s:<4}")
        if status == PlayerStatus.FOLDED:
            return DIM(f"{s:<4}")
        if status == PlayerStatus.BUSTED:
            return RED(f"{s:<4}")
        return MAGENTA(f"{s:<4}")

    @staticmethod
    def _bar(score: int, target: int, width: int) -> str:
        filled = min(width, int(score / target * width))
        return "[" + GREEN("█" * filled) + DIM("░" * (width - filled)) + "]"

    @staticmethod
    def _render_my_hint(state: GameState, me) -> None:
        rem = state.remaining
        total = max(rem.total(), 1)
        bust_count = sum(rem.numbers[v] for v in me.unique_numbers)
        bust_p = bust_count / total * 100
        cur = sum(me.hand_numbers) + me.bonus_flat_total
        unique = len(me.unique_numbers)

        hint_parts = []
        if me.total_score + cur >= state.config.target_score:
            hint_parts.append(BOLD(GREEN("🏆 现在跑路即可获胜！")))
        else:
            color = GREEN if bust_p < 15 else YELLOW if bust_p < 30 else RED
            hint_parts.append(f"爆牌概率 {color(f'{bust_p:.0f}%')}")

            if unique == 5:
                safe_count = sum(c for v, c in rem.numbers.items() if v not in me.unique_numbers)
                safe_p = safe_count / total * 100
                hint_parts.append(BOLD(YELLOW(f"6翻成功率 {safe_p:.0f}% 🚀")))
            hint_parts.append(f"跑路锁 {cur} 分")

        print(f"           💡 " + "    ".join(hint_parts))


# --------------------------------------------------------------- AI 注册
OPP_REGISTRY: Dict[str, callable] = {
    "expectimax": lambda: ExpectimaxAgent(),
    "ev": lambda: EVAgent(),
    "greedy": lambda: GreedyAgent(),
    "greedy20": lambda: GreedyAgent(fold_at=20),
    "random": lambda: RandomAgent(),
}

OPP_NAME_CN = {
    "expectimax": "期望值递归（最强 40%）",
    "ev": "启发式期望（35%）",
    "greedy": "固定阈值贪心（24%）",
    "greedy20": "激进贪心",
    "random": "随机",
    "neural": "神经网络（21%）",
    "neural_mcts": "神经网络+搜索",
}


def _maybe_register_neural() -> None:
    try:
        from agents.neural_agent import NeuralAgent
        OPP_REGISTRY["neural"] = lambda: NeuralAgent("model.pt")
        OPP_REGISTRY["neural_mcts"] = lambda: NeuralAgent("model.pt", use_mcts=True, n_simulations=80)
    except Exception:
        pass


def main() -> None:
    _maybe_register_neural()
    parser = argparse.ArgumentParser(description="终端版牌局对战")
    parser.add_argument("--opponents", default="expectimax,ev,greedy",
                        help="3 个 AI 对手，逗号分隔")
    parser.add_argument("--target", type=int, default=200, help="比赛目标分（默认 200）")
    parser.add_argument("--list", action="store_true", help="列出所有可用 AI 类型")
    args = parser.parse_args()

    if args.list:
        print("可用 AI 类型：")
        for name in OPP_REGISTRY:
            cn = OPP_NAME_CN.get(name, "")
            print(f"  {name:<14} {cn}")
        return

    names = [n.strip() for n in args.opponents.split(",")]
    if len(names) != 3:
        print(f"必须正好 3 个对手，给了 {len(names)} 个")
        sys.exit(1)
    for n in names:
        if n not in OPP_REGISTRY:
            print(f"未知对手 '{n}' — 用 --list 查看可用类型")
            sys.exit(1)

    print(BOLD(CYAN("━" * 60)))
    print(BOLD(CYAN("  终端版牌局对战")))
    print(BOLD(CYAN("━" * 60)))
    print(f"  目标分：{BOLD(str(args.target))}")
    print(f"  你    ：{GREEN('P0（人类）')}")
    for i, n in enumerate(names, start=1):
        cn = OPP_NAME_CN.get(n, "")
        print(f"  对手 P{i}：{n}  {DIM(cn)}")
    print()
    print(DIM("  规则：抽不重复的数字凑手牌，6 个不同 = 6翻了 (+15 分)"))
    print(DIM("        重复数字 = 爆牌（除非有保险），跑路任意时刻锁定本局分"))
    print(DIM("        加分牌：+10 / 翻倍当前数字总和"))
    print(DIM("        技能牌：保险（免一次爆牌）/ 放逐（强制对手跑路）/ 三连（强制对手摸 3 张）"))
    print()

    cfg = GameConfig(num_players=4, target_score=args.target)
    agents = [HumanAgent()] + [OPP_REGISTRY[n]() for n in names]
    engine = GameEngine(cfg, agents)
    winner = engine.play_match()

    print()
    print(BOLD(CYAN("━━━━━━━━━━━━━━━━ 比赛结束 ━━━━━━━━━━━━━━━━")))
    rank = sorted(engine.state.players, key=lambda p: -p.total_score)
    medals = ["🥇", "🥈", "🥉", "  "]
    for medal, p in zip(medals, rank):
        you = BOLD(GREEN(" ★ 你")) if p.index == 0 else ""
        agent_name = "你" if p.index == 0 else names[p.index - 1]
        print(f"  {medal}  P{p.index}（{agent_name}）{you}：总分 {BOLD(str(p.total_score))}")
    print()
    if winner == 0:
        print(BOLD(GREEN("  恭喜，你赢了！🎉")))
    else:
        print(BOLD(RED(f"  {names[winner - 1]} 获胜。")))


if __name__ == "__main__":
    main()
